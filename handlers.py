"""
handlers.py — Skill handlers that run on the Raspberry Pi 4B.
Each function returns a dict so results are structured and loggable.

System handlers: CPU temp, memory, disk, uptime, etc.
Robot handlers:  Motor, servo, ultrasonic, camera, LED (Freenove Tank Kit).
"""

import subprocess
import os
import sys
import platform
import json
import time
from datetime import datetime

# ── Freenove Library Path ──────────────────────────────────────
# The Freenove Tank Robot Kit Python modules live on the Pi at:
#   /home/nion/Freenove_Tank_Robot_Kit_for_Raspberry_Pi-main/Code/Server/
# Add to sys.path so we can import Motor, servo, Ultrasonic, Led, etc.
FREENOVE_PATH = os.path.expanduser(
    "~/Freenove_Tank_Robot_Kit_for_Raspberry_Pi-main/Code/Server"
)
if os.path.isdir(FREENOVE_PATH) and FREENOVE_PATH not in sys.path:
    sys.path.insert(0, FREENOVE_PATH)

# ── Lazy Hardware Imports ──────────────────────────────────────
# These only succeed on the Pi with Freenove libs installed.
# On dev machines they fail gracefully — handlers return stubs.
_hw_available = False
_motor = None
_servo = None
_ultrasonic = None
_led = None
_picam2 = None

def _init_hardware():
    """Initialize Freenove hardware modules (call once on Pi).
    Each component inits independently so one failure doesn't block others.
    Starts pigpiod if not already running (required for motor/servo PWM)."""
    global _hw_available, _motor, _servo, _ultrasonic, _led, _picam2
    if _hw_available:
        return True
    # Ensure pigpiod is running (Motor and Servo need it)
    import subprocess as _sp
    _sp.run(["sudo", "pigpiod"], capture_output=True, timeout=5)
    import time as _t
    _t.sleep(0.5)
    try:
        from Motor import PWM as motor_instance
        _motor = motor_instance
    except Exception:
        pass
    try:
        from servo import Servo
        _servo = Servo()
    except Exception:
        pass
    try:
        from Ultrasonic import Ultrasonic
        _ultrasonic = Ultrasonic()
    except Exception:
        pass
    try:
        from Led import Led
        _led = Led()
    except Exception:
        pass
    try:
        from picamera2 import Picamera2
        from libcamera import Transform
        _picam2 = Picamera2()
        config = _picam2.create_preview_configuration(
            main={"size": (640, 480)},
            transform=Transform(hflip=1, vflip=1),
        )
        _picam2.configure(config)
        _picam2.start()
    except Exception:
        pass
    _hw_available = _motor is not None
    return _hw_available


# ── System Handlers ────────────────────────────────────────────

def get_cpu_temp() -> dict:
    """Read the Pi's CPU temperature."""
    try:
        raw = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        celsius = int(raw) / 1000.0
        return {"temp_c": celsius, "temp_f": round(celsius * 9 / 5 + 32, 1)}
    except FileNotFoundError:
        return {"error": "thermal zone not found (not a Pi?)"}


def get_memory_usage() -> dict:
    """Return memory stats in MB."""
    import shutil
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                mem[parts[0].rstrip(":")] = int(parts[1]) // 1024  # KB → MB
    mem["UsedMB"] = mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)
    return mem


def get_disk_usage() -> dict:
    """Disk usage for root partition."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    gb = 1 << 30
    return {
        "total_gb": round(total / gb, 1),
        "used_gb": round(used / gb, 1),
        "free_gb": round(free / gb, 1),
        "percent_used": round(used / total * 100, 1),
    }


def get_uptime() -> dict:
    """System uptime."""
    raw = open("/proc/uptime").read().split()[0]
    seconds = float(raw)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return {"seconds": seconds, "human": f"{hours}h {minutes}m"}


def get_system_info() -> dict:
    """General system identification."""
    uname = platform.uname()
    return {
        "hostname": uname.node,
        "arch": uname.machine,
        "kernel": uname.release,
        "python": platform.python_version(),
    }


def get_network_info() -> dict:
    """IP addresses for all non-loopback interfaces."""
    result = subprocess.run(
        ["ip", "-j", "addr", "show"],
        capture_output=True, text=True
    )
    try:
        interfaces = json.loads(result.stdout)
        info = {}
        for iface in interfaces:
            name = iface.get("ifname", "")
            if name == "lo":
                continue
            addrs = [
                a["local"]
                for a in iface.get("addr_info", [])
                if a.get("family") == "inet"
            ]
            if addrs:
                info[name] = addrs
        return info
    except json.JSONDecodeError:
        return {"error": "could not parse ip output"}


def get_running_processes() -> dict:
    """Top 10 processes by CPU usage."""
    result = subprocess.run(
        ["ps", "aux", "--sort=-%cpu"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")[1:11]  # skip header, top 10
    procs = []
    for line in lines:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            procs.append({
                "user": parts[0],
                "pid": parts[1],
                "cpu": parts[2],
                "mem": parts[3],
                "command": parts[10][:80],
            })
    return {"top_processes": procs}


def get_timestamp() -> dict:
    """Current date/time on the Pi."""
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %B %d %Y — %I:%M %p"),
        "epoch": int(now.timestamp()),
    }


def ping_host(host: str = "8.8.8.8", count: int = 3) -> dict:
    """Ping a host and return latency stats."""
    result = subprocess.run(
        ["ping", "-c", str(count), host],
        capture_output=True, text=True, timeout=15
    )
    return {"host": host, "output": result.stdout.strip().split("\n")[-2:]}


# ── Robot Movement Handlers ─────────────────────────────────────
# Freenove Tank Robot Kit — Motor.py API:
#   PWM.setMotorModel(left_duty, right_duty)
#   Duty range: -4095 to 4095
#   Forward: (positive, positive)  Backward: (negative, negative)
#   Turn left: (positive, negative) Turn right: (negative, positive)

def drive_forward(speed: int = 30, duration_ms: int = 500) -> dict:
    """Drive the tank forward for a short distance (~10cm at default)."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "forward", "simulated": True}
    duty = int(speed / 100 * 4095)
    _motor.setMotorModel(duty, duty)
    time.sleep(duration_ms / 1000.0)
    _motor.setMotorModel(0, 0)  # safety stop
    return {"action": "forward", "speed": speed, "duration_ms": duration_ms}


def drive_backward(speed: int = 30, duration_ms: int = 500) -> dict:
    """Drive the tank backward for a short distance (~10cm at default)."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "backward", "simulated": True}
    duty = int(speed / 100 * 4095)
    _motor.setMotorModel(-duty, -duty)
    time.sleep(duration_ms / 1000.0)
    _motor.setMotorModel(0, 0)  # safety stop
    return {"action": "backward", "speed": speed, "duration_ms": duration_ms}


def turn_left(speed: int = 40, duration_ms: int = 400) -> dict:
    """Pivot the tank left ~90 degrees."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "turn_left", "simulated": True}
    duty = int(speed / 100 * 4095)
    # Right wheel forward, left wheel backward = pivot left
    _motor.setMotorModel(-duty, duty)
    time.sleep(duration_ms / 1000.0)
    _motor.setMotorModel(0, 0)  # safety stop
    return {"action": "turn_left", "speed": speed, "duration_ms": duration_ms}


def turn_right(speed: int = 40, duration_ms: int = 400) -> dict:
    """Pivot the tank right ~90 degrees."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "turn_right", "simulated": True}
    duty = int(speed / 100 * 4095)
    # Left wheel forward, right wheel backward = pivot right
    _motor.setMotorModel(duty, -duty)
    time.sleep(duration_ms / 1000.0)
    _motor.setMotorModel(0, 0)  # safety stop
    return {"action": "turn_right", "speed": speed, "duration_ms": duration_ms}


def stop() -> dict:
    """Immediately stop all motors."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "stop", "simulated": True}
    _motor.setMotorModel(0, 0)
    return {"action": "stop"}


def drive_forward_slow(speed: int = 15, duration_ms: int = 500) -> dict:
    """Creep forward ~5cm at low speed for precision navigation."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "forward_slow", "simulated": True}
    duty = int(speed / 100 * 4095)
    _motor.setMotorModel(duty, duty)
    time.sleep(duration_ms / 1000.0)
    _motor.setMotorModel(0, 0)  # safety stop
    return {"action": "forward_slow", "speed": speed, "duration_ms": duration_ms}


# ── Robot Sensor Handlers ──────────────────────────────────────

def read_ultrasonic() -> dict:
    """Read the front ultrasonic distance sensor, return cm."""
    if not _init_hardware():
        return {"error": "hardware not available", "sensor": "ultrasonic", "simulated": True}
    distance_cm = _ultrasonic.get_distance()
    return {"sensor": "ultrasonic", "distance_cm": round(distance_cm, 1)}


def read_battery() -> dict:
    """Read battery voltage level (stubbed — no ADC on this kit)."""
    # TODO: Wire up ADS1115 or INA219 for real battery readings
    return {"sensor": "battery", "voltage": None, "status": "no ADC — stub"}


# ── Robot Camera Handlers ──────────────────────────────────────
# Servo channel '0' = pan (horizontal), range 90-150, center=90
# Servo channel '1' = tilt (vertical), range 90-150, default=140

_servo_pan = 90    # center position
_servo_tilt = 140  # default position

def capture_image() -> dict:
    """Take a photo with the Pi camera, save to a timestamped file."""
    if not _init_hardware() or _picam2 is None:
        return {"error": "camera not available", "simulated": True}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"/tmp/robot_img_{timestamp}.jpg"
    _picam2.capture_file(filepath)
    return {"action": "capture_image", "filepath": filepath}


def pan_camera_left(step: int = 20) -> dict:
    """Pan the camera servo left (increase angle toward 150)."""
    global _servo_pan
    if not _init_hardware():
        return {"error": "hardware not available", "action": "pan_left", "simulated": True}
    _servo_pan = min(150, _servo_pan + step)
    _servo.setServoPwm('0', _servo_pan)
    return {"action": "pan_left", "angle": _servo_pan}


def pan_camera_right(step: int = 20) -> dict:
    """Pan the camera servo right (decrease angle toward 90)."""
    global _servo_pan
    if not _init_hardware():
        return {"error": "hardware not available", "action": "pan_right", "simulated": True}
    _servo_pan = max(90, _servo_pan - step)
    _servo.setServoPwm('0', _servo_pan)
    return {"action": "pan_right", "angle": _servo_pan}


def tilt_camera_up(step: int = 15) -> dict:
    """Tilt the camera servo up (decrease angle toward 90)."""
    global _servo_tilt
    if not _init_hardware():
        return {"error": "hardware not available", "action": "tilt_up", "simulated": True}
    _servo_tilt = max(90, _servo_tilt - step)
    _servo.setServoPwm('1', _servo_tilt)
    return {"action": "tilt_up", "angle": _servo_tilt}


def tilt_camera_down(step: int = 15) -> dict:
    """Tilt the camera servo down (increase angle toward 150)."""
    global _servo_tilt
    if not _init_hardware():
        return {"error": "hardware not available", "action": "tilt_down", "simulated": True}
    _servo_tilt = min(150, _servo_tilt + step)
    _servo.setServoPwm('1', _servo_tilt)
    return {"action": "tilt_down", "angle": _servo_tilt}


def camera_center() -> dict:
    """Return camera to center position (pan=90, tilt=140)."""
    global _servo_pan, _servo_tilt
    if not _init_hardware():
        return {"error": "hardware not available", "action": "camera_center", "simulated": True}
    _servo_pan = 90
    _servo_tilt = 140
    _servo.setServoPwm('0', _servo_pan)
    _servo.setServoPwm('1', _servo_tilt)
    return {"action": "camera_center", "pan": _servo_pan, "tilt": _servo_tilt}


# ── Robot Obstacle Avoidance ────────────────────────────────────

def obstacle_avoid(steps: int = 20) -> dict:
    """Drive forward while avoiding obstacles using the ultrasonic sensor.
    Runs for the given number of steps. Ctrl+C to stop early."""
    if not _init_hardware() or _ultrasonic is None:
        return {"error": "hardware not available", "action": "obstacle_avoid", "simulated": True}
    avoided = 0
    for i in range(steps):
        dist = _ultrasonic.get_distance()
        if dist > 0 and dist <= 5:
            # Too close — reverse
            _motor.setMotorModel(-1200, -1200)
            time.sleep(0.4)
            _motor.setMotorModel(0, 0)
            avoided += 1
        elif dist > 5 and dist <= 15:
            # Obstacle ahead — stop, turn right
            _motor.setMotorModel(0, 0)
            time.sleep(0.1)
            _motor.setMotorModel(1500, -1500)
            time.sleep(0.4)
            _motor.setMotorModel(0, 0)
            avoided += 1
        else:
            # Clear — drive forward
            _motor.setMotorModel(1200, 1200)
        time.sleep(0.2)
    _motor.setMotorModel(0, 0)
    return {"action": "obstacle_avoid", "steps": steps, "obstacles_avoided": avoided}


# ── Robot LED Handlers ─────────────────────────────────────────

def led_on(r: int = 255, g: int = 255, b: int = 255) -> dict:
    """Turn on LED strip with the given color (default: white)."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "led_on", "simulated": True}
    from rpi_ws281x import Color
    _led.colorWipe(_led.strip, Color(r, g, b), wait_ms=0)
    return {"action": "led_on", "color": {"r": r, "g": g, "b": b}}


def led_off() -> dict:
    """Turn off all LEDs."""
    if not _init_hardware():
        return {"error": "hardware not available", "action": "led_off", "simulated": True}
    from rpi_ws281x import Color
    _led.colorWipe(_led.strip, Color(0, 0, 0), wait_ms=0)
    return {"action": "led_off"}


# ── Skill Registry ──────────────────────────────────────────────
# Maps handler string names → callables.

SKILL_CATALOG = {
    # System skills
    "system.cpu_temp":        get_cpu_temp,
    "system.memory":          get_memory_usage,
    "system.disk":            get_disk_usage,
    "system.uptime":          get_uptime,
    "system.info":            get_system_info,
    "system.network":         get_network_info,
    "system.processes":       get_running_processes,
    "system.timestamp":       get_timestamp,
    "system.ping":            ping_host,
    # Robot movement
    "robot.drive_forward":      drive_forward,
    "robot.drive_backward":     drive_backward,
    "robot.turn_left":          turn_left,
    "robot.turn_right":         turn_right,
    "robot.stop":               stop,
    "robot.drive_forward_slow": drive_forward_slow,
    "robot.obstacle_avoid":     obstacle_avoid,
    # Robot sensors
    "robot.read_ultrasonic":    read_ultrasonic,
    "robot.read_battery":       read_battery,
    # Robot camera
    "robot.capture_image":      capture_image,
    "robot.pan_camera_left":    pan_camera_left,
    "robot.pan_camera_right":   pan_camera_right,
    "robot.tilt_camera_up":     tilt_camera_up,
    "robot.tilt_camera_down":   tilt_camera_down,
    "robot.camera_center":      camera_center,
    # Robot LEDs
    "robot.led_on":             led_on,
    "robot.led_off":            led_off,
}
