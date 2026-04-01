"""
handlers.py — Skill handlers that run on the Raspberry Pi 5.
Each function returns a dict so results are structured and loggable.
Swap these out later for real hardware calls (GPIO, tank, servos, etc.)
"""

import subprocess
import os
import platform
import json
from datetime import datetime


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


# ── Skill Registry ──────────────────────────────────────────────
# Maps handler string names → callables.
# This is what the router uses after vector search returns a match.

SKILL_CATALOG = {
    "system.cpu_temp":        get_cpu_temp,
    "system.memory":          get_memory_usage,
    "system.disk":            get_disk_usage,
    "system.uptime":          get_uptime,
    "system.info":            get_system_info,
    "system.network":         get_network_info,
    "system.processes":       get_running_processes,
    "system.timestamp":       get_timestamp,
    "system.ping":            ping_host,
}
