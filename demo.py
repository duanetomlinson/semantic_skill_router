#!/usr/bin/env python3
"""
demo.py — Semantic Skill Router Demo

Interactive CLI that takes natural language commands,
routes them through Redis Vector Search, and executes
the matched handler function on the Pi.

Tier 1: Direct skill match via vector search (~61ms)
Tier 2: Agent mode for complex multi-step tasks (--agent prefix)

Usage:
    python demo.py setup     # Create index + register skills (run once)
    python demo.py            # Interactive command loop
    python demo.py bench      # Run benchmark suite
"""

import sys
import json
import re
import time
from registry import SkillRegistry
from handlers import SKILL_CATALOG

# Calibration: approximate duration_ms per cm/degree at default speeds
# Forward/backward: ~10cm in 500ms at speed 30 -> 50ms per cm
MS_PER_CM = 50
# Turn: ~90 degrees in 400ms at speed 40 -> ~4.4ms per degree
MS_PER_DEGREE = 4.44

# Handlers that accept distance (cm) vs rotation (degrees)
DISTANCE_HANDLERS = {
    "robot.drive_forward", "robot.drive_backward", "robot.drive_forward_slow",
}
ROTATION_HANDLERS = {
    "robot.turn_left", "robot.turn_right",
}


def parse_modifiers(raw):
    """Parse --X (value) and -r X (repeat) from the command string.
    Returns (clean_query, value, repeat_count)."""
    value = None
    repeat = 1
    # Extract -r N
    r_match = re.search(r'-r\s+(\d+)', raw)
    if r_match:
        repeat = int(r_match.group(1))
        raw = raw[:r_match.start()] + raw[r_match.end():]
    # Extract --N
    v_match = re.search(r'--(\d+)', raw)
    if v_match:
        value = int(v_match.group(1))
        raw = raw[:v_match.start()] + raw[v_match.end():]
    return raw.strip(), value, repeat


def apply_value_override(handler_name, default_args, value):
    """Override duration_ms based on --X value (cm or degrees)."""
    if value is None:
        return default_args
    args = dict(default_args)
    if handler_name in DISTANCE_HANDLERS:
        args["duration_ms"] = int(value * MS_PER_CM)
    elif handler_name in ROTATION_HANDLERS:
        args["duration_ms"] = int(value * MS_PER_DEGREE)
    return args


def colorize(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t): return colorize(t, 32)
def yellow(t): return colorize(t, 33)
def red(t): return colorize(t, 31)
def cyan(t): return colorize(t, 36)
def dim(t): return colorize(t, 90)
def magenta(t): return colorize(t, 35)


def setup(registry: SkillRegistry):
    """One-time setup: create index and register all skills."""
    print("\n" + cyan("=" * 55))
    print(cyan("  Redis Skill Router — Setup"))
    print(cyan("=" * 55))
    print()
    print("Step 1: Creating vector index...")
    registry.create_index(drop_existing=True)
    print()
    print("Step 2: Embedding & registering skills...")
    registry.register_skills()
    print()
    print(green("Done. Run `python demo.py` to start.\n"))


def run_agent_mode(registry, initial_goal=None):
    """Run the Tier 2 agent in multi-turn conversation mode."""
    from agent import start_agent_session, save_learned_skill

    session = start_agent_session(print_fn=print)
    if session is None:
        return

    print()
    print(magenta("=" * 55))
    print(magenta("  Agent Mode (Tier 2)"))
    print(magenta("=" * 55))
    print(dim("  Multi-turn conversation. Type 'end' to exit."))
    print(dim("  The agent can see, move, and navigate autonomously."))
    print()

    last_goal = initial_goal

    # If initial goal provided, send it immediately
    if initial_goal:
        print(cyan(f"  > {initial_goal}"))
        try:
            response = session.chat(initial_goal)
            print(f"  {response}\n")
        except Exception as e:
            print(red(f"  Agent error: {e}\n"))

    # Conversation loop
    while True:
        try:
            user_input = input(cyan("  agent> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("end", "exit", "quit"):
            break

        last_goal = last_goal or user_input

        try:
            response = session.chat(user_input)
            print(f"  {response}\n")
        except Exception as e:
            print(red(f"  Agent error: {e}\n"))

    # Session ended
    session._restore_sigint()
    steps = session.tool_call_count
    print()
    print(dim(f"  Session ended. {steps} tool calls made."))

    # Offer to save as learned skill
    if steps > 0 and last_goal:
        try:
            save = input(magenta("  Save as learned skill? [Y/n] ")).strip().lower()
            if save in ("y", "yes", ""):
                # Get the agent's strategy distillation
                print(dim("  Generating strategy prompt..."))
                strategy = session.get_strategy_prompt(last_goal)
                print(dim(f"  Strategy: {strategy[:100]}..."))

                # Ask for skill name
                suggested = last_goal.lower().replace(" ", "_")[:30]
                name = input(dim(f"  Skill name [{suggested}]: ")).strip()
                name = name or suggested

                # Ask for description
                desc = input(dim(f"  Description [{last_goal}]: ")).strip()
                desc = desc or last_goal

                # Save to Redis
                skill = save_learned_skill(
                    registry, name, desc, strategy, last_goal
                )
                print(green(f"  Saved skill '{name}' with learned strategy."))
                print(dim(f"  Next time, say '{last_goal}' and it routes via Tier 1."))
            else:
                print(dim("  Skipped."))
        except (EOFError, KeyboardInterrupt):
            print()


def interactive(registry: SkillRegistry):
    """Main interactive loop with Tier 1 (vector search) and Tier 2 (agent)."""
    print("\n" + cyan("=" * 55))
    print(cyan("  Redis Skill Router — Interactive Mode"))
    print(cyan("=" * 55))
    print(dim("  Type natural language commands. 'quit' to exit."))
    print(dim("  'top3' prefix shows top 3 matches (e.g., 'top3 check temp')"))
    print(dim("  '--agent' prefix for complex tasks (e.g., '--agent find the door')"))
    print(dim("  '--X'  set distance in cm or degrees (e.g., 'drive forward --20')"))
    print(dim("  '-r X' repeat command X times (e.g., 'turn left -r 4')"))
    print()

    # Warm up the model on first run
    print(dim("  Warming up embedding model..."))
    t0 = time.perf_counter()
    registry.embed("warmup")
    print(dim(f"  Model ready in {(time.perf_counter() - t0)*1000:.0f}ms\n"))

    while True:
        try:
            raw = input(cyan("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + dim("Goodbye."))
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print(dim("Goodbye."))
            break

        # ── Agent mode ───────────────────────────────────────
        if raw.strip() == "--agent":
            run_agent_mode(registry)
            continue
        if raw.startswith("--agent "):
            goal = raw[8:].strip()
            run_agent_mode(registry, initial_goal=goal)
            continue

        # ── Parse modifiers (--X value, -r X repeat) ────────
        query, value, repeat = parse_modifiers(raw)

        # ── Show top 3 mode ─────────────────────────────────
        show_top3 = query.lower().startswith("top3 ")
        query = query[5:].strip() if show_top3 else query

        # ── Search ───────────────────────────────────────────
        t_start = time.perf_counter()

        if show_top3:
            matches = registry.search(query, top_k=3)
            t_search = (time.perf_counter() - t_start) * 1000
            print(dim(f"\n  Search: {t_search:.1f}ms"))
            for i, m in enumerate(matches):
                bar = "#" * int(m["similarity"] * 20)
                print(
                    f"  {i+1}. {m['similarity']:.3f} {dim(bar)} "
                    f"{m['name']} -> {m['handler']}"
                )
            print()
            continue

        # ── Route to best match ──────────────────────────────
        match = registry.route(query, threshold=0.48)
        t_search = (time.perf_counter() - t_start) * 1000

        if match is None:
            print(red(f"  No confident match ({t_search:.1f}ms search)"))
            print(dim(f"  Tip: try '--agent {query}' for complex tasks\n"))
            continue

        handler_name = match["handler"]
        similarity = match["similarity"]

        # ── Check for learned skill (agent strategy) ────────
        if handler_name.startswith("agent.learned:"):
            from agent import run_learned_skill
            skill_name = handler_name.split(":", 1)[1]
            # Load the strategy prompt from Redis
            import json as _json
            raw_doc = registry.r.get(f"skill:{skill_name}")
            if raw_doc:
                doc = _json.loads(raw_doc)
                agent_prompt = doc.get("agent_prompt", "")
                print(f"\n  -> {cyan('learned skill')}: {skill_name}")
                print(dim(f"  similarity: {similarity:.3f}  |  search: {t_search:.1f}ms"))
                print(dim(f"  Running learned strategy...\n"))
                run_learned_skill(agent_prompt, skill_name, print_fn=print)
            else:
                print(red(f"  Learned skill data not found: {skill_name}\n"))
            continue

        # ── Dispatch ─────────────────────────────────────────
        handler_fn = SKILL_CATALOG.get(handler_name)
        if handler_fn is None:
            print(red(f"  Handler not found: {handler_name}\n"))
            continue

        # Apply --X override (cm or degrees -> duration_ms)
        args = apply_value_override(
            handler_name, match.get("default_args", {}), value
        )

        # Show what we're doing
        mod_info = ""
        if value and handler_name in DISTANCE_HANDLERS:
            mod_info += f"  {value}cm"
        elif value and handler_name in ROTATION_HANDLERS:
            mod_info += f"  {value}deg"
        if repeat > 1:
            mod_info += f"  x{repeat}"

        print(f"\n  -> {match['name']} -> {cyan(handler_name)}{dim(mod_info)}")
        print(dim(f"  similarity: {similarity:.3f}  |  search: {t_search:.1f}ms"))

        # Execute (with repeat)
        t_exec_start = time.perf_counter()
        results = []
        try:
            for i in range(repeat):
                r = handler_fn(**args)
                results.append(r)
                if repeat > 1 and i < repeat - 1:
                    time.sleep(0.1)  # small gap between repeats
        except Exception as e:
            results.append({"error": str(e)})
        t_exec = (time.perf_counter() - t_exec_start) * 1000

        print(dim(f"  exec: {t_exec:.0f}ms  |  total: {t_search + t_exec:.0f}ms"))
        if len(results) == 1:
            print(f"  {json.dumps(results[0])}")
        else:
            print(f"  {repeat}x -> {json.dumps(results[-1])}")
        print()


def benchmark(registry: SkillRegistry):
    """Run a test suite of queries and show routing accuracy + timing."""
    print("\n" + cyan("=" * 55))
    print(cyan("  Redis Skill Router — Benchmark"))
    print(cyan("=" * 55))

    # Warm up
    registry.embed("warmup")

    test_cases = [
        # (query, expected_handler)
        # ── System skills ──────────────────────────────────
        ("how hot is the CPU",                   "system.cpu_temp"),
        ("what's the processor temperature",     "system.cpu_temp"),
        ("is it overheating",                    "system.cpu_temp"),
        ("check RAM usage",                      "system.memory"),
        ("is memory running low",                "system.memory"),
        ("how much storage do I have",           "system.disk"),
        ("disk space remaining",                 "system.disk"),
        ("how long has this been running",       "system.uptime"),
        ("when did the system start",            "system.uptime"),
        ("what machine is this",                 "system.info"),
        ("show me the hostname",                 "system.info"),
        ("what's my IP",                         "system.network"),
        ("list network interfaces",              "system.network"),
        ("what's eating all the CPU",            "system.processes"),
        ("show running tasks",                   "system.processes"),
        ("what time is it",                      "system.timestamp"),
        ("today's date",                         "system.timestamp"),
        ("test internet connectivity",           "system.ping"),
        ("ping google DNS",                      "system.ping"),
        # ── Robot movement skills ──────────────────────────
        ("drive forward",                        "robot.drive_forward"),
        ("go backwards",                         "robot.drive_backward"),
        ("spin to the left",                     "robot.turn_left"),
        ("rotate right",                         "robot.turn_right"),
        ("stop moving",                          "robot.stop"),
        ("inch forward slowly",                  "robot.drive_forward_slow"),
        # ── Robot sensor skills ────────────────────────────
        ("check for obstacles",                  "robot.read_ultrasonic"),
        ("how much battery is left",             "robot.read_battery"),
        # ── Robot camera skills ────────────────────────────
        ("take a photo",                         "robot.capture_image"),
        ("look to the right",                    "robot.pan_camera_right"),
        ("tilt camera up",                       "robot.tilt_camera_up"),
        ("center the camera",                    "robot.camera_center"),
        # ── Robot LED skills ───────────────────────────────
        ("turn on the lights",                   "robot.led_on"),
        ("lights off",                           "robot.led_off"),
        # ── Edge cases — should NOT match (-> agent tier) ──
        ("tell me a joke",                       None),
        ("what's the meaning of life",           None),
        ("find the kitchen",                     None),
        ("patrol the room",                      None),
        ("go to the charging station",           None),
    ]

    correct = 0
    total_search_ms = 0
    print()

    for query, expected in test_cases:
        t0 = time.perf_counter()
        match = registry.route(query, threshold=0.48)
        elapsed = (time.perf_counter() - t0) * 1000
        total_search_ms += elapsed

        got = match["handler"] if match else None
        ok = got == expected
        if ok:
            correct += 1

        status = green("OK") if ok else red("FAIL")
        sim = f"{match['similarity']:.3f}" if match else "---  "
        got_display = got or dim("(none)")
        print(f"  {status} {sim}  {elapsed:6.1f}ms  {query[:40]:<40}  -> {got_display}")

    print()
    pct = correct / len(test_cases) * 100
    avg = total_search_ms / len(test_cases)
    color = green if pct >= 90 else yellow if pct >= 70 else red
    print(f"  Accuracy: {color(f'{correct}/{len(test_cases)} ({pct:.0f}%)')}")
    print(f"  Avg search latency: {avg:.1f}ms")
    print()


if __name__ == "__main__":
    registry = SkillRegistry(redis_url="redis://localhost:6379")

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "setup":
            setup(registry)
        elif cmd == "bench":
            benchmark(registry)
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python demo.py [setup|bench]")
    else:
        interactive(registry)
