#!/usr/bin/env python3
"""
demo.py — Semantic Skill Router Demo

Interactive CLI that takes natural language commands,
routes them through Redis Vector Search, and executes
the matched handler function on the Pi.

Usage:
    python demo.py setup     # Create index + register skills (run once)
    python demo.py            # Interactive command loop
    python demo.py bench      # Run benchmark suite
"""

import sys
import json
import time
from registry import SkillRegistry
from handlers import SKILL_CATALOG


def colorize(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t): return colorize(t, 32)
def yellow(t): return colorize(t, 33)
def red(t): return colorize(t, 31)
def cyan(t): return colorize(t, 36)
def dim(t): return colorize(t, 90)


def setup(registry: SkillRegistry):
    """One-time setup: create index and register all skills."""
    print("\n" + cyan("═" * 55))
    print(cyan("  Redis Skill Router — Setup"))
    print(cyan("═" * 55))
    print()
    print("Step 1: Creating vector index...")
    registry.create_index(drop_existing=True)
    print()
    print("Step 2: Embedding & registering skills...")
    registry.register_skills()
    print()
    print(green("✓ Setup complete. Run `python demo.py` to start.\n"))


def interactive(registry: SkillRegistry):
    """Main interactive loop."""
    print("\n" + cyan("═" * 55))
    print(cyan("  Redis Skill Router — Interactive Mode"))
    print(cyan("═" * 55))
    print(dim("  Type natural language commands. 'quit' to exit."))
    print(dim("  'top3' prefix shows top 3 matches (e.g., 'top3 check temp')"))
    print()

    # Warm up the model on first run
    print(dim("  Warming up embedding model + Redis pipeline + Redis pipeline..."))
    t0 = time.perf_counter()
    #registry.embed("warmup") -- replacing with loop below for faster warmup
    for phrase in ["warmup query one", "check the system temperature", "show network info"]:
        registry.route(phrase)

    print(dim(f"  Model ready in {(time.perf_counter() - t0)*1000:.0f}ms\n"))

    while True:
        try:
            raw = input(cyan("❯ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + dim("Goodbye."))
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print(dim("Goodbye."))
            break

        # ── Show top 3 mode ─────────────────────────────────
        show_top3 = raw.lower().startswith("top3 ")
        query = raw[5:].strip() if show_top3 else raw

        # ── Search ───────────────────────────────────────────
        t_start = time.perf_counter()
        
        if show_top3:
            matches = registry.search(query, top_k=3)
            t_search = (time.perf_counter() - t_start) * 1000
            print(dim(f"\n  Search: {t_search:.1f}ms"))
            for i, m in enumerate(matches):
                bar = "█" * int(m["similarity"] * 20)
                print(
                    f"  {i+1}. {m['similarity']:.3f} {dim(bar)} "
                    f"{m['name']} → {m['handler']}"
                )
            print()
            continue

        # ── Route to best match ──────────────────────────────
        match = registry.route(query, threshold=0.48)
        t_search = (time.perf_counter() - t_start) * 1000

        if match is None:
            print(red(f"  ✗ No confident match ({t_search:.1f}ms search)\n"))
            continue

        handler_name = match["handler"]
        similarity = match["similarity"]

        # ── Dispatch ─────────────────────────────────────────
        handler_fn = SKILL_CATALOG.get(handler_name)
        if handler_fn is None:
            print(red(f"  ✗ Handler not found: {handler_name}\n"))
            continue

        t_exec_start = time.perf_counter()
        try:
            result = handler_fn(**match.get("default_args", {}))
        except Exception as e:
            result = {"error": str(e)}
        t_exec = (time.perf_counter() - t_exec_start) * 1000
        t_total = t_search + t_exec

        # ── Output ───────────────────────────────────────────
        print(f"\n  {green('→')} {match['name']} → {cyan(handler_name)}")
        print(dim(f"  similarity: {similarity:.3f}  |  "
                   f"search: {t_search:.1f}ms  |  "
                   f"exec: {t_exec:.1f}ms  |  "
                   f"total: {t_total:.1f}ms"))
        print()
        print(json.dumps(result, indent=2))
        print()


def benchmark(registry: SkillRegistry):
    """Run a test suite of queries and show routing accuracy + timing."""
    print("\n" + cyan("═" * 55))
    print(cyan("  Redis Skill Router — Benchmark"))
    print(cyan("═" * 55))

    # Warm up
    registry.embed("warmup")

    test_cases = [
        # (query, expected_handler)
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
        # Edge cases — should these match anything?
        ("tell me a joke",                       None),
        ("what's the meaning of life",           None),
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

        status = green("✓") if ok else red("✗")
        sim = f"{match['similarity']:.3f}" if match else "—    "
        got_display = got or dim("(none)")
        print(f"  {status} {sim}  {elapsed:6.1f}ms  {query[:40]:<40}  → {got_display}")

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
