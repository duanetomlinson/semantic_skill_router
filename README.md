# Redis Semantic Skill Router

**Team:** Duane Tomlinson (solo)
**Build Tool:** Claude Code (Anthropic Claude Opus 4.6)

---

## What It Does

Routes natural language to executable functions using Redis Vector Search. No LLM in the routing loop. The entire pipeline — embed, search, execute — completes in ~40 milliseconds on a Raspberry Pi 4B.

> **This branch (`claude/redis-v3-pi4`) is the art-of-the-possible cut.** The other branches ran on a Pi 5; this one stays on the same Pi 4B that drives the Freenove Tank Robot, and upgrades the embedding model from `langcache-embed-v2` (128-dim) to `langcache-embed-v3-small` (384-dim). Same pipeline, older board, better accuracy, 8x faster than the v2 baseline on the same hardware.

The system controls a physical Freenove Tank Robot on a Raspberry Pi 4B. Motors, camera, ultrasonic sensors, servos, and LEDs are all dispatched through natural language matched against a skill library stored in Redis.

```
"drive the robot forward"  -->  Embed (~37ms)  -->  Redis KNN (~2ms)  -->  Handler executes
```

When a query doesn't match any known skill, a Tier 2 agent steps in. The agent uses the same Redis skill library as its tool catalog, solves the task, and saves the strategy back to Redis so the next identical request routes at vector-search speed.

---

## The Problem

LLM-based command dispatch is slow, expensive, and non-deterministic. Every "turn left" costs tokens, takes seconds, and might be interpreted differently each time. For IoT and robotics, where most commands are simple and repetitive, this overhead is unnecessary.

The question: **can Redis replace the LLM for the 90% of commands that don't need reasoning?**

---

## Three Layers of Redis

### 1. Intent Layer — Vector Search for Command Routing

Each skill is stored as a Redis document with a 384-dimensional embedding vector (generated from the skill's description and example phrases). When a user speaks, the query is embedded and matched via `FT.SEARCH KNN` in ~2ms on a Pi 4B.

This is the core of the system. Redis acts as the intent classifier — no LLM needed, no prompt engineering, no token cost. The same input always produces the same match.

| Key Pattern | What's Stored | Why |
|-------------|--------------|-----|
| `skill:drive_forward` | Description, example phrases, handler name, 384-dim vector | Semantic matching via cosine similarity |
| `skill:cpu_temperature` | System skill with the same structure | Unified pipeline for all command types |

**Redis features used:** FT.SEARCH KNN (vector similarity), RedisJSON (structured skill documents), Key-Value (fallback when modules aren't available)

### 2. Learning Layer — Agent Strategies as Cached Prompts

When the Tier 2 agent solves a complex task like "find the front door," it doesn't save the specific motor commands (turn right, go forward 30cm). Those are environment-specific and won't work next time.

Instead, it saves the **strategy** — a prompt describing the general approach: scan by rotating and photographing, identify doors in images, move toward them, verify by getting closer. This prompt is embedded and stored in Redis as a learned skill.

Next time someone says "find the front door," Redis matches it in milliseconds and spawns a mini-agent with the saved strategy. The agent adapts to the current environment using the learned approach.

| Key Pattern | What's Stored | Why |
|-------------|--------------|-----|
| `skill:find_front_door` | Description + `agent_prompt` (strategy) + vector | Learned behavior that adapts to new environments |

Redis becomes long-term memory. The agent teaches Redis what it learned, and Redis makes it available instantly.

### 3. Security Layer — User-Controlled Skill Library

The agent cannot modify the skill library on its own. It cannot write files, execute arbitrary code, or add tools. Skills are added by the user through an explicit approval flow:

1. Agent completes a task
2. System asks: "Save as learned skill?"
3. User reviews and confirms
4. Only then is the strategy embedded and stored in Redis

The skill catalog in Redis is the single source of truth for what the robot can do. The user controls what goes in. The agent is a consumer of skills, not an author of system capabilities.

This matters for any deployment where an LLM has access to physical actuators.

---

## How It Uses Redis

### Architecture

```
User Input
    |
    v
Embed query (ONNX Runtime, ~37ms on Pi 4B / ~9ms on Pi 5)
    |
    v
Redis FT.SEARCH KNN (~2ms on Pi 4B)
    |
    +---> Match found (similarity > 0.48)
    |         |
    |         +--> Standard skill --> Execute handler directly
    |         |    Total: ~40ms on Pi 4B
    |         |
    |         +--> Learned skill --> Spawn mini-agent with saved strategy
    |              Adapts to current environment
    |
    +---> No match --> Tier 2 LLM Agent
                         |
                         +--> Uses Redis skills as tool catalog
                         +--> Vision + autonomous navigation
                         +--> Offers to save strategy to Redis
```

### Redis Features

| Feature | Role |
|---------|------|
| **Vector Search (FT.SEARCH KNN)** | Intent classification. FLAT index, COSINE distance, 384-dim vectors. |
| **RedisJSON** | Structured skill storage with nested fields (embedding, phrases, default_args, agent_prompt). |
| **Key-Value** | Fallback vector search when RediSearch modules aren't loaded. Numpy handles the math. |

### Path to Redis Cloud

The system connects to Redis via a URL. Changing `redis://localhost:6379` to a Redis Cloud endpoint is a one-line change. This enables:

- **Shared skill libraries** across multiple devices
- **Central vector index** without compiling modules on ARM
- **Cross-device learning** — one robot learns a strategy, all robots gain it
- **Telemetry via Streams** for fleet-level analytics

---

## How It Uses AI

### Build Process

I designed the two-tier architecture, the three-layer Redis model (intent, learning, security), and the strategy-as-prompt learning approach. Claude Code (Opus 4.6) was my implementation partner — I directed the architecture and design decisions while Claude handled code generation, SSH deployment to the Pi, package installation, and hardware debugging. The collaboration covered everything from flashing the OS to wiring up GPIO pin mappings.

### Runtime

- **Embedding model** — `redis/langcache-embed-v3-small` (Redis's own model, Apache 2.0). Runs locally via ONNX Runtime. No API calls for Tier 1 routing.
- **Tier 2 Agent** — Claude API with native tool_use and vision. The agent sees through the robot's camera, reasons about its environment, and chains tool calls from the Redis skill catalog.
- **Strategy distillation** — After solving a task, the agent generates a reusable prompt describing its approach. This prompt is what gets saved to Redis — not the raw actions.

---

## Benchmarks

The v2 → ONNX → v3-small progression was walked end-to-end on the prior two
Pi 5 branches (`main` and `onnx-optimization`). By the time the robot tank
work landed, v3-small was the validated winner — so this branch was built
**on v3-small only**, no v2 baseline rerun on the Pi 4B.

### Raspberry Pi 4B (robot tank, 4GB RAM) — this branch

| Backend | Model | Avg Latency | Accuracy |
|---------|-------|------------|----------|
| **ONNX** | **langcache-embed-v3-small (384-dim)** | **40ms** | **97%** |

Only row measured. The v2 rungs were already climbed on Pi 5; rerunning them
on a slower board would have added no new information.

### Raspberry Pi 5 (prior branches, for reference)

| Backend | Model | Avg Latency | Accuracy |
|---------|-------|------------|----------|
| PyTorch | langcache-embed-v2 (128-dim) | 260ms | 90% |
| ONNX | langcache-embed-v2 (128-dim) | 61ms | 100% |
| PyTorch | langcache-embed-v3-small (384-dim) | ~20ms | 100% |
| ONNX | langcache-embed-v3-small (384-dim) | ~9ms | 100% |

The v3-small ONNX row is what this branch ships. ~9ms on a Pi 5; ~40ms on
this Pi 4B — same model, same code path, different silicon.

### Time Breakdown (Pi 4B, ONNX v3-small)

```
Embedding:  ████████████████████░░  ~37ms  (92%)
Redis KNN:  █░░░░░░░░░░░░░░░░░░░░░   ~2ms  (6%)
Handler:    ░░░░░░░░░░░░░░░░░░░░░░   <1ms  (2%)
```

---

## Branch Progression

The model and backend work was done on the Pi 5 branches. The robot tank
branch inherited the winning configuration — ONNX + v3-small — and moved
onto the Pi 4B without re-benchmarking the losers.

| Branch | Hardware | Backend | Models tested | Key Finding |
|--------|----------|---------|----------------|-------------|
| `main` | Pi 5 | PyTorch | v2 → **v3-small** | Zero-agent routing works. 260ms on v2, **~20ms after the v3-small upgrade**. |
| `onnx-optimization` | Pi 5 | ONNX Runtime | v2 → **v3-small** | One-line backend swap drops v2 from 260ms to 61ms. The v3-small upgrade then takes it to **~9ms**. |
| **`claude/redis-v3-pi4`** (robot tank) | **Pi 4B** | **ONNX Runtime** | **v3-small only** | **Art of the possible on the robot's own board.** Skipped the v2 rungs — already validated on Pi 5 — and jumped straight to v3-small. **40ms end-to-end, 97% accuracy, no Pi 5 required.** Plus 26 robot skills, a Tier 2 agent with vision, and a learning loop that saves strategies back to Redis. |

In short: the Pi 5 branches proved which backend and which model win. This
branch takes that answer, puts it on a Pi 4B, and builds the robot on top
of it.

---

## Quick Start

```bash
# Redis
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Install
git clone https://github.com/duanetomlinson/semantic_skill_router.git
cd semantic_skill_router
pip install redis numpy sentence-transformers onnxruntime 'optimum[onnxruntime]'

# Run
python demo.py setup    # Embed and index all skills
python demo.py           # Interactive mode
python demo.py bench     # Benchmark suite
```

### Interactive Commands

```
> drive forward              # Tier 1: 40ms route + execute
> drive forward --20         # 20cm forward
> turn left -r 4             # Repeat 4x = 360 degrees
> take a photo               # Capture from Pi camera
> avoid obstacles             # Autonomous ultrasonic navigation
> --agent                    # Enter Tier 2 conversation mode
> --agent find the front door # Agent navigates autonomously
```

---

## File Structure

```
semantic_skill_router/
├── demo.py              # CLI: setup, interactive, benchmark, agent mode
├── registry.py          # Redis vector search, 26 skill definitions
├── handlers.py          # Skill handlers + SKILL_CATALOG
├── agent.py             # Tier 2 agent: Claude tool_use, vision, learning
├── export_onnx_model.py # One-time ONNX model export
├── ARCHITECTURE.md      # Technical architecture document
├── pyproject.toml       # Dependencies
├── LICENSE
└── README.md
```
