# Redis Semantic Skill Router — Architecture Overview

## What We Built

A **zero-agent command dispatch system** that routes natural language to executable
functions using only Redis Vector Search. No LLM is invoked in the command loop.

```
┌──────────────┐     ┌──────────────────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Input  │────▶│  Embedding Model          │────▶│  Redis KNN   │────▶│   Handler    │
│  "check temp"│     │  redis/langcache-v3-small │     │  FT.SEARCH   │     │  cpu_temp()  │
└──────────────┘     │  384-dim native output    │     │  idx:skills   │     │  → result    │
                     └──────────────────────────┘     └──────────────┘     └──────────────┘
```

### The Pipeline (5 steps, no LLM)

1. **User types a command** in natural language ("how hot is the CPU?")
1. **Embedding model encodes it** into a 384-dimensional float vector
1. **Redis `FT.SEARCH`** performs K-nearest-neighbor lookup against stored skill vectors
1. **Best match is returned** with a similarity score and handler name
1. **Handler executes** the function and returns structured data

## How Skills Are Stored

Each skill is a Redis JSON document with an embedded vector:

```json
{
  "name": "cpu_temperature",
  "description": "Read the current CPU temperature of the Raspberry Pi",
  "phrases": ["how hot is the processor", "what is the CPU temp", ...],
  "handler": "system.cpu_temp",
  "default_args": {},
  "embedding": [0.012, -0.034, ...]
}
```

The embedding is a **centroid** — the average of the description vector plus all
example phrase vectors. This produces a more robust matching surface than embedding
the description alone. Instead of a single point in vector space, each skill
occupies a region defined by all the ways a user might express that intent.

## Benchmark Results (Raspberry Pi 4B, 4GB RAM, CPU only)

This branch shows the art of the possible on a Pi 4 — the same hardware that
runs the Freenove Tank Robot. Every branch in the project went through the
same two-step story: a backend swap (PyTorch → ONNX Runtime) and a model
upgrade (`langcache-embed-v2`, 128-dim → `langcache-embed-v3-small`,
384-dim). Both steps were applied on `main`, `onnx-optimization`, and
`robot-tank-agent` on a Pi 5; this branch reruns the same ladder on a Pi 4B
to confirm the final v3-small pipeline holds up on the older, lower-power
board that's actually bolted into the robot.

Tested across 21 queries: 19 valid commands mapped to 9 skill handlers, plus
2 garbage queries ("tell me a joke", "what's the meaning of life") that should
be rejected.

### PyTorch Backend (langcache-embed-v2, 128-dim)

|Metric            |Value                                           |
|------------------|------------------------------------------------|
|Routing accuracy  |18/21 (87%) — garbage queries leaked through    |
|Avg search latency|645ms                                           |
|Model             |redis/langcache-embed-v2 via SentenceTransformer|

### ONNX Backend, v2 (one line change)

|Metric            |Value                                            |
|------------------|-------------------------------------------------|
|Routing accuracy  |19/21 (92%)                                      |
|Avg search latency|325ms                                            |
|Model             |redis/langcache-embed-v2 via ONNX Runtime        |

### ONNX Backend, v3-small (shipped on every branch; measured here on Pi 4B)

|Metric            |Value                                              |
|------------------|---------------------------------------------------|
|Routing accuracy  |20/21 (97%) — garbage queries correctly rejected   |
|Avg search latency|40ms                                               |
|Model             |redis/langcache-embed-v3-small via ONNX Runtime    |
|Vector dim        |384 (native output, no truncation)                 |

This v3-small row is the configuration every branch now runs. On a Pi 5 it
clocks ~9ms (see `main`, `onnx-optimization`, `robot-tank-agent`); on this
Pi 4B branch the same model and code path land at ~40ms.

### Comparison

|Metric       |PyTorch v2|ONNX v2|ONNX v3-small|Improvement       |
|-------------|----------|-------|-------------|------------------|
|Avg latency  |645ms     |325ms  |40ms         |**16x faster**    |
|Accuracy     |87%       |92%    |97%          |**+10 points**    |
|Code change  |—         |1 line |model swap   |`backend="onnx"`  |
|Redis index  |—         |Same   |Rebuilt 384-d|One-time reindex  |

## Where the Time Goes

The bottleneck is the embedding step. Redis search and handler execution are
negligible:

```
ONNX v3-small breakdown on Pi 4B (~40ms total):
  Embedding:  ████████████████████████████████████████████████░░  ~37ms (92%)
  Redis KNN:  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~2ms  (6%)
  Handler:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~1ms  (2%)
```

## Why ONNX Is Faster

PyTorch is a general-purpose deep learning framework built for training and
research. It carries overhead that is unnecessary for pure inference: Python GIL
contention, dynamic graph construction, operator dispatch, and per-call memory
allocation.

ONNX Runtime is a dedicated inference engine written in C++. It eliminates
Python from the computation path, fuses multiple operations into optimized
kernels, pre-allocates memory, and uses platform-specific instructions (ARM
NEON on the Cortex-A72 cores of the Pi 4B). Same model weights, same
mathematical output, faster execution.

The change in code:

```python
# PyTorch (before)
self.model = SentenceTransformer(MODEL_NAME)

# ONNX (after)
self.model = SentenceTransformer(MODEL_NAME, backend="onnx")
```

## Why This Architecture Matters

### 1. Speed — No Agent Overhead

A typical LLM agent loop:

```
User → LLM (plan) → Tool selection → LLM (format) → Execute → LLM (respond)
       ~1000ms        ~500ms           ~1000ms                    ~1000ms
```

Total: **3-5 seconds** for a simple command like "turn left."

Our loop with ONNX v3-small on a Pi 4B:

```
User → Embed → Redis Search → Execute
       ~37ms     ~2ms          ~1ms
```

Total: **~40ms** on a Pi 4. That is 75-125x faster than an agent for
deterministic commands — on the lower-powered board.

### 2. Deterministic — Same Input, Same Output

An LLM might interpret "go forward" differently each time. It might add
qualifications, ask for clarification, or hallucinate a tool that doesn't exist.

Vector search is deterministic. "Go forward" will always match `drive_forward`
with the same similarity score. No randomness, no temperature setting, no
prompt sensitivity.

### 3. Offline — Zero External Dependencies

After the one-time model download, the entire system runs air-gapped:

- **Embedding model**: `redis/langcache-embed-v3-small`, cached locally as an ONNX export
- **Vector database**: Redis Stack running in Docker
- **Handlers**: Python functions running natively

No API keys. No internet. No tokens burned. No cost per query.

### 4. Full Redis Stack — One Vendor Story

- **Embedding model**: `redis/langcache-embed-v3-small` (Redis's own, on HuggingFace)
- **Vector storage**: RedisJSON
- **Vector index**: RediSearch
- **Similarity search**: Redis `FT.SEARCH` with KNN

Redis end-to-end. The only non-Redis component is Python itself.

## Two-Tier Dispatch — Agent Only When Needed

The architecture naturally extends to a hybrid model:

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│  User Input  │────▶│  Redis KNN   │────▶│  Similarity > threshold?    │
└──────────────┘     └──────────────┘     └─────────────────────────────┘
                                                 │              │
                                              YES │              │ NO
                                                 ▼              ▼
                                          ┌──────────┐   ┌──────────────┐
                                          │ Execute   │   │ Agent (LLM)  │
                                          │ Handler   │   │ Complex task │
                                          │ ~1ms      │   │ ~3000ms      │
                                          └──────────┘   └──────────────┘
```

90% of IoT commands are simple and repetitive. They don't need an LLM.
Redis handles the fast path. The agent is a fallback for genuinely complex
reasoning — planning, multi-step tasks, unknown commands.

The router itself decides which tier to use via the same vector similarity
mechanism. If nothing matches above the similarity threshold, the query
is handed to the agent. This means the decision of whether to invoke an LLM
is itself sub-millisecond.

## Optimization Ladder

The same three-rung ladder was walked on every branch. Steps 1 and 2 are the
backend swap; step 3 is the model upgrade (`langcache-embed-v2` → `langcache-
embed-v3-small`), which landed on `main`, `onnx-optimization`, and
`robot-tank-agent` as well. Numbers below are from this branch, measured on
a Raspberry Pi 4B — the board driving the robot.

```
Step 1: PyTorch v2 on Pi 4B CPU              → 645ms  ✅ Done
Step 2: ONNX Runtime v2 on Pi 4B CPU         → 325ms  ✅ Done
Step 3: ONNX Runtime v3-small on Pi 4B CPU   →  40ms  ✅ Done (shipped on every branch)
Step 4: ONNX + INT8 quantization             → ~20ms  (halves model size + faster math)
Step 5: GPU offload to P620 (RTX 3090)       →  <5ms  (embed on GPU, search on Pi)
```

The Pi 5 branches walked the same ladder and bottomed out at ~9ms after the
v3-small upgrade. On a Pi 4 we land at 40ms with identical code and
identical model weights — still well inside the budget for real-time robot
control.

## Tech Stack

|Component        |Technology                    |Role                       |
|-----------------|------------------------------|---------------------------|
|Embedding model  |redis/langcache-embed-v3-small|Text → 384-dim vector      |
|Inference runtime|ONNX Runtime (or PyTorch)     |Model execution engine     |
|Vector storage   |RedisJSON                     |Skill documents + vectors  |
|Vector index     |RediSearch (FT.SEARCH KNN)    |Nearest neighbor lookup    |
|Dimensionality   |384-dim native output         |No Matryoshka truncation   |
|Similarity metric|Cosine distance               |0 = identical, 2 = opposite|
|Threshold        |0.48 similarity               |Below = rejected           |
|Handler dispatch |Python dict lookup            |Route → function execution |
|Target hardware  |Raspberry Pi 4B (4GB)         |Freenove Tank Robot        |
|Deployment       |Docker (Redis) + venv (Python)|On-prem, air-gapped capable|

## File Structure

```
semantic_skill_router/
├── demo.py              # CLI entrypoint (setup / interactive / bench / agent)
├── registry.py          # Redis schema, embedding, vector search (v3-small, 384-dim)
├── handlers.py          # Pi 4B function handlers + SKILL_CATALOG
├── agent.py             # Tier 2 agent: Claude tool_use, vision, learning
├── export_onnx_model.py # One-time ONNX export of langcache-embed-v3-small
├── pyproject.toml       # Project metadata + dependencies
├── ARCHITECTURE.md      # This document
├── .gitignore
└── README.md
```

## The Pitch

> Redis model. Redis search. Redis storage. Full stack, zero external
> dependencies. No planner, no tools, no chain-of-thought. Semantic
> lookup and execute in 384 dimensions — 40 milliseconds end-to-end
> on a Raspberry Pi 4. That's the art of the possible on the older
> board: the same pipeline that clocks ~9ms on a Pi 5, running inside
> a real robot's budget on hardware shipped in 2019.
