# Redis Semantic Skill Router — Architecture Overview

## What We Built

A **zero-agent command dispatch system** that routes natural language to executable
functions using only Redis Vector Search. No LLM is invoked in the command loop.

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Input  │────▶│  Embedding Model     │────▶│  Redis KNN   │────▶│   Handler    │
│  "check temp"│     │  redis/langcache-v2  │     │  FT.SEARCH   │     │  cpu_temp()  │
└──────────────┘     │  128-dim Matryoshka  │     │  idx:skills   │     │  → result    │
                     └─────────────────────┘     └──────────────┘     └──────────────┘
```

### The Pipeline (5 steps, no LLM)

1. **User types a command** in natural language ("how hot is the CPU?")
1. **Embedding model encodes it** into a 128-dimensional float vector
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

## Benchmark Results (Raspberry Pi 5, CPU only)

Tested across 21 queries: 19 valid commands mapped to 9 skill handlers, plus
2 garbage queries ("tell me a joke", "what's the meaning of life") that should
be rejected.

### PyTorch Backend

|Metric            |Value                                           |
|------------------|------------------------------------------------|
|Routing accuracy  |19/21 (90%) — garbage queries leaked through    |
|Avg search latency|260.1ms                                         |
|Model             |redis/langcache-embed-v2 via SentenceTransformer|

### ONNX Backend (one line change)

|Metric            |Value                                            |
|------------------|-------------------------------------------------|
|Routing accuracy  |21/21 (100%) — garbage queries correctly rejected|
|Avg search latency|60.8ms                                           |
|Model             |redis/langcache-embed-v2 via ONNX Runtime        |

### Comparison

|Metric       |PyTorch|ONNX  |Improvement      |
|-------------|-------|------|-----------------|
|Avg latency  |260.1ms|60.8ms|**4.3x faster**  |
|Accuracy     |90%    |100%  |**Perfect**      |
|Code change  |—      |1 line|`backend="onnx"` |
|Model weights|Same   |Same  |Identical output |
|Redis index  |Same   |Same  |No rebuild needed|

## Where the Time Goes

The bottleneck is the embedding step. Redis search and handler execution are
negligible:

```
ONNX breakdown (~61ms total):
  Embedding:  ████████████████████████████████████████████████░░  ~57ms (93%)
  Redis KNN:  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~3ms  (5%)
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
NEON on the Pi 5). Same model weights, same mathematical output, faster
execution.

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

Our loop with ONNX:

```
User → Embed → Redis Search → Execute
       ~57ms     ~3ms          ~1ms
```

Total: **~61ms**. That is 50-80x faster than an agent for deterministic commands.

### 2. Deterministic — Same Input, Same Output

An LLM might interpret "go forward" differently each time. It might add
qualifications, ask for clarification, or hallucinate a tool that doesn't exist.

Vector search is deterministic. "Go forward" will always match `drive_forward`
with the same similarity score. No randomness, no temperature setting, no
prompt sensitivity.

### 3. Offline — Zero External Dependencies

After the one-time model download, the entire system runs air-gapped:

- **Embedding model**: `redis/langcache-embed-v2`, cached locally
- **Vector database**: Redis Stack running in Docker
- **Handlers**: Python functions running natively

No API keys. No internet. No tokens burned. No cost per query.

### 4. Full Redis Stack — One Vendor Story

- **Embedding model**: `redis/langcache-embed-v2` (Redis's own, on HuggingFace)
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

Each step is tested and proven. Steps 1 and 2 are complete.

```
Step 1: PyTorch on Pi 5 CPU             → 260ms   ✅ Done (main branch)
Step 2: ONNX Runtime on Pi 5 CPU        → 61ms    ✅ Done (onnx-optimization branch)
Step 3: ONNX + INT8 quantization        → ~30ms   (halves model size + faster math)
Step 4: GPU offload to P620 (RTX 3090)  → <5ms    (embed on GPU, search on Pi)
```

## Tech Stack

|Component        |Technology                    |Role                       |
|-----------------|------------------------------|---------------------------|
|Embedding model  |redis/langcache-embed-v2      |Text → 128-dim vector      |
|Inference runtime|ONNX Runtime (or PyTorch)     |Model execution engine     |
|Vector storage   |RedisJSON                     |Skill documents + vectors  |
|Vector index     |RediSearch (FT.SEARCH KNN)    |Nearest neighbor lookup    |
|Dimensionality   |128-dim Matryoshka truncation |Reduced from 768-dim       |
|Similarity metric|Cosine distance               |0 = identical, 2 = opposite|
|Threshold        |0.48 similarity               |Below = rejected           |
|Handler dispatch |Python dict lookup            |Route → function execution |
|Deployment       |Docker (Redis) + venv (Python)|On-prem, air-gapped capable|

## File Structure

```
redis-skill-router/
├── demo.py            # CLI entrypoint (setup / interactive / bench)
├── registry.py        # Redis schema, embedding, vector search
├── handlers.py        # Pi function handlers + SKILL_CATALOG
├── deploy.sh          # One-command deployment script
├── pyproject.toml     # Project metadata + dependencies
├── ARCHITECTURE.md    # This document
├── .gitignore
└── README.md
```

## The Pitch

> Redis model. Redis search. Redis storage. Full stack, zero external
> dependencies. No planner, no tools, no chain-of-thought. Semantic
> lookup and execute in 128 dimensions — 61 milliseconds end-to-end
> on a Raspberry Pi.
