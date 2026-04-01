# Redis Semantic Skill Router

Zero-agent command dispatch: natural language → vector search → function execution.
Redis model, Redis search, Redis storage — full stack, zero external dependencies.

## Architecture
```
User Input          Embedding Model           Redis Vector Search       Handler
"how hot is    →   redis/langcache-embed-v2  →  FT.SEARCH KNN        →  system.cpu_temp()
 the CPU?"          (128-dim Matryoshka)         idx:skills                → {"temp_c": 52.1}
```

No LLM in the loop. No chain-of-thought. Semantic lookup and execute.

## Prerequisites

### Docker

Redis Stack runs in Docker. Install Docker if not already present:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

### Python 3.11+

Verify with `python3 --version`.

## Quick Start
```bash
chmod +x deploy.sh
./deploy.sh
```

This handles everything: Docker, venv, dependencies, model download, index creation.

## Usage
```bash
source .venv/bin/activate

# Interactive mode
python demo.py

# Benchmark (accuracy + latency)
python demo.py bench

# Re-register skills after editing
python demo.py setup
```

## Benchmark (Raspberry Pi 5)

| Metric               | Value       |
|----------------------|-------------|
| Routing accuracy      | 21/21 (100%) |
| Avg search latency    | 260ms       |
| Vector dimensions     | 128         |
| Skills registered     | 9           |
| Runtime               | PyTorch     |

See the `onnx-optimization` branch for **4.3x faster inference** (61ms avg)
with a single line change.

## On-Prem Model

The embedding model (`redis/langcache-embed-v2`) downloads from HuggingFace on
first run and caches locally. After that, the entire pipeline runs offline.
No API keys. No internet. No cost per query.

## Adding New Skills

1. Write a handler function in `handlers.py`
2. Register it in `SKILL_CATALOG`
3. Add a skill definition in `registry.py` → `SKILLS` list
4. Re-run `python demo.py setup`

## File Structure
```
redis-skill-router/
├── demo.py            # CLI entrypoint (setup / interactive / bench)
├── registry.py        # Redis schema, embedding, vector search
├── handlers.py        # Pi function handlers + SKILL_CATALOG
├── deploy.sh          # One-command deployment script
├── pyproject.toml     # Project metadata + dependencies
├── ARCHITECTURE.md    # Detailed architecture overview
├── VISION.md          # Product vision document
├── .gitignore
└── README.md
```

## Branches

| Branch               | Runtime  | Avg Latency | Notes                     |
|---------------------|----------|-------------|---------------------------|
| `main`               | PyTorch  | 260ms       | Baseline                  |
| `onnx-optimization`  | ONNX     | 61ms        | 4.3x faster, 1 line change|

