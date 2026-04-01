# Redis Semantic Skill Router

Zero-agent command dispatch: natural language → vector search → function execution.

## Architecture

```
User Input          Embedding Model           Redis Vector Search       Handler
"how hot is    →   redis/langcache-embed-v2  →  FT.SEARCH KNN        →  system.cpu_temp()
 the CPU?"          (128-dim Matryoshka)         idx:skills                → {"temp_c": 52.1}
```

No LLM in the loop. No chain-of-thought. Sub-50ms routing after warm-up.

## Prerequisites

**Redis Stack** (includes RediSearch + RedisJSON):
```bash
# Docker (easiest)
docker run -d --name redis-stack \
  -p 6379:6379 -p 8001:8001 \
  redis/redis-stack:latest

# Or install natively on Debian/Ubuntu:
# See https://redis.io/docs/install/install-stack/
```

**Python deps:**
```bash
pip install -r requirements.txt
```

> Note: `sentence-transformers` will download `redis/langcache-embed-v2` (~430MB)
> on first run. On a Pi 5 with 8GB RAM this runs fine. Vectors are only 128-dim
> thanks to Matryoshka truncation, keeping Redis memory usage minimal.

## Usage

```bash
# 1. Create index + embed & register all skills (run once)
python demo.py setup

# 2. Interactive mode — type commands in natural language
python demo.py

# 3. Benchmark — run accuracy + latency test suite
python demo.py bench
```

## Interactive Commands

```
❯ how hot is the processor
  → cpu_temperature → system.cpu_temp
  similarity: 0.891  |  search: 3.2ms  |  exec: 0.4ms  |  total: 3.6ms
  {"temp_c": 52.1, "temp_f": 125.8}

❯ top3 check the temperature       # show top 3 matches
  1. 0.876 ████████████████ cpu_temperature → system.cpu_temp
  2. 0.412 ████████         system_info → system.info
  3. 0.301 ██████           current_time → system.timestamp
```

## Adding New Skills

1. Write a handler function in `handlers.py`
2. Register it in `SKILL_CATALOG`
3. Add a skill definition in `registry.py` → `SKILLS` list
4. Re-run `python demo.py setup`

## File Structure

```
skill-router/
├── demo.py          # CLI entrypoint (setup / interactive / bench)
├── registry.py      # Redis schema, embedding, vector search
├── handlers.py      # Pi function handlers + SKILL_CATALOG
├── requirements.txt
└── README.md
```

## Competition Narrative

> "Redis model, Redis search, Redis storage — full stack, zero external
>  dependencies. No planner, no tools, no chain-of-thought. Semantic
>  lookup and execute in 128 dimensions — sub-50ms."

## Next Steps

- [ ] Wire up GPIO / robot tank / servo handlers
- [ ] Add ESP32 as voice input frontend (Whisper → text → embed → Redis)
- [ ] Build e-ink or web dashboard showing real-time routing telemetry
- [ ] Add confidence thresholding + "did you mean?" fallback
- [ ] Benchmark against agent-based approach for the speed comparison slide
