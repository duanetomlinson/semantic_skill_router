#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# deploy.sh — Redis Semantic Skill Router deployment for Pi 5
#
# Run this on your Pi over SSH:
#   chmod +x deploy.sh && ./deploy.sh
#
# Prerequisites:
#   - Docker installed (for Redis Stack)
#   - Python 3.11+ installed
#   - Internet access (first run only, to pull model + docker image)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[deploy]${NC} $1"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $1"; }
ok()   { echo -e "${GREEN}[deploy]${NC} $1"; }

# ── Step 1: Redis Stack via Docker ───────────────────────────
log "Step 1/5: Starting Redis Stack..."

if docker ps --format '{{.Names}}' | grep -q '^redis-stack$'; then
    ok "Redis Stack already running."
elif docker ps -a --format '{{.Names}}' | grep -q '^redis-stack$'; then
    log "Redis Stack container exists but stopped. Starting..."
    docker start redis-stack
    ok "Redis Stack started."
else
    log "Pulling and starting Redis Stack..."
    docker run -d \
        --name redis-stack \
        --restart unless-stopped \
        -p 6379:6379 \
        -p 8001:8001 \
        redis/redis-stack:latest
    ok "Redis Stack running on :6379 (RedisInsight on :8001)."
fi

# Wait for Redis to be ready
log "Waiting for Redis to accept connections..."
for i in $(seq 1 30); do
    if docker exec redis-stack redis-cli ping 2>/dev/null | grep -q PONG; then
        ok "Redis is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        warn "Redis didn't respond in 30s. Check 'docker logs redis-stack'."
        exit 1
    fi
    sleep 1
done

# ── Step 2: Python virtual environment ───────────────────────
log "Step 2/5: Setting up Python virtual environment..."

VENV_DIR="$(pwd)/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "Venv already exists."
fi

source "$VENV_DIR/bin/activate"

# ── Step 3: Install dependencies ─────────────────────────────
log "Step 3/5: Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Dependencies installed."

# ── Step 4: Download model (first run only) ──────────────────
log "Step 4/5: Pre-downloading embedding model..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading redis/langcache-embed-v2 (if not cached)...')
model = SentenceTransformer('redis/langcache-embed-v2')
print('  Model ready.')
"
ok "Model cached locally."

# ── Step 5: Create index + register skills ───────────────────
log "Step 5/5: Creating Redis index and registering skills..."
python3 demo.py setup
ok "Setup complete!"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment complete! Run these commands:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}source .venv/bin/activate${NC}"
echo ""
echo -e "  # Interactive mode"
echo -e "  ${CYAN}python3 demo.py${NC}"
echo ""
echo -e "  # Benchmark (accuracy + latency)"
echo -e "  ${CYAN}python3 demo.py bench${NC}"
echo ""
echo -e "  # Re-register skills after editing"
echo -e "  ${CYAN}python3 demo.py setup${NC}"
echo ""
