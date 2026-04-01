"""
registry.py — Redis Vector Search skill registry.

Stores skill definitions as JSON documents in Redis with vector embeddings.
Performs KNN similarity search to route natural language → handler function.
"""

import json
import numpy as np
from redis import Redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────
MODEL_NAME = "redis/langcache-embed-v3-small"  # Redis's v3 small model, 384-dim
VECTOR_DIM = 384                               # Native output dim (no truncation needed)
INDEX_NAME = "idx:skills"
KEY_PREFIX = "skill:"
DISTANCE_METRIC = "COSINE"

# ── Skill Definitions ───────────────────────────────────────────
# Add new skills here. The 'description' is what gets embedded.
# 'phrases' are extra training examples embedded + averaged with description.

SKILLS = [
    {
        "name": "cpu_temperature",
        "description": "Read the current CPU temperature of the Raspberry Pi",
        "phrases": [
            "how hot is the processor",
            "what is the CPU temp",
            "check thermal status",
            "is the pi overheating",
            "temperature reading",
        ],
        "handler": "system.cpu_temp",
        "default_args": {},
    },
    {
        "name": "memory_usage",
        "description": "Check how much RAM is being used on the system",
        "phrases": [
            "how much memory is free",
            "RAM usage",
            "show memory stats",
            "is the system running low on memory",
            "available memory",
        ],
        "handler": "system.memory",
        "default_args": {},
    },
    {
        "name": "disk_usage",
        "description": "Check disk space and storage capacity",
        "phrases": [
            "how much disk space is left",
            "storage usage",
            "is the drive full",
            "check free space",
            "filesystem capacity",
        ],
        "handler": "system.disk",
        "default_args": {},
    },
    {
        "name": "system_uptime",
        "description": "How long the system has been running since last reboot",
        "phrases": [
            "how long has the pi been on",
            "when was the last reboot",
            "system uptime",
            "how long since restart",
            "time since boot",
        ],
        "handler": "system.uptime",
        "default_args": {},
    },
    {
        "name": "system_info",
        "description": "Get system identification info like hostname and architecture",
        "phrases": [
            "what system is this",
            "show hostname",
            "what OS and kernel",
            "system details",
            "identify this machine",
        ],
        "handler": "system.info",
        "default_args": {},
    },
    {
        "name": "network_info",
        "description": "Show network interfaces and IP addresses",
        "phrases": [
            "what is my IP address",
            "show network config",
            "list network interfaces",
            "what IP does the pi have",
            "network status",
        ],
        "handler": "system.network",
        "default_args": {},
    },
    {
        "name": "running_processes",
        "description": "List the top running processes by CPU usage",
        "phrases": [
            "what processes are running",
            "show top processes",
            "what is using the most CPU",
            "process list",
            "task manager",
        ],
        "handler": "system.processes",
        "default_args": {},
    },
    {
        "name": "current_time",
        "description": "Get the current date and time on the Pi",
        "phrases": [
            "what time is it",
            "current date",
            "show the clock",
            "what day is it",
            "timestamp",
        ],
        "handler": "system.timestamp",
        "default_args": {},
    },
    {
        "name": "ping_host",
        "description": "Ping a network host to check connectivity and latency",
        "phrases": [
            "ping google",
            "check internet connection",
            "test network latency",
            "is the network up",
            "ping 8.8.8.8",
        ],
        "handler": "system.ping",
        "default_args": {"host": "8.8.8.8", "count": 3},
    },
]


class SkillRegistry:
    """Redis-backed semantic skill router."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = Redis.from_url(redis_url, decode_responses=False)
        self.model = None  # lazy-load

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"  Loading embedding model: {MODEL_NAME}...")
            self.model = SentenceTransformer(MODEL_NAME)
        return self.model

    def embed(self, text: str) -> np.ndarray:
        """Generate a single embedding vector (384-dim, normalized)."""
        return self._get_model().encode(text, normalize_embeddings=True)

    def embed_skill(self, skill: dict) -> np.ndarray:
        """Embed a skill by averaging the description + all example phrases."""
        texts = [skill["description"]] + skill.get("phrases", [])
        vectors = self._get_model().encode(texts, normalize_embeddings=True)
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        return centroid

    def create_index(self, drop_existing: bool = True):
        """Create the RediSearch vector index."""
        if drop_existing:
            try:
                self.r.ft(INDEX_NAME).dropindex(delete_documents=True)
                print(f"  Dropped existing index: {INDEX_NAME}")
            except Exception:
                pass

        schema = (
            TagField("$.name", as_name="name"),
            TextField("$.description", as_name="description"),
            TagField("$.handler", as_name="handler"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": DISTANCE_METRIC,
                },
                as_name="embedding",
            ),
        )

        definition = IndexDefinition(
            prefix=[KEY_PREFIX],
            index_type=IndexType.JSON,
        )

        self.r.ft(INDEX_NAME).create_index(
            fields=schema,
            definition=definition,
        )
        print(f"  Created index: {INDEX_NAME}")

    def register_skills(self, skills: list[dict] | None = None):
        """Embed and store all skills in Redis."""
        skills = skills or SKILLS
        for skill in skills:
            vec = self.embed_skill(skill)
            doc = {
                "name": skill["name"],
                "description": skill["description"],
                "phrases": skill.get("phrases", []),
                "handler": skill["handler"],
                "default_args": skill.get("default_args", {}),
                "embedding": vec.tolist(),
            }
            key = f"{KEY_PREFIX}{skill['name']}"
            self.r.json().set(key, "$", doc)
            print(f"  Registered: {skill['name']} → {skill['handler']}")

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Semantic search: embed the query and find nearest skills.
        Returns list of matches with score + skill metadata.
        """
        vec = self.embed(query_text)
        blob = vec.astype(np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("name", "description", "handler", "default_args", "score")
            .dialect(2)
        )

        results = self.r.ft(INDEX_NAME).search(q, query_params={"vec": blob})

        matches = []
        for doc in results.docs:
            score = float(doc.score)
            # COSINE distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - distance
            similarity = 1.0 - score
            matches.append({
                "name": doc.name.decode() if isinstance(doc.name, bytes) else doc.name,
                "description": (
                    doc.description.decode()
                    if isinstance(doc.description, bytes)
                    else doc.description
                ),
                "handler": (
                    doc.handler.decode()
                    if isinstance(doc.handler, bytes)
                    else doc.handler
                ),
                "default_args": json.loads(
                    doc.default_args.decode()
                    if isinstance(doc.default_args, bytes)
                    else doc.default_args
                ),
                "distance": round(score, 4),
                "similarity": round(similarity, 4),
            })

        return matches

    def route(self, query_text: str, threshold: float = 0.40) -> dict | None:
        """
        Route a query to the best skill.
        Returns None if the best match is below the similarity threshold.
        """
        matches = self.search(query_text, top_k=1)
        if not matches:
            return None

        best = matches[0]
        if best["similarity"] < threshold:
            return None

        return best
