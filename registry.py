"""
registry.py — Redis Vector Search skill registry.

Stores skill definitions as JSON documents in Redis with vector embeddings.
Performs KNN similarity search to route natural language → handler function.
"""

import json
import numpy as np
from redis import Redis
from sentence_transformers import SentenceTransformer

# RediSearch imports — optional, only needed when modules are loaded
try:
    from redis.commands.search.field import TextField, TagField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    _HAS_SEARCH_LIB = True
except ImportError:
    _HAS_SEARCH_LIB = False

# ── Config ───────────────────────────────────────────────────────
# Local ONNX export (no HuggingFace download, no torch re-export needed)
# Run export_onnx_model.py once to create this directory
MODEL_NAME = "/home/nion/redis-robot/models/langcache-embed-v3-small-onnx"
VECTOR_DIM = 384  # Native output dim, no truncation
INDEX_NAME = "idx:skills"
KEY_PREFIX = "skill:"
DISTANCE_METRIC = "COSINE"

# ── Skill Definitions ───────────────────────────────────────────
# Add new skills here. The 'description' is what gets embedded.
# 'phrases' are extra training examples embedded + averaged with description.

SKILLS = [
    # ── System Skills ──────────────────────────────────────
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
    # ── Robot Movement Skills ──────────────────────────────────
    {
        "name": "drive_forward",
        "description": "Drive the robot tank forward a short distance",
        "phrases": [
            "move forward",
            "go straight ahead",
            "drive forward",
            "go forward",
            "advance",
        ],
        "handler": "robot.drive_forward",
        "default_args": {"speed": 30, "duration_ms": 500},
    },
    {
        "name": "drive_backward",
        "description": "Drive the robot tank backward a short distance",
        "phrases": [
            "go backwards",
            "reverse",
            "back up",
            "drive backward",
            "move back",
        ],
        "handler": "robot.drive_backward",
        "default_args": {"speed": 30, "duration_ms": 500},
    },
    {
        "name": "turn_left",
        "description": "Pivot the robot tank to the left about 90 degrees",
        "phrases": [
            "turn left",
            "spin to the left",
            "rotate left",
            "go left",
            "pivot left",
        ],
        "handler": "robot.turn_left",
        "default_args": {"speed": 40, "duration_ms": 400},
    },
    {
        "name": "turn_right",
        "description": "Pivot the robot tank to the right about 90 degrees",
        "phrases": [
            "turn right",
            "spin to the right",
            "rotate right",
            "go right",
            "pivot right",
        ],
        "handler": "robot.turn_right",
        "default_args": {"speed": 40, "duration_ms": 400},
    },
    {
        "name": "stop",
        "description": "Stop all robot motors immediately",
        "phrases": [
            "stop",
            "halt",
            "freeze",
            "stop moving",
            "emergency stop",
        ],
        "handler": "robot.stop",
        "default_args": {},
    },
    {
        "name": "drive_forward_slow",
        "description": "Creep the robot forward slowly for precision movement",
        "phrases": [
            "inch forward",
            "creep forward",
            "move forward slowly",
            "nudge forward",
            "go forward a little bit",
        ],
        "handler": "robot.drive_forward_slow",
        "default_args": {"speed": 15, "duration_ms": 500},
    },
    {
        "name": "obstacle_avoid",
        "description": "Drive forward while automatically avoiding obstacles",
        "phrases": [
            "avoid obstacles",
            "navigate around obstacles",
            "drive and dodge obstacles",
            "autonomous driving",
            "explore without hitting things",
        ],
        "handler": "robot.obstacle_avoid",
        "default_args": {"steps": 20},
    },
    # ── Robot Sensor Skills ────────────────────────────────────
    {
        "name": "read_ultrasonic",
        "description": "Read the front ultrasonic distance sensor to detect obstacles",
        "phrases": [
            "check for obstacles",
            "how far is the wall",
            "measure distance ahead",
            "ultrasonic reading",
            "what's in front of me",
        ],
        "handler": "robot.read_ultrasonic",
        "default_args": {},
    },
    {
        "name": "read_battery",
        "description": "Check the robot battery voltage level",
        "phrases": [
            "how much battery is left",
            "check battery level",
            "battery status",
            "what is the battery voltage",
            "is the battery low",
        ],
        "handler": "robot.read_battery",
        "default_args": {},
    },
    # ── Robot Camera Skills ────────────────────────────────────
    {
        "name": "capture_image",
        "description": "Take a photo with the robot's camera",
        "phrases": [
            "take a photo",
            "capture an image",
            "snap a picture",
            "take a picture",
            "photograph what you see",
        ],
        "handler": "robot.capture_image",
        "default_args": {},
    },
    {
        "name": "pan_camera_left",
        "description": "Pan the robot camera to look left",
        "phrases": [
            "look left",
            "pan camera left",
            "turn the camera left",
            "look to the left",
            "camera left",
        ],
        "handler": "robot.pan_camera_left",
        "default_args": {"step": 20},
    },
    {
        "name": "pan_camera_right",
        "description": "Pan the robot camera to look right",
        "phrases": [
            "look right",
            "pan camera right",
            "turn the camera right",
            "look to the right",
            "camera right",
        ],
        "handler": "robot.pan_camera_right",
        "default_args": {"step": 20},
    },
    {
        "name": "tilt_camera_up",
        "description": "Tilt the robot camera upward",
        "phrases": [
            "look up",
            "tilt camera up",
            "angle the camera up",
            "look upward",
            "camera up",
        ],
        "handler": "robot.tilt_camera_up",
        "default_args": {"step": 15},
    },
    {
        "name": "tilt_camera_down",
        "description": "Tilt the robot camera downward",
        "phrases": [
            "look down",
            "tilt camera down",
            "angle the camera down",
            "look downward",
            "camera down",
        ],
        "handler": "robot.tilt_camera_down",
        "default_args": {"step": 15},
    },
    {
        "name": "camera_center",
        "description": "Return the robot camera to its center position",
        "phrases": [
            "center the camera",
            "reset camera position",
            "camera home",
            "look straight ahead",
            "camera to default",
        ],
        "handler": "robot.camera_center",
        "default_args": {},
    },
    # ── Robot LED Skills ───────────────────────────────────────
    {
        "name": "led_on",
        "description": "Turn on the robot LED strip lights",
        "phrases": [
            "turn on the lights",
            "lights on",
            "enable LEDs",
            "switch on lights",
            "illuminate",
        ],
        "handler": "robot.led_on",
        "default_args": {"r": 255, "g": 255, "b": 255},
    },
    {
        "name": "led_off",
        "description": "Turn off the robot LED strip lights",
        "phrases": [
            "turn off the lights",
            "lights off",
            "disable LEDs",
            "switch off lights",
            "kill the lights",
        ],
        "handler": "robot.led_off",
        "default_args": {},
    },
]


class SkillRegistry:
    """Redis-backed semantic skill router.

    Auto-detects whether RediSearch + RedisJSON modules are available.
    If yes: uses FT.SEARCH KNN for vector search (fastest).
    If no:  stores vectors in plain Redis keys and uses numpy cosine
            similarity for search (still fast for <100 skills).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = Redis.from_url(redis_url, decode_responses=False)
        self.model = None  # lazy-load
        self._has_modules = self._check_modules()

    def _check_modules(self) -> bool:
        """Check if RediSearch and RedisJSON modules are loaded."""
        if not _HAS_SEARCH_LIB:
            return False
        try:
            modules = self.r.module_list()
            names = {m[b"name"].decode().lower() for m in modules}
            has_search = "search" in names or "ft" in names
            has_json = "rejson" in names or "redisjson" in names
            return has_search and has_json
        except Exception:
            return False

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"  Loading embedding model: {MODEL_NAME} (ONNX)...")
            self.model = SentenceTransformer(MODEL_NAME, backend="onnx")
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

    # ── Index + Registration (dual mode) ───────────────────────

    def create_index(self, drop_existing: bool = True):
        """Create the vector index (RediSearch) or clear plain keys (fallback)."""
        if self._has_modules:
            self._create_index_redisearch(drop_existing)
        else:
            print("  RediSearch not available — using numpy fallback mode")
            if drop_existing:
                for key in self.r.keys(f"{KEY_PREFIX}*"):
                    self.r.delete(key)
                print(f"  Cleared existing keys with prefix: {KEY_PREFIX}")

    def _create_index_redisearch(self, drop_existing: bool):
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
        definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.JSON)
        self.r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print(f"  Created index: {INDEX_NAME}")

    def register_skills(self, skills=None):
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
            if self._has_modules:
                self.r.json().set(key, "$", doc)
            else:
                self.r.set(key, json.dumps(doc).encode())
            print(f"  Registered: {skill['name']} -> {skill['handler']}")

    # ── Search (dual mode) ─────────────────────────────────────

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Semantic search — auto-selects RediSearch KNN or numpy fallback."""
        if self._has_modules:
            return self._search_redisearch(query_text, top_k)
        return self._search_numpy(query_text, top_k)

    def _search_redisearch(self, query_text: str, top_k: int) -> list[dict]:
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

    def _search_numpy(self, query_text: str, top_k: int) -> list[dict]:
        """Fallback: load all skill vectors from Redis, compute cosine similarity."""
        vec = self.embed(query_text)

        keys = self.r.keys(f"{KEY_PREFIX}*")
        if not keys:
            return []

        scored = []
        for key in keys:
            raw = self.r.get(key)
            if raw is None:
                continue
            doc = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            skill_vec = np.array(doc["embedding"], dtype=np.float32)
            # Cosine similarity (both vectors are already normalized)
            similarity = float(np.dot(vec, skill_vec))
            scored.append((similarity, doc))

        # Sort by similarity descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        matches = []
        for similarity, doc in scored[:top_k]:
            matches.append({
                "name": doc["name"],
                "description": doc["description"],
                "handler": doc["handler"],
                "default_args": doc.get("default_args", {}),
                "distance": round(1.0 - similarity, 4),
                "similarity": round(similarity, 4),
            })
        return matches

    def route(self, query_text: str, threshold: float = 0.48):
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
