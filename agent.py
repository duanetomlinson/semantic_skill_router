# -*- coding: utf-8 -*-
"""
agent.py -- Tier 2 conversational agent for complex robot tasks.

Uses Claude's native tool_use API with the Redis skill catalog as tools.
Supports multi-turn conversation, vision analysis, autonomous navigation,
and skill learning (saves successful strategies as reusable prompts).

Architecture:
    User <-> AgentSession <-> Claude API (tool_use)
                  |
                  v
            SKILL_CATALOG (handlers.py)
                  |
                  v
            Robot Hardware (motors, camera, sensors)
"""

import os
import sys
import json
import signal
import base64
import inspect
import time
from datetime import datetime

from handlers import SKILL_CATALOG

# ── Config ─────────────────────────────────────────────────────
MAX_TOOL_CALLS = 100
OBSTACLE_THRESHOLD_CM = 15
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AGENT_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", "claude-sonnet-4-20250514")


def _load_env_file():
    """Load .env from project directory."""
    global ANTHROPIC_API_KEY, AGENT_LLM_MODEL
    for env_path in [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "redis-robot", ".env"),
    ]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    AGENT_LLM_MODEL = os.environ.get("AGENT_LLM_MODEL", AGENT_LLM_MODEL)


_load_env_file()


# ── Tool Definition Builder ────────────────────────────────────

def build_tools():
    """Build Claude tool definitions from SKILL_CATALOG."""
    tools = []
    for handler_name, fn in SKILL_CATALOG.items():
        sig = inspect.signature(fn)
        properties = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if ann == int:
                ptype = "integer"
            elif ann == str:
                ptype = "string"
            else:
                ptype = "integer" if isinstance(param.default, int) else "string"
            prop = {"type": ptype}
            if param.default != inspect.Parameter.empty:
                prop["description"] = f"Default: {param.default}"
            properties[pname] = prop
        # Claude tool names can't have dots
        tool_name = handler_name.replace(".", "_")
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        tools.append({
            "name": tool_name,
            "description": doc,
            "input_schema": {
                "type": "object",
                "properties": properties,
            },
        })
    return tools


# Tool name mapping: claude_name <-> handler_name
def _to_handler_name(tool_name):
    """Convert Claude tool name back to handler name (underscores -> first dot)."""
    # robot_drive_forward -> robot.drive_forward
    # system_cpu_temp -> system.cpu_temp
    parts = tool_name.split("_", 1)
    return f"{parts[0]}.{parts[1]}" if len(parts) > 1 else tool_name


# ── Tool Execution ─────────────────────────────────────────────

def _encode_image(filepath):
    """Base64-encode an image for vision. Only /tmp/ images allowed."""
    if not filepath or not os.path.exists(filepath):
        return None
    real_path = os.path.realpath(filepath)
    if not real_path.startswith("/tmp/"):
        return None
    if not real_path.lower().endswith((".jpg", ".jpeg", ".png")):
        return None
    with open(real_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def execute_tool(tool_name, tool_input, print_fn=print):
    """Execute a skill and return the result dict + optional image."""
    handler_name = _to_handler_name(tool_name)
    fn = SKILL_CATALOG.get(handler_name)
    if fn is None:
        return {"error": f"unknown tool: {handler_name}"}, None

    # Filter args to only accepted parameters
    sig = inspect.signature(fn)
    valid_args = {k: v for k, v in tool_input.items() if k in sig.parameters}

    # Safety: ultrasonic check before forward movement
    if handler_name in ("robot.drive_forward", "robot.drive_forward_slow"):
        ultrasonic_fn = SKILL_CATALOG.get("robot.read_ultrasonic")
        if ultrasonic_fn:
            dist_result = ultrasonic_fn()
            dist = dist_result.get("distance_cm")
            if dist is not None and dist < OBSTACLE_THRESHOLD_CM:
                stop_fn = SKILL_CATALOG.get("robot.stop")
                if stop_fn:
                    stop_fn()
                return {
                    "warning": "obstacle_detected",
                    "distance_cm": dist,
                    "action": "stopped",
                }, None

    # Execute
    try:
        result = fn(**valid_args)
    except Exception as e:
        result = {"error": str(e)}

    # Check for image result
    image_b64 = None
    filepath = result.get("filepath")
    if filepath and handler_name == "robot.capture_image":
        image_b64 = _encode_image(filepath)

    brief = json.dumps(result)
    if len(brief) > 120:
        brief = brief[:120] + "..."
    print_fn(f"  [tool] {handler_name} -> {brief}")

    return result, image_b64


# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are the brain of a Freenove Tank Robot on a Raspberry Pi 4B.
You control the robot by calling tools. You can see through its camera.

CAPABILITIES:
- Move: drive forward/backward, turn left/right, creep slowly, stop
- Look: capture images, pan/tilt camera, center camera
- Sense: read ultrasonic distance, check battery
- Navigate: obstacle avoidance mode
- Light: LED strip on/off with RGB colors
- System: CPU temp, memory, disk, network, processes, time, ping

BEHAVIOR:
- For navigation tasks: take a photo first to assess, then plan movement.
- Before driving forward, always check ultrasonic distance.
- Take photos frequently to verify progress toward goals.
- If the user asks you to find something, systematically scan by rotating
  and photographing in each direction, then move toward the target.
- Describe what you see in photos concisely.
- When you accomplish a goal, say so clearly.
- If you need user input or confirmation, ask directly.
- Keep movements small and safe. Prefer multiple small steps over one large one.

SAFETY:
- Maximum {max_calls} tool calls per session.
- Ultrasonic auto-check before forward movement (< {threshold}cm = stop).
- If you detect an obstacle, report it and choose a different action.
""".format(max_calls=MAX_TOOL_CALLS, threshold=OBSTACLE_THRESHOLD_CM)


# ── Agent Session ──────────────────────────────────────────────

class AgentSession:
    """Multi-turn conversational agent that controls the robot via tools."""

    def __init__(self, print_fn=print, system_prompt=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = AGENT_LLM_MODEL
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.tools = build_tools()
        self.messages = []
        self.tool_call_count = 0
        self.action_log = []
        self.print_fn = print_fn
        self._original_sigint = None

    def _log(self, msg):
        self.print_fn(f"  [agent] {msg}")

    def _setup_sigint(self):
        self._original_sigint = signal.getsignal(signal.SIGINT)
        def handler(sig, frame):
            self._log("INTERRUPTED -- stopping motors")
            stop_fn = SKILL_CATALOG.get("robot.stop")
            if stop_fn:
                stop_fn()
            signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGINT, handler)

    def _restore_sigint(self):
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)

    def chat(self, user_input):
        """Send a message and get the agent's response.
        The agent may call tools autonomously before responding."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            if self.tool_call_count >= MAX_TOOL_CALLS:
                self._log(f"Reached max tool calls ({MAX_TOOL_CALLS})")
                return "I've reached the maximum number of actions for this session."

            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=self.messages,
                tools=self.tools,
                max_tokens=4096,
            )

            # Append assistant response to history
            self.messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Check for tool calls
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                # Text-only response -- return to user
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts)

            # Execute each tool call and build results
            tool_results = []
            for tool_use in tool_uses:
                self.tool_call_count += 1
                result, image_b64 = execute_tool(
                    tool_use.name, tool_use.input, self.print_fn
                )
                self.action_log.append((tool_use.name, tool_use.input, result))

                # Build tool_result content
                content = [{"type": "text", "text": json.dumps(result)}]

                # If image captured, include it for vision analysis
                if image_b64:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": content,
                })

            # Send tool results back to Claude
            self.messages.append({"role": "user", "content": tool_results})
            # Loop continues -- Claude will process results and decide next action

    def get_summary(self):
        """Ask the agent to summarize what it accomplished."""
        if not self.action_log:
            return "No actions were taken."
        return self.chat(
            "Summarize what you just accomplished in 1-2 sentences. "
            "Focus on the strategy you used, not the specific movements."
        )

    def get_strategy_prompt(self, task_description):
        """Ask the agent to distill its approach into a reusable strategy prompt."""
        actions_taken = "\n".join(
            f"  {i+1}. {name}({json.dumps(inp)}) -> {json.dumps(res)}"
            for i, (name, inp, res) in enumerate(self.action_log)
        )
        return self.chat(
            f"You just completed this task: '{task_description}'\n\n"
            f"Actions you took:\n{actions_taken}\n\n"
            "Write a reusable STRATEGY PROMPT that a future robot agent could "
            "follow to accomplish this same type of task in any environment. "
            "The prompt should describe the general approach and decision-making "
            "process, NOT specific directions or distances. "
            "Write it as instructions to a robot agent. Keep it under 200 words."
        )


# ── Learned Skill Execution ────────────────────────────────────

def run_learned_skill(agent_prompt, task_name, print_fn=print):
    """Execute a learned skill by spawning a mini-agent with the saved strategy."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set"}

    # Build a system prompt that combines the base + learned strategy
    learned_system = SYSTEM_PROMPT + f"\n\nLEARNED STRATEGY for '{task_name}':\n{agent_prompt}"

    session = AgentSession(print_fn=print_fn, system_prompt=learned_system)
    session._setup_sigint()

    print_fn(f"  [agent] Running learned skill: {task_name}")
    response = session.chat(f"Execute the learned strategy for: {task_name}")
    print_fn(f"  [agent] {response}")

    session._restore_sigint()
    return {
        "action": "learned_skill",
        "task": task_name,
        "steps": session.tool_call_count,
        "response": response,
    }


# ── Skill Learning ─────────────────────────────────────────────

def save_learned_skill(registry, name, description, strategy_prompt, original_query):
    """Save a learned agent strategy as a new Tier 1 skill in Redis."""
    # The handler for learned skills is a special string that demo.py checks
    skill = {
        "name": name,
        "description": description,
        "phrases": [original_query],
        "handler": f"agent.learned:{name}",
        "default_args": {},
        "agent_prompt": strategy_prompt,
    }

    # Embed and store
    vec = registry.embed_skill(skill)
    doc = {**skill, "embedding": vec.tolist()}
    key = f"skill:{name}"
    registry.r.set(key, json.dumps(doc).encode())

    return skill


# ── Public Entry Points ────────────────────────────────────────

def start_agent_session(print_fn=print):
    """Create a new multi-turn agent session."""
    if not ANTHROPIC_API_KEY:
        print_fn("  [agent] Error: ANTHROPIC_API_KEY not set. Add to .env")
        return None
    session = AgentSession(print_fn=print_fn)
    session._setup_sigint()
    return session
