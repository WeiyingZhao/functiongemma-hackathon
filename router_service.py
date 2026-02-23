"""
EdgeAction Concierge — FastAPI Router Service

Wraps generate_hybrid() as an HTTP API with routing trace metadata.
Provides OpenAI-compatible endpoint for OpenClaw integration.
"""

import json
import os
import sys
import time
import uuid

sys.path.insert(0, "cactus/python/src")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from main import generate_hybrid, generate_cactus, generate_cloud
from demo_tools import execute_tool_call, TOOL_EXECUTORS

app = FastAPI(
    title="EdgeAction Concierge",
    description="Hybrid edge-cloud routing for function calling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory trace store
_traces = {}

# Default tool definitions for demo
DEMO_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour (0-23)"},
                "minute": {"type": "integer", "description": "Minute (0-59)"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {"type": "string", "description": "Time for the reminder"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name to search"}},
            "required": ["query"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "type": "object",
            "properties": {"song": {"type": "string", "description": "Song or playlist name"}},
            "required": ["song"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer", "description": "Minutes"}},
            "required": ["minutes"],
        },
    },
]


class RouteRequest(BaseModel):
    messages: list
    tools: Optional[list] = None
    execute: bool = False  # Whether to actually execute the tool calls


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "edgeaction-concierge"
    messages: list[ChatMessage]
    tools: Optional[list] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "EdgeAction Concierge v1.0"}


@app.post("/route")
def route(req: RouteRequest):
    """Direct routing endpoint with full trace metadata."""
    request_id = str(uuid.uuid4())[:8]
    tools = req.tools or DEMO_TOOLS

    start = time.time()
    result = generate_hybrid(req.messages, tools)
    latency_ms = (time.time() - start) * 1000

    source = result.get("source", "unknown")
    tool_calls = result.get("function_calls", [])

    # Build trace
    trace = {
        "request_id": request_id,
        "source": source,
        "confidence": result.get("confidence", result.get("local_confidence", 0)),
        "latency_ms": round(latency_ms, 1),
        "tools_available": [t["name"] for t in tools],
        "tools_called": [c["name"] for c in tool_calls],
        "data_sent_to_cloud": "cloud" in source,
        "internal_time_ms": round(result.get("total_time_ms", 0), 1),
    }
    _traces[request_id] = trace

    # Optionally execute tool calls
    execution_results = []
    if req.execute:
        for call in tool_calls:
            exec_result = execute_tool_call(call["name"], call.get("arguments", {}))
            execution_results.append({
                "tool": call["name"],
                "arguments": call.get("arguments", {}),
                "result": exec_result,
            })

    return {
        "request_id": request_id,
        "function_calls": tool_calls,
        "trace": trace,
        "execution_results": execution_results if req.execute else None,
    }


@app.get("/trace/{request_id}")
def get_trace(request_id: str):
    """Retrieve routing trace for a previous request."""
    trace = _traces.get(request_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible endpoint for OpenClaw/LiteLLM integration."""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    tools = DEMO_TOOLS

    result = generate_hybrid(messages, tools)
    tool_calls = result.get("function_calls", [])
    source = result.get("source", "unknown")

    # Execute tool calls and build response
    responses = []
    for call in tool_calls:
        exec_result = execute_tool_call(call["name"], call.get("arguments", {}))
        responses.append(f"{call['name']}: {json.dumps(exec_result)}")

    content = "\n".join(responses) if responses else "No tool calls generated."
    content += f"\n\n[Routed via: {source}]"

    # OpenAI-compatible response format
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": "edgeaction-concierge",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments", {})),
                            },
                        }
                        for i, call in enumerate(tool_calls)
                    ] if tool_calls else None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "_routing": {
            "source": source,
            "confidence": result.get("confidence", 0),
            "latency_ms": result.get("total_time_ms", 0),
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  EdgeAction Concierge starting on http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
