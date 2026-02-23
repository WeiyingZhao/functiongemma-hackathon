"""
EdgeAction Concierge — Interactive Demo App

Interactive CLI demo that routes through generate_hybrid(),
executes tool calls, and displays rich routing traces.
"""

import sys
import json
import time

sys.path.insert(0, "cactus/python/src")

from main import generate_hybrid
from demo_tools import execute_tool_call

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# All available demo tools
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


def print_banner():
    print(f"""
{BOLD}{CYAN}{'='*60}
  EdgeAction Concierge - Hybrid Edge-Cloud Agent
  Powered by FunctionGemma (on-device) + Gemini Flash (cloud)
{'='*60}{RESET}

{DIM}Available tools: get_weather, set_alarm, send_message,
  create_reminder, search_contacts, play_music, set_timer

Type a natural language request, or 'quit' to exit.
Type 'demo' to run the 3 scripted demo scenarios.{RESET}
""")


def print_trace(result, wall_time_ms):
    """Print a color-coded routing trace."""
    source = result.get("source", "unknown")
    confidence = result.get("confidence", result.get("local_confidence", 0))
    internal_time = result.get("total_time_ms", 0)
    tool_calls = result.get("function_calls", [])

    # Color based on routing decision
    if "on-device" in source and "retry" not in source:
        color = GREEN
        icon = "LOCAL"
    elif "on-device" in source:
        color = YELLOW
        icon = "LOCAL (retry)"
    else:
        color = RED
        icon = "CLOUD"

    print(f"\n{BOLD}--- Routing Trace ---{RESET}")
    print(f"  Source:      {color}{BOLD}{icon}{RESET} {DIM}({source}){RESET}")
    print(f"  Confidence:  {confidence:.4f}")
    print(f"  Internal:    {internal_time:.1f}ms")
    print(f"  Wall time:   {wall_time_ms:.1f}ms")
    print(f"  Tool calls:  {len(tool_calls)}")
    if "cloud" in source:
        print(f"  {RED}Data sent to cloud: Yes{RESET}")
    else:
        print(f"  {GREEN}Data sent to cloud: No (private){RESET}")


def print_tool_results(tool_calls, results):
    """Print tool execution results."""
    print(f"\n{BOLD}--- Tool Execution ---{RESET}")
    for call, res in zip(tool_calls, results):
        print(f"  {CYAN}{call['name']}{RESET}({json.dumps(call.get('arguments', {}), separators=(',', ':'))})")
        print(f"    -> {json.dumps(res, indent=6)}")


def process_request(user_input):
    """Process a single user request through the hybrid router."""
    messages = [{"role": "user", "content": user_input}]

    print(f"\n{DIM}Routing request...{RESET}")
    start = time.time()
    result = generate_hybrid(messages, DEMO_TOOLS)
    wall_time_ms = (time.time() - start) * 1000

    # Print trace
    print_trace(result, wall_time_ms)

    # Execute tool calls
    tool_calls = result.get("function_calls", [])
    if tool_calls:
        results = []
        for call in tool_calls:
            res = execute_tool_call(call["name"], call.get("arguments", {}))
            results.append(res)
        print_tool_results(tool_calls, results)
    else:
        print(f"\n{YELLOW}No tool calls generated.{RESET}")

    return result


def run_demo_scenarios():
    """Run 3 scripted demo scenarios."""
    scenarios = [
        {
            "label": "Easy: Single tool, direct request",
            "query": "What's the weather in San Francisco?",
        },
        {
            "label": "Medium: Tool selection among many",
            "query": "Play some jazz music.",
        },
        {
            "label": "Hard: Multi-tool, multi-action",
            "query": "Text Emma good night, check weather in Chicago, and set an alarm for 5 AM.",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{BOLD}{CYAN}{'='*60}")
        print(f"  Demo {i}/3: {scenario['label']}")
        print(f"  Query: \"{scenario['query']}\"")
        print(f"{'='*60}{RESET}")
        process_request(scenario["query"])
        if i < len(scenarios):
            print(f"\n{DIM}Press Enter for next demo...{RESET}")
            input()


def main():
    print_banner()

    while True:
        try:
            user_input = input(f"\n{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye!{RESET}")
            break
        if user_input.lower() == "demo":
            run_demo_scenarios()
            continue

        process_request(user_input)


if __name__ == "__main__":
    main()
