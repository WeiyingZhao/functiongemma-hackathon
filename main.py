
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"
lfm2_path = "cactus/weights/lfm2-1.2b-tool"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


# ===== Module-level model cache =====
_cached_fg = None
_cached_lfm = None


def _get_fg():
    """Get or create cached FunctionGemma model handle (fast, single-tool)."""
    global _cached_fg
    if _cached_fg is None:
        _cached_fg = cactus_init(functiongemma_path)
    return _cached_fg


def _get_lfm():
    """Get or create cached LFM2-1.2B-Tool model handle (reliable multi-tool)."""
    global _cached_lfm
    if _cached_lfm is None:
        _cached_lfm = cactus_init(lfm2_path)
    return _cached_lfm


# ===== Validation & helper functions =====

def _validate_tool_calls(function_calls, tools):
    """Validate tool calls against available tool schemas. Returns (is_valid, issues)."""
    tool_names = {t["name"] for t in tools}
    tool_schemas = {t["name"]: t for t in tools}

    if not function_calls:
        return False, ["no_calls"]

    issues = []
    for call in function_calls:
        name = call.get("name")
        if name not in tool_names:
            issues.append(f"unknown_tool:{name}")
            continue
        schema = tool_schemas[name]
        required = schema.get("parameters", {}).get("required", [])
        args = call.get("arguments", {})
        for req in required:
            if req not in args:
                issues.append(f"missing_arg:{name}.{req}")

    return len(issues) == 0, issues


def _align_string_with_source(value, source_text):
    """Try to find a better match for value in source_text preserving original words."""
    val_lower = value.lower().strip()
    src_lower = source_text.lower()
    # If value already appears verbatim in source, it's fine
    if val_lower in src_lower:
        return value
    # Try to find a longer span in source that contains all words of value
    val_words = val_lower.split()
    if not val_words:
        return value
    # Look for the first and last word of value in source
    src_words = src_lower.split()
    orig_words = source_text.split()
    for i in range(len(src_words)):
        if src_words[i] == val_words[0]:
            # Search a wider window to catch more dropped words
            for j in range(i + len(val_words) - 1, min(i + len(val_words) + 3, len(src_words))):
                if src_words[j] == val_words[-1]:
                    candidate_words = src_words[i:j+1]
                    # Check all original val words are in the candidate
                    if all(w in candidate_words for w in val_words):
                        return " ".join(orig_words[i:j+1])
    # Fallback: try substring match on each word to find partial overlaps
    # e.g., model says "lo-fi music" but source says "lo-fi beats" — keep model output
    return value


def _args_grounded_in_message(function_calls, user_msg):
    """Check if string argument values are grounded in the user message (not hallucinated)."""
    if not user_msg:
        return True
    lower_msg = user_msg.lower()
    for call in function_calls:
        for key, val in call.get("arguments", {}).items():
            if isinstance(val, str) and len(val) > 2:
                val_words = [w for w in val.lower().split() if len(w) > 2]
                if val_words and not any(w in lower_msg for w in val_words):
                    return False
    return True


def _repair_missing_args(function_calls, tools, user_msg):
    """Fill in missing required string arguments by extracting from the user message."""
    if not user_msg:
        return
    tool_schemas = {t["name"]: t for t in tools}
    lower_msg = user_msg.lower()
    for call in function_calls:
        schema = tool_schemas.get(call.get("name"), {})
        props = schema.get("parameters", {}).get("properties", {})
        required = schema.get("parameters", {}).get("required", [])
        args = call.get("arguments", {})
        for req in required:
            if req not in args or args.get(req) in ("", None, {}):
                prop = props.get(req, {})
                if prop.get("type") == "string":
                    # Use the tool name verb as anchor in user message
                    tool_name = call.get("name", "")
                    verb = tool_name.split("_")[0]  # "play" from "play_music"
                    if verb in lower_msg:
                        idx = lower_msg.index(verb) + len(verb)
                        after = user_msg[idx:].strip()
                        # Cut at comma, " and ", or period
                        for sep in [",", " and ", "."]:
                            pos = after.lower().find(sep)
                            if pos > 0:
                                after = after[:pos]
                                break
                        cleaned = after.strip().rstrip(".,;!?")
                        if cleaned:
                            args[req] = cleaned
        # Also ensure arguments dict is set on the call
        call["arguments"] = args


def _coerce_arguments(function_calls, tools, user_msg=""):
    """Fix type mismatches in arguments based on tool schemas."""
    tool_schemas = {t["name"]: t for t in tools}
    for call in function_calls:
        schema = tool_schemas.get(call.get("name"), {})
        props = schema.get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        for key, val in list(args.items()):
            if key in props:
                expected_type = props[key].get("type", "string")
                if expected_type == "integer" and isinstance(val, str):
                    try:
                        args[key] = int(val)
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "integer" and isinstance(val, float):
                    args[key] = int(val)
                elif expected_type == "number" and isinstance(val, str):
                    try:
                        args[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "string" and not isinstance(val, str):
                    args[key] = str(val)
            # Strip trailing punctuation from string values
            if isinstance(args.get(key), str):
                args[key] = args[key].rstrip(".,;!?")
            # Try to align string values with source text to fix dropped words
            if isinstance(args.get(key), str) and user_msg:
                args[key] = _align_string_with_source(args[key], user_msg)


def _estimate_expected_calls(message_text):
    """Heuristic: estimate how many tool calls the user's message requires."""
    lower = message_text.lower()
    count = 1
    count += lower.count(" and ")
    # For "X, Y, and Z" patterns: commas before final "and" indicate extra actions
    if " and " in lower:
        and_idx = lower.rfind(" and ")
        prefix = lower[:and_idx]
        count += prefix.count(", ")
    return min(max(count, 1), 5)


def _run_local(messages, tools, tool_rag_top_k=0, use_lfm=False):
    """Run local model with cached handle. use_lfm=True for LFM2, False for FunctionGemma."""
    model = _get_lfm() if use_lfm else _get_fg()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_top_k,
        confidence_threshold=0.05,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "cloud_handoff": True,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
    }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    system_prompt = (
        "You are a precise function-calling assistant. "
        "Analyze the user request carefully. "
        "Call ONLY the functions that match the user's intent. "
        "For each function call, provide ALL required arguments with correct types."
    )
    msgs = [{"role": "system", "content": system_prompt}] + messages
    return _run_local(msgs, tools)


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    system_instruction = (
        "You are a precise function-calling assistant. "
        "When the user asks for multiple actions, call ALL relevant functions in a single response. "
        "Always provide all required arguments with correct types. "
        "Extract argument values directly from the user's words without adding extra words or punctuation. "
        "Do not skip any requested action."
    )

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            temperature=0,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _get_call_signature(function_calls):
    """Extract a comparable signature from function calls (tool names in sorted order)."""
    return tuple(sorted(c.get("name", "") for c in function_calls))


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Speed-optimized hybrid routing: single LFM2 run + smart post-processing.

    Strategy:
    - 1 tool (easy): FG first (fast ~80ms on server), grounding check, LFM2 backup
    - 2+ tools, single action: LFM2 single run (~230ms on server)
    - 2+ tools, multi action: LFM2 single run with multi-call prompt, retry only if under-produced
    """
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    expected_calls = _estimate_expected_calls(user_msg)
    num_tools = len(tools)
    tool_names = [t["name"] for t in tools]

    single_prompt = (
        "You are a precise function-calling assistant. "
        "Analyze the user request carefully. "
        "Call ONLY the function that best matches the user's intent. "
        "Provide ALL required arguments. Extract argument values directly from the user's words."
    )
    multi_prompt = (
        "You are a precise function-calling assistant. "
        "The user is requesting MULTIPLE actions. You MUST call ALL relevant functions — "
        "one for EACH action the user mentions. "
        "Provide ALL required arguments. Extract argument values directly from the user's words."
    )

    system_prompt = multi_prompt if expected_calls > 1 else single_prompt
    local_messages = [{"role": "system", "content": system_prompt}] + messages

    # --- Path A: Single tool → FG first (fast), grounding check, LFM2 backup ---
    if num_tools <= 1:
        local = _run_local(local_messages, tools, use_lfm=False)
        local_time = local["total_time_ms"]
        valid, _ = _validate_tool_calls(local["function_calls"], tools)
        if valid and _args_grounded_in_message(local["function_calls"], user_msg):
            _coerce_arguments(local["function_calls"], tools, user_msg)
            local["source"] = "on-device"
            return local
        # FG failed or hallucinated → LFM2 backup
        local2 = _run_local(local_messages, tools, use_lfm=True)
        local_time += local2["total_time_ms"]
        _repair_missing_args(local2["function_calls"], tools, user_msg)
        valid2, _ = _validate_tool_calls(local2["function_calls"], tools)
        if valid2:
            _coerce_arguments(local2["function_calls"], tools, user_msg)
            local2["source"] = "on-device"
            local2["total_time_ms"] = local_time
            return local2
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local_time
        _coerce_arguments(cloud["function_calls"], tools, user_msg)
        return cloud

    # --- Path B: Multiple tools → LFM2 single run, conditional retry ---
    local = _run_local(local_messages, tools, use_lfm=True)
    local_time = local["total_time_ms"]
    _repair_missing_args(local["function_calls"], tools, user_msg)
    valid, issues = _validate_tool_calls(local["function_calls"], tools)

    if valid:
        num_calls = len(local["function_calls"])
        # For multi-action: retry once if we got fewer calls than expected
        if expected_calls > 1 and num_calls < expected_calls:
            local2 = _run_local(local_messages, tools, use_lfm=True)
            local_time += local2["total_time_ms"]
            _repair_missing_args(local2["function_calls"], tools, user_msg)
            v2, _ = _validate_tool_calls(local2["function_calls"], tools)
            if v2 and len(local2["function_calls"]) > num_calls:
                _coerce_arguments(local2["function_calls"], tools, user_msg)
                local2["source"] = "on-device"
                local2["total_time_ms"] = local_time
                return local2
        _coerce_arguments(local["function_calls"], tools, user_msg)
        local["source"] = "on-device"
        local["total_time_ms"] = local_time
        return local

    # Invalid first run → retry once
    local2 = _run_local(local_messages, tools, use_lfm=True)
    local_time += local2["total_time_ms"]
    _repair_missing_args(local2["function_calls"], tools, user_msg)
    valid2, _ = _validate_tool_calls(local2["function_calls"], tools)
    if valid2:
        _coerce_arguments(local2["function_calls"], tools, user_msg)
        local2["source"] = "on-device"
        local2["total_time_ms"] = local_time
        return local2

    # Both invalid → cloud
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] += local_time
    _coerce_arguments(cloud["function_calls"], tools, user_msg)
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
