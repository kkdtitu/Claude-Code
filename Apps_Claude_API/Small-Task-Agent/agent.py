import anthropic
import json
import sys
import io
import contextlib
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
# Load API key from .env file; fail fast with a clear message if missing
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found.\n"
        "Create a .env file:\n  ANTHROPIC_API_KEY=sk-ant-your-key-here"
    )

MODEL           = "claude-sonnet-4-5"
MAX_TOKENS      = 16000   # upper bound for response + thinking tokens combined
MAX_STEPS       = 10      # max agentic iterations before forcing a partial result
OUTPUT_FILE     = "agent_session.json"
THINKING_BUDGET = 10000   # tokens reserved for extended thinking per API call

client = anthropic.Anthropic(api_key=API_KEY)

# Appended to every user task so Claude always returns structured JSON at the end
JSON_INSTRUCTION = """

When your task is fully complete, respond with valid JSON only.
No markdown, no code fences, no extra text — pure JSON:
{
  "status": "completed | failed | partial",
  "summary": "what you did in 1-2 sentences",
  "result": "the main output, answer, or finding",
  "steps_taken": 0,
  "tools_used": [],
  "follow_up": "suggested next step or null"
}"""


# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(user_name: str, user_profession: str) -> str:
    # Inject user identity so the model can tailor tone and depth of explanation
    return f"""You are a capable, friendly task agent.

USER CONTEXT:
- Name      : {user_name}
- Profession: {user_profession}

Tailor your reasoning and explanations to this user's background.
Address them by name when appropriate.

AGENT RULES:
1. Think carefully before acting — use your reasoning to plan each step.
2. Break complex tasks into clear steps. Execute one tool at a time.
3. After each tool result, reason about what to do next.
4. When the task is complete, return a valid JSON summary.
5. Keep output focused and concise — not verbose.
6. Never produce unsafe, harmful, or offensive content."""


# ── Tool schemas (passed to every API call) ───────────────────────────────────
tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    },
    {
        "name": "run_python",
        "description": "Execute Python code and return stdout. Use for calculations and data processing.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to run"}},
            "required": ["code"]
        }
    },
    {
        "name": "read_file",
        "description": "Read a local file by path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a local file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        }
    }
]


# ── Tool implementations ──────────────────────────────────────────────────────

def web_search(query: str) -> str:
    # Stub — swap in a real provider (Brave, Serper, Tavily, etc.) for live results
    return f"[Mock search results for: {query}]\nResult 1: ...\nResult 2: ..."

def run_python(code: str) -> str:
    # Capture stdout so exec() output is returned as a string instead of printed
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue() or "Executed with no output."
    except Exception as e:
        return f"Error: {e}"

def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"

def execute_tool(name: str, inputs: dict) -> str:
    # Dispatch by tool name; unknown names return an error string (not an exception)
    return {
        "web_search": web_search,
        "run_python":  run_python,
        "read_file":   read_file,
        "write_file":  write_file,
    }.get(name, lambda **_: f"Unknown tool: {name}")(**inputs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_multiline(prompt: str) -> str:
    # Use sys.stdin.read() so the user can paste multi-paragraph text.
    # Reopen /dev/tty afterward so input() works again in the same session.
    print(f"\n{prompt}")
    print("  (Press Ctrl+D on Mac/Linux when done)\n")
    try:
        text = sys.stdin.read().strip()
    except EOFError:
        text = ""
    try:
        sys.stdin = open("/dev/tty")
    except OSError:
        pass
    return text

def parse_json(raw: str) -> dict:
    # Strip markdown fences Claude sometimes wraps around JSON despite instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts   = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Return a valid fallback dict so callers never receive raw unparsed text
        return {
            "status": "partial",
            "summary": "JSON parse failed",
            "result": raw,
            "steps_taken": 0,
            "tools_used": [],
            "follow_up": None,
            "parse_error": str(e),
        }

def save_session(user_name: str, user_profession: str, all_tasks: list) -> None:
    # Persist the full session so the user has an audit trail after exit
    output = {
        "session_date": datetime.now().isoformat(),
        "model":        MODEL,
        "user":         {"name": user_name, "profession": user_profession},
        "total_tasks":  len(all_tasks),
        "tasks":        all_tasks,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Session saved -> {OUTPUT_FILE}  ({len(all_tasks)} tasks)")


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(task: str, system: str) -> dict:
    # Seed the conversation with the user task + JSON output instructions
    messages   = [{"role": "user", "content": task + JSON_INSTRUCTION}]
    tools_used = []
    step       = 0

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  Step {step}/{MAX_STEPS}")

        try:
            # client.beta required to pass the betas + thinking parameters
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=1,           # required by extended thinking mode
                system=system,
                tools=tools,
                betas=["interleaved-thinking-2025-05-14"],
                thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
                messages=messages,
            )

        except anthropic.RateLimitError:
            print("  [rate limit — returning partial result]")
            return {
                "status": "failed",
                "summary": "Rate limit hit.",
                "result": "",
                "steps_taken": step,
                "tools_used": tools_used,
                "follow_up": "Wait a moment and retry.",
            }

        except anthropic.APIError as e:
            print(f"  [API error: {e}]")
            return {
                "status": "failed",
                "summary": f"API error: {str(e)[:120]}",
                "result": "",
                "steps_taken": step,
                "tools_used": tools_used,
                "follow_up": "Check API key.",
            }

        print(f"  stop_reason: {response.stop_reason}")

        # Print a preview of each thinking block so the user can follow reasoning
        for block in response.content:
            if block.type == "thinking":
                preview = block.thinking[:150].replace("\n", " ")
                print(f"  [thinking] {preview}...")

        # end_turn → task is done; extract the text block and parse as JSON
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    result = parse_json(block.text)
                    result["steps_taken"] = step
                    result["tools_used"]  = list(set(tools_used))
                    return result
            break

        # tool_use → execute all requested tools, append results, continue loop
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({json.dumps(block.input)[:80]})")
                    result = execute_tool(block.name, block.input)
                    print(f"  [result] {str(result)[:100]}...")
                    tools_used.append(block.name)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })

            # Feed all tool results back as a single user message
            messages.append({"role": "user", "content": tool_results})

    # Exceeded MAX_STEPS without completing — return what we have
    return {
        "status": "partial",
        "summary": f"Max steps ({MAX_STEPS}) reached.",
        "result": "",
        "steps_taken": step,
        "tools_used": list(set(tools_used)),
        "follow_up": "Try a simpler task.",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 54)
    print("  Task Agent — powered by Claude")
    print("  Type 'quit' at any prompt to exit")
    print("=" * 54)

    # Collect user profile once at startup to personalize every subsequent task
    print("\n  Tell me a bit about yourself first.")
    try:
        user_name       = input("  Your name: ").strip() or "Friend"
        user_profession = input("  Your profession: ").strip() or "General"
    except (EOFError, KeyboardInterrupt):
        user_name, user_profession = "Friend", "General"

    system    = build_system_prompt(user_name, user_profession)
    all_tasks = []

    print(f"\n  Ready, {user_name}! Describe your task below.")

    while True:
        task = read_multiline("Your task:")

        # Empty input or "quit" both exit gracefully and save the session
        if not task or task.lower() == "quit":
            save_session(user_name, user_profession, all_tasks)
            break

        print(f"\n  Running agent on task...")
        print("─" * 54)

        result = run_agent(task, system)

        print("\n" + "=" * 54)
        print("  Result")
        print("=" * 54)
        print(json.dumps(result, indent=2))

        all_tasks.append({"task": task, "output": result})

        try:
            choice = input("\nRun another task? (yes / quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            save_session(user_name, user_profession, all_tasks)
            break

        if choice != "yes":
            save_session(user_name, user_profession, all_tasks)
            break


if __name__ == "__main__":
    main()
