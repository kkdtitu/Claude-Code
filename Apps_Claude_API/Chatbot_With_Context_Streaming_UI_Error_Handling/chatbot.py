"""
Web AI Chatbot — Flask + Anthropic Claude
Streaming · Multi-turn history · Browser UI · JSON output
"""

import anthropic
import json
import os
import socket
import time
from flask import Flask, Response, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MAX_HISTORY = 20
MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
MAX_RETRIES = 5
RETRY_DELAY = 10

history: list[dict] = []
user_profile: dict = {"name": "friend", "profession": "professional"}

SYSTEM_PROMPT = """You are a friendly, helpful AI assistant. You are talking to {name}, who works as a {profession}.

RESPONSE RULES — follow all of these on every single turn:

1. OUTPUT FORMAT: Respond ONLY with a valid JSON object. No markdown, no code fences, no extra text.
   Exact schema:
   {{"message": "<your reply>", "follow_up": "<one clarifying question or empty string>", "tip": "<encouragement or empty string>"}}

2. TONE: Warm, conversational, human. Like texting a knowledgeable friend. Never stiff or formal.

3. GREETING: If the user's first message is hi/hello/hey, respond with a warm personalised greeting using their name and profession.

4. FOLLOW-UP: Before a long or complex answer, put ONE focused clarifying question in "follow_up" instead of guessing.

5. STUCK USERS: If the user seems lost, guide and motivate them via "follow_up". Never assume or guess their intent.

6. SAFETY: Decline harmful, offensive, or dangerous requests politely and redirect.

7. BREVITY: Keep "message" concise. No walls of text. Short paragraphs or tight bullets only when they help.

8. PERSONALISATION: Naturally reference {name}'s profession ({profession}) when it adds value.
"""


def trim_history() -> list[dict]:
    return history[-MAX_HISTORY:]


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        name=user_profile["name"],
        profession=user_profile["profession"],
    )


def _attempt_stream(full_text_ref: list[str]):
    full_text_ref[0] = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(),
        messages=trim_history(),
    ) as stream:
        for chunk in stream.text_stream:
            full_text_ref[0] += chunk
            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"


def sse_stream(user_message: str):
    full_text_ref = [""]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            yield from _attempt_stream(full_text_ref)
            break

        except anthropic.RateLimitError:
            reason = "Rate limit reached"
        except anthropic.APIStatusError as e:
            reason = f"API status error {e.status_code}"
        except anthropic.APIConnectionError:
            reason = "Connection to Anthropic failed"
        except anthropic.APITimeoutError:
            reason = "Request timed out"
        except anthropic.APIError as e:
            reason = f"API error: {e}"
        except (socket.gaierror, OSError) as e:
            reason = f"Network error: {e}"
        except Exception as e:
            reason = f"Unexpected error: {e}"

        if attempt < MAX_RETRIES:
            yield f"data: {json.dumps({'type': 'retry', 'attempt': attempt, 'of': MAX_RETRIES, 'reason': reason})}\n\n"
            time.sleep(RETRY_DELAY)
            full_text_ref[0] = ""
        else:
            # Remove the failed user message from history
            if history and history[-1]["role"] == "user":
                history.pop()
            yield f"data: {json.dumps({'type': 'error', 'text': f'{reason}. All {MAX_RETRIES} retries exhausted. Please try again later.'})}\n\n"
            return

    try:
        parsed = json.loads(full_text_ref[0].strip())
    except json.JSONDecodeError:
        parsed = {"message": full_text_ref[0].strip(), "follow_up": "", "tip": ""}

    full_reply = parsed.get("message", "")
    if parsed.get("follow_up"):
        full_reply += f"\n\n{parsed['follow_up']}"
    if parsed.get("tip"):
        full_reply += f"\n\n💡 {parsed['tip']}"
    history.append({"role": "assistant", "content": full_reply})

    yield f"data: {json.dumps({'type': 'done', 'data': parsed})}\n\n"


@app.route("/")
def index():
    return HTML


@app.route("/profile", methods=["POST"])
def set_profile():
    data = request.get_json(force=True)
    user_profile["name"] = data.get("name", "friend").strip() or "friend"
    user_profile["profession"] = data.get("profession", "professional").strip() or "professional"
    history.clear()
    return jsonify({"status": "ok", "name": user_profile["name"]})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    if message.lower() in {"quit", "exit"}:
        history.clear()
        farewell = json.dumps({
            "message": f"It was great chatting with you, {user_profile['name']}! Take care 👋",
            "follow_up": "",
            "tip": "",
        })

        def _bye():
            yield f"data: {json.dumps({'type': 'token', 'text': ''})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': json.loads(farewell)})}\n\n"

        return Response(_bye(), mimetype="text/event-stream")

    history.append({"role": "user", "content": message})
    return Response(sse_stream(message), mimetype="text/event-stream")


@app.route("/reset")
def reset():
    history.clear()
    return jsonify({"status": "reset"})


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Chat</title>
  <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f0f4f8;
      --panel-bg: #ffffff;
      --user-bubble: #3b7dd8;
      --user-text: #ffffff;
      --assistant-bubble: #ffffff;
      --assistant-text: #2d3748;
      --input-bg: #ffffff;
      --border: #e2e8f0;
      --accent: #3b7dd8;
      --accent-hover: #2c6bc4;
      --retry-bg: #fffbeb;
      --retry-border: #f59e0b;
      --retry-text: #92400e;
      --error-bg: #fff5f5;
      --error-border: #fc8181;
      --error-text: #c53030;
      --meta-text: #718096;
      --shadow: 0 4px 24px rgba(0,0,0,0.08);
      --radius: 18px;
      --radius-sm: 10px;
    }

    body {
      font-family: 'Nunito', sans-serif;
      background: var(--bg);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }

    /* ── Profile Screen ── */
    #profile-screen {
      background: var(--panel-bg);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 48px 40px;
      width: 100%;
      max-width: 440px;
      text-align: center;
      animation: fadeIn 0.4s ease;
    }

    .profile-icon {
      width: 72px;
      height: 72px;
      background: linear-gradient(135deg, #3b7dd8, #6fa8f5);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 20px;
      font-size: 32px;
    }

    #profile-screen h1 {
      font-size: 1.7rem;
      font-weight: 700;
      color: #1a202c;
      margin-bottom: 8px;
    }

    #profile-screen p {
      color: var(--meta-text);
      font-size: 0.95rem;
      margin-bottom: 32px;
      line-height: 1.5;
    }

    .form-group {
      margin-bottom: 16px;
      text-align: left;
    }

    .form-group label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      color: #4a5568;
      margin-bottom: 6px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .form-group input {
      width: 100%;
      padding: 12px 16px;
      border: 2px solid var(--border);
      border-radius: var(--radius-sm);
      font-family: 'Nunito', sans-serif;
      font-size: 1rem;
      color: #2d3748;
      transition: border-color 0.2s;
      outline: none;
    }

    .form-group input:focus {
      border-color: var(--accent);
    }

    .form-group input::placeholder { color: #a0aec0; }

    #start-btn {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #3b7dd8, #6fa8f5);
      color: #fff;
      border: none;
      border-radius: var(--radius-sm);
      font-family: 'Nunito', sans-serif;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      margin-top: 8px;
      transition: transform 0.15s, box-shadow 0.15s;
      letter-spacing: 0.02em;
    }

    #start-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(59,125,216,0.35);
    }

    #start-btn:active { transform: translateY(0); }

    /* ── Chat Screen ── */
    #chat-screen {
      display: none;
      flex-direction: column;
      width: 100%;
      max-width: 740px;
      height: 92vh;
      max-height: 860px;
      background: var(--panel-bg);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
      animation: fadeIn 0.4s ease;
    }

    .chat-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px;
      background: linear-gradient(135deg, #3b7dd8, #6fa8f5);
      color: #fff;
      flex-shrink: 0;
    }

    .chat-header-left {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .chat-avatar {
      width: 40px;
      height: 40px;
      background: rgba(255,255,255,0.25);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
    }

    .chat-header-info h2 {
      font-size: 1.05rem;
      font-weight: 700;
    }

    .chat-header-info span {
      font-size: 0.78rem;
      opacity: 0.85;
    }

    .header-actions { display: flex; gap: 8px; }

    .icon-btn {
      background: rgba(255,255,255,0.2);
      border: none;
      border-radius: 8px;
      color: #fff;
      padding: 6px 12px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    .icon-btn:hover { background: rgba(255,255,255,0.35); }

    /* ── Messages Area ── */
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px 20px 8px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: var(--bg);
      scroll-behavior: smooth;
    }

    #messages::-webkit-scrollbar { width: 6px; }
    #messages::-webkit-scrollbar-track { background: transparent; }
    #messages::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }

    .bubble-row {
      display: flex;
      animation: slideIn 0.25s ease;
    }

    .bubble-row.user { justify-content: flex-end; }
    .bubble-row.assistant { justify-content: flex-start; }

    .bubble {
      max-width: 72%;
      padding: 12px 16px;
      border-radius: var(--radius);
      line-height: 1.55;
      font-size: 0.95rem;
      position: relative;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .bubble-row.user .bubble {
      background: var(--user-bubble);
      color: var(--user-text);
      border-bottom-right-radius: 4px;
    }

    .bubble-row.assistant .bubble {
      background: var(--assistant-bubble);
      color: var(--assistant-text);
      border-bottom-left-radius: 4px;
    }

    .msg-text { white-space: pre-wrap; word-break: break-word; }

    .msg-follow-up {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #e2e8f0;
      font-size: 0.88rem;
      color: #4a5568;
      font-style: italic;
    }

    .msg-tip {
      margin-top: 8px;
      font-size: 0.85rem;
      color: #667eea;
      font-weight: 600;
    }

    /* ── Typing Indicator ── */
    #typing {
      display: none;
      padding: 4px 0 4px 20px;
      flex-shrink: 0;
    }

    .typing-dots {
      display: inline-flex;
      gap: 4px;
      background: var(--panel-bg);
      padding: 10px 16px;
      border-radius: var(--radius);
      border-bottom-left-radius: 4px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .typing-dots span {
      width: 8px;
      height: 8px;
      background: #a0aec0;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }

    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

    /* ── Banners ── */
    #retry-banner {
      display: none;
      margin: 0 16px 0;
      padding: 10px 16px;
      background: var(--retry-bg);
      border: 1px solid var(--retry-border);
      border-radius: var(--radius-sm);
      color: var(--retry-text);
      font-size: 0.85rem;
      font-weight: 600;
      flex-shrink: 0;
    }

    #error-banner {
      display: none;
      margin: 0 16px 0;
      padding: 10px 16px;
      background: var(--error-bg);
      border: 1px solid var(--error-border);
      border-radius: var(--radius-sm);
      color: var(--error-text);
      font-size: 0.85rem;
      font-weight: 600;
      flex-shrink: 0;
    }

    /* ── Input Bar ── */
    .input-bar {
      display: flex;
      align-items: flex-end;
      gap: 10px;
      padding: 14px 16px;
      background: var(--panel-bg);
      border-top: 1px solid var(--border);
      flex-shrink: 0;
    }

    #user-input {
      flex: 1;
      padding: 12px 16px;
      border: 2px solid var(--border);
      border-radius: 14px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.95rem;
      color: #2d3748;
      resize: none;
      outline: none;
      max-height: 120px;
      transition: border-color 0.2s;
      line-height: 1.4;
    }

    #user-input:focus { border-color: var(--accent); }
    #user-input::placeholder { color: #a0aec0; }

    #send-btn {
      width: 46px;
      height: 46px;
      background: linear-gradient(135deg, #3b7dd8, #6fa8f5);
      border: none;
      border-radius: 14px;
      color: #fff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      transition: transform 0.15s, box-shadow 0.15s;
    }

    #send-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(59,125,216,0.4);
    }

    #send-btn:active { transform: translateY(0); }
    #send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

    #send-btn svg { width: 20px; height: 20px; }

    /* ── Animations ── */
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30%           { transform: translateY(-6px); }
    }

    /* ── Responsive ── */
    @media (max-width: 600px) {
      body { padding: 0; align-items: stretch; }
      #profile-screen { border-radius: 0; box-shadow: none; padding: 36px 24px; min-height: 100vh; justify-content: center; display: flex; flex-direction: column; }
      #chat-screen { border-radius: 0; height: 100vh; max-height: none; }
      .bubble { max-width: 85%; }
    }
  </style>
</head>
<body>

  <!-- Profile Screen -->
  <div id="profile-screen">
    <div class="profile-icon">🤖</div>
    <h1>Welcome to AI Chat</h1>
    <p>Tell me a little about yourself so I can personalize our conversation.</p>

    <div class="form-group">
      <label for="name-input">Your Name</label>
      <input type="text" id="name-input" placeholder="e.g. Alex" autocomplete="name">
    </div>

    <div class="form-group">
      <label for="profession-input">Your Profession</label>
      <input type="text" id="profession-input" placeholder="e.g. Software Engineer" autocomplete="organization-title">
    </div>

    <button id="start-btn" onclick="startChat()">Start Chatting →</button>
  </div>

  <!-- Chat Screen -->
  <div id="chat-screen">
    <div class="chat-header">
      <div class="chat-header-left">
        <div class="chat-avatar">🤖</div>
        <div class="chat-header-info">
          <h2>AI Assistant</h2>
          <span id="header-subtitle">Streaming · Claude-powered</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick="resetChat()">↺ Reset</button>
      </div>
    </div>

    <div id="messages"></div>

    <div id="typing">
      <div class="typing-dots">
        <span></span><span></span><span></span>
      </div>
    </div>

    <div id="retry-banner"></div>
    <div id="error-banner"></div>

    <div class="input-bar">
      <textarea id="user-input" rows="1" placeholder="Type a message… (Enter to send, Shift+Enter for new line)" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button id="send-btn" onclick="sendMessage()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  </div>

  <script>
    const profileScreen = document.getElementById('profile-screen');
    const chatScreen    = document.getElementById('chat-screen');
    const messagesEl    = document.getElementById('messages');
    const typingEl      = document.getElementById('typing');
    const retryBanner   = document.getElementById('retry-banner');
    const errorBanner   = document.getElementById('error-banner');
    const sendBtn       = document.getElementById('send-btn');
    const inputEl       = document.getElementById('user-input');
    const headerSub     = document.getElementById('header-subtitle');

    let streaming = false;

    async function startChat() {
      const name = document.getElementById('name-input').value.trim();
      const profession = document.getElementById('profession-input').value.trim();

      const res = await fetch('/profile', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, profession}),
      });
      const data = await res.json();

      headerSub.textContent = name ? `Hi ${data.name} · Claude-powered` : 'Streaming · Claude-powered';

      profileScreen.style.display = 'none';
      chatScreen.style.display = 'flex';
      messagesEl.innerHTML = '';
      inputEl.focus();
    }

    function showProfile() {
      chatScreen.style.display = 'none';
      profileScreen.style.display = 'block';
    }

    async function resetChat() {
      await fetch('/reset');
      showProfile();
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!streaming) sendMessage();
      }
    }

    function autoResize(el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function createBubble(role) {
      const row = document.createElement('div');
      row.className = `bubble-row ${role}`;

      const bubble = document.createElement('div');
      bubble.className = 'bubble';

      const msgText = document.createElement('div');
      msgText.className = 'msg-text';
      bubble.appendChild(msgText);
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      scrollToBottom();
      return bubble;
    }

    function appendBubble(role, text) {
      const bubble = createBubble(role);
      bubble.querySelector('.msg-text').textContent = text;
      scrollToBottom();
    }

    function renderExtras(bubble, data) {
      // Remove any previously appended extras
      const oldFU  = bubble.querySelector('.msg-follow-up');
      const oldTip = bubble.querySelector('.msg-tip');
      if (oldFU)  oldFU.remove();
      if (oldTip) oldTip.remove();

      if (data.follow_up) {
        const fu = document.createElement('div');
        fu.className = 'msg-follow-up';
        fu.textContent = '❓ ' + data.follow_up;
        bubble.appendChild(fu);
      }
      if (data.tip) {
        const tip = document.createElement('div');
        tip.className = 'msg-tip';
        tip.textContent = '💡 ' + data.tip;
        bubble.appendChild(tip);
      }
      scrollToBottom();
    }

    function showTyping(visible) {
      typingEl.style.display = visible ? 'block' : 'none';
      scrollToBottom();
    }

    function showRetryBanner(msg) {
      retryBanner.textContent = '⏳ ' + msg;
      retryBanner.style.display = 'block';
    }

    function hideRetryBanner() {
      retryBanner.style.display = 'none';
      retryBanner.textContent = '';
    }

    function showError(msg) {
      errorBanner.textContent = '⚠️ ' + msg;
      errorBanner.style.display = 'block';
      setTimeout(() => { errorBanner.style.display = 'none'; }, 8000);
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || streaming) return;

      streaming = true;
      sendBtn.disabled = true;
      inputEl.value = '';
      inputEl.style.height = 'auto';
      errorBanner.style.display = 'none';
      hideRetryBanner();

      appendBubble('user', text);
      showTyping(true);

      const assistantBubble = createBubble('assistant');

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: text}),
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({error: 'Request failed'}));
          showTyping(false);
          showError(err.error || 'Request failed');
          assistantBubble.closest('.bubble-row').remove();
          return;
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {stream: true});
          const lines = buffer.split('\\n\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            let event;
            try { event = JSON.parse(line.slice(6)); } catch { continue; }

            if (event.type === 'token') {
              assistantBubble.querySelector('.msg-text').textContent += event.text;
              scrollToBottom();
            } else if (event.type === 'retry') {
              showRetryBanner(`Retrying ${event.attempt}/${event.of} — ${event.reason}. Waiting 10 s…`);
              assistantBubble.querySelector('.msg-text').textContent = '';
            } else if (event.type === 'done') {
              hideRetryBanner();
              showTyping(false);
              // Replace streamed raw text with clean parsed message
              if (event.data && event.data.message) {
                assistantBubble.querySelector('.msg-text').textContent = event.data.message;
              }
              renderExtras(assistantBubble, event.data || {});
              if (['quit', 'exit'].includes(text.toLowerCase())) {
                setTimeout(showProfile, 1800);
              }
            } else if (event.type === 'error') {
              hideRetryBanner();
              showTyping(false);
              assistantBubble.closest('.bubble-row').remove();
              showError(event.text);
            }
          }
        }
      } catch (err) {
        showTyping(false);
        hideRetryBanner();
        showError('Connection lost: ' + err.message);
        assistantBubble.closest('.bubble-row').remove();
      } finally {
        streaming = false;
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
