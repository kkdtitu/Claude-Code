"""
Web AI Chatbot — Flask + Anthropic Claude
Streaming · Multi-turn history · Browser UI · JSON output
"""

import anthropic
import json
import os
import time
from flask import Flask, Response, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MAX_HISTORY = 20
MODEL       = "claude-opus-4-5"
MAX_TOKENS  = 1024

history: list[dict] = []
user_profile: dict  = {"name": "friend", "profession": "professional"}

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


def sse_stream(user_message: str):
    """
    Generator that yields SSE events.
    Event types:
      {"type": "token",  "text": "..."}   — partial text chunk
      {"type": "done",   "data": {...}}   — final parsed JSON response
      {"type": "error",  "text": "..."}   — recoverable error message
    """
    full_text = ""

    def _call_api():
        nonlocal full_text
        full_text = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_prompt(),
            messages=trim_history(),
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

    try:
        yield from _call_api()

    except anthropic.RateLimitError:
        yield f"data: {json.dumps({'type': 'error', 'text': 'Rate limit reached — retrying in 30 s…'})}\n\n"
        time.sleep(30)
        try:
            yield from _call_api()
        except Exception as e:
            # Remove the failed user turn from history
            if history and history[-1]["role"] == "user":
                history.pop()
            yield f"data: {json.dumps({'type': 'error', 'text': f'Retry failed: {e}'})}\n\n"
            return

    except anthropic.APIError as e:
        if history and history[-1]["role"] == "user":
            history.pop()
        yield f"data: {json.dumps({'type': 'error', 'text': f'API error: {e}'})}\n\n"
        return

    except Exception as e:
        if history and history[-1]["role"] == "user":
            history.pop()
        yield f"data: {json.dumps({'type': 'error', 'text': f'Unexpected error: {e}'})}\n\n"
        return

    try:
        parsed = json.loads(full_text.strip())
    except json.JSONDecodeError:
        parsed = {"message": full_text.strip(), "follow_up": "", "tip": ""}

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
    user_profile["name"]       = data.get("name", "friend").strip() or "friend"
    user_profile["profession"] = data.get("profession", "professional").strip() or "professional"
    history.clear()
    return jsonify({"status": "ok", "name": user_profile["name"]})


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    if message.lower() in {"quit", "exit"}:
        history.clear()
        farewell = {
            "message": f"It was great chatting with you, {user_profile['name']}! Take care 👋",
            "follow_up": "",
            "tip": "",
        }
        def _bye():
            yield f"data: {json.dumps({'type': 'token', 'text': ''})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': farewell})}\n\n"
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

    body {
      font-family: 'Nunito', sans-serif;
      background: linear-gradient(135deg, #e0f2fe 0%, #f0fdf4 50%, #fdf4ff 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }

    /* ── Profile Screen ─────────────────────────────── */
    #profile-screen {
      background: #fff;
      border-radius: 24px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.12);
      padding: 2.5rem 2rem;
      width: 100%;
      max-width: 420px;
      animation: fadeIn 0.4s ease;
    }

    #profile-screen h1 {
      font-size: 1.8rem;
      font-weight: 700;
      color: #1e293b;
      text-align: center;
      margin-bottom: 0.4rem;
    }

    #profile-screen p.subtitle {
      text-align: center;
      color: #64748b;
      font-size: 0.95rem;
      margin-bottom: 2rem;
    }

    .field-group {
      margin-bottom: 1.2rem;
    }

    .field-group label {
      display: block;
      font-weight: 600;
      color: #374151;
      margin-bottom: 0.4rem;
      font-size: 0.9rem;
    }

    .field-group input {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 2px solid #e2e8f0;
      border-radius: 12px;
      font-family: 'Nunito', sans-serif;
      font-size: 1rem;
      color: #1e293b;
      transition: border-color 0.2s;
      outline: none;
    }

    .field-group input:focus {
      border-color: #14b8a6;
    }

    #start-btn {
      width: 100%;
      padding: 0.85rem;
      background: linear-gradient(135deg, #14b8a6, #0ea5e9);
      color: #fff;
      border: none;
      border-radius: 12px;
      font-family: 'Nunito', sans-serif;
      font-size: 1.05rem;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
      margin-top: 0.5rem;
    }

    #start-btn:hover { opacity: 0.92; }
    #start-btn:active { transform: scale(0.98); }

    /* ── Chat Screen ────────────────────────────────── */
    #chat-screen {
      display: none;
      flex-direction: column;
      width: 100%;
      max-width: 700px;
      height: 90vh;
      max-height: 820px;
      background: #f8fafc;
      border-radius: 24px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.12);
      overflow: hidden;
      animation: fadeIn 0.4s ease;
    }

    /* Header */
    .chat-header {
      background: linear-gradient(135deg, #14b8a6, #0ea5e9);
      padding: 1rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }

    .chat-header .title-area h2 {
      color: #fff;
      font-size: 1.1rem;
      font-weight: 700;
    }

    .chat-header .title-area p {
      color: rgba(255,255,255,0.8);
      font-size: 0.8rem;
    }

    #reset-btn {
      background: rgba(255,255,255,0.2);
      color: #fff;
      border: 1px solid rgba(255,255,255,0.4);
      border-radius: 8px;
      padding: 0.4rem 0.8rem;
      font-family: 'Nunito', sans-serif;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    #reset-btn:hover { background: rgba(255,255,255,0.35); }

    /* Messages area */
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 1.2rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      scroll-behavior: smooth;
    }

    #messages::-webkit-scrollbar { width: 6px; }
    #messages::-webkit-scrollbar-track { background: transparent; }
    #messages::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

    /* Bubbles */
    .bubble-row {
      display: flex;
      animation: bubbleIn 0.25s ease;
    }

    .bubble-row.user { justify-content: flex-end; }
    .bubble-row.assistant { justify-content: flex-start; }

    .bubble {
      max-width: 75%;
      padding: 0.75rem 1rem;
      border-radius: 18px;
      font-size: 0.95rem;
      line-height: 1.5;
      word-wrap: break-word;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .bubble-row.user .bubble {
      background: linear-gradient(135deg, #14b8a6, #0ea5e9);
      color: #fff;
      border-bottom-right-radius: 4px;
    }

    .bubble-row.assistant .bubble {
      background: #fff;
      color: #1e293b;
      border-bottom-left-radius: 4px;
    }

    .bubble .msg-text { white-space: pre-wrap; }

    .bubble .follow-up {
      margin-top: 0.6rem;
      padding-top: 0.6rem;
      border-top: 1px solid #e2e8f0;
      color: #0369a1;
      font-size: 0.88rem;
      font-weight: 600;
    }

    .bubble .tip-text {
      margin-top: 0.5rem;
      color: #065f46;
      font-size: 0.88rem;
      font-style: italic;
    }

    /* Error notice */
    .error-notice {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
      border-radius: 10px;
      padding: 0.6rem 0.9rem;
      font-size: 0.88rem;
      text-align: center;
      animation: bubbleIn 0.2s ease;
    }

    /* Typing indicator */
    #typing {
      display: none;
      align-items: center;
      gap: 5px;
      padding: 0.5rem 1rem;
    }

    #typing span {
      width: 8px;
      height: 8px;
      background: #94a3b8;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }

    #typing span:nth-child(2) { animation-delay: 0.2s; }
    #typing span:nth-child(3) { animation-delay: 0.4s; }

    /* Input bar */
    .input-bar {
      display: flex;
      gap: 0.6rem;
      padding: 0.9rem 1rem;
      background: #fff;
      border-top: 1px solid #e2e8f0;
      flex-shrink: 0;
    }

    #msg-input {
      flex: 1;
      padding: 0.7rem 1rem;
      border: 2px solid #e2e8f0;
      border-radius: 12px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.97rem;
      color: #1e293b;
      outline: none;
      transition: border-color 0.2s;
      resize: none;
    }

    #msg-input:focus { border-color: #14b8a6; }

    #send-btn {
      padding: 0.7rem 1.2rem;
      background: linear-gradient(135deg, #14b8a6, #0ea5e9);
      color: #fff;
      border: none;
      border-radius: 12px;
      font-family: 'Nunito', sans-serif;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
      white-space: nowrap;
    }

    #send-btn:hover { opacity: 0.9; }
    #send-btn:active { transform: scale(0.96); }
    #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Animations */
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(16px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes bubbleIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30%            { transform: translateY(-6px); }
    }

    /* Responsive */
    @media (max-width: 480px) {
      #chat-screen { height: 100vh; border-radius: 0; max-height: none; }
      body { padding: 0; align-items: stretch; }
      .bubble { max-width: 88%; }
    }
  </style>
</head>
<body>

  <!-- ── Profile Screen ── -->
  <div id="profile-screen">
    <h1>👋 Welcome!</h1>
    <p class="subtitle">Let's personalise your chat experience</p>

    <div class="field-group">
      <label for="name-input">Your Name</label>
      <input id="name-input" type="text" placeholder="e.g. Alex" autocomplete="off">
    </div>

    <div class="field-group">
      <label for="prof-input">Your Profession</label>
      <input id="prof-input" type="text" placeholder="e.g. Software Engineer" autocomplete="off">
    </div>

    <button id="start-btn" onclick="startChat()">Start Chatting →</button>
  </div>

  <!-- ── Chat Screen ── -->
  <div id="chat-screen">
    <div class="chat-header">
      <div class="title-area">
        <h2 id="chat-title">AI Assistant</h2>
        <p id="chat-subtitle">Here to help you</p>
      </div>
      <button id="reset-btn" onclick="resetChat()">↩ New Chat</button>
    </div>

    <div id="messages"></div>

    <div id="typing">
      <span></span><span></span><span></span>
    </div>

    <div class="input-bar">
      <input id="msg-input" type="text" placeholder="Type a message… (Enter to send)" autocomplete="off">
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    const profileScreen = document.getElementById('profile-screen');
    const chatScreen    = document.getElementById('chat-screen');
    const messagesEl    = document.getElementById('messages');
    const typingEl      = document.getElementById('typing');
    const msgInput      = document.getElementById('msg-input');
    const sendBtn       = document.getElementById('send-btn');

    // ── Profile ──────────────────────────────────────
    async function startChat() {
      const name = document.getElementById('name-input').value.trim();
      const prof = document.getElementById('prof-input').value.trim();

      if (!name) {
        document.getElementById('name-input').focus();
        return;
      }

      await fetch('/profile', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, profession: prof || 'professional'}),
      });

      document.getElementById('chat-title').textContent = `Hi, ${name}! 👋`;
      document.getElementById('chat-subtitle').textContent = prof || 'AI Assistant';
      profileScreen.style.display = 'none';
      chatScreen.style.display    = 'flex';
      messagesEl.innerHTML        = '';
      msgInput.focus();
    }

    function showProfile() {
      chatScreen.style.display    = 'none';
      profileScreen.style.display = 'block';
      messagesEl.innerHTML        = '';
    }

    async function resetChat() {
      await fetch('/reset');
      showProfile();
    }

    // ── Bubble helpers ───────────────────────────────
    function createBubble(role) {
      const row    = document.createElement('div');
      row.className = `bubble-row ${role}`;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      const msgText = document.createElement('span');
      msgText.className = 'msg-text';
      bubble.appendChild(msgText);
      row.appendChild(bubble);
      messagesEl.appendChild(row);
      scrollToBottom();
      return row;
    }

    function appendBubble(role, text) {
      const row = createBubble(role);
      row.querySelector('.msg-text').textContent = text;
      return row;
    }

    function renderExtras(row, data) {
      const bubble = row.querySelector('.bubble');
      if (data.follow_up) {
        const fu = document.createElement('div');
        fu.className = 'follow-up';
        fu.textContent = '❓ ' + data.follow_up;
        bubble.appendChild(fu);
      }
      if (data.tip) {
        const tip = document.createElement('div');
        tip.className = 'tip-text';
        tip.textContent = '💡 ' + data.tip;
        bubble.appendChild(tip);
      }
      scrollToBottom();
    }

    function showError(text) {
      const el = document.createElement('div');
      el.className = 'error-notice';
      el.textContent = '⚠️ ' + text;
      messagesEl.appendChild(el);
      scrollToBottom();
    }

    function scrollToBottom() {
      messagesEl.scrollTo({top: messagesEl.scrollHeight, behavior: 'smooth'});
    }

    function showTyping(visible) {
      typingEl.style.display = visible ? 'flex' : 'none';
      if (visible) messagesEl.scrollTo({top: messagesEl.scrollHeight});
    }

    function setInputEnabled(enabled) {
      msgInput.disabled = !enabled;
      sendBtn.disabled  = !enabled;
    }

    // ── Send Message ─────────────────────────────────
    async function sendMessage() {
      const text = msgInput.value.trim();
      if (!text || sendBtn.disabled) return;

      msgInput.value = '';
      setInputEnabled(false);
      appendBubble('user', text);
      showTyping(true);

      const assistantRow = createBubble('assistant');
      const assistantText = assistantRow.querySelector('.msg-text');

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({message: text}),
        });

        if (!response.ok) {
          showTyping(false);
          showError('Server error: ' + response.status);
          setInputEnabled(true);
          return;
        }

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer    = '';

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
              assistantText.textContent += event.text;
              scrollToBottom();
            } else if (event.type === 'done') {
              showTyping(false);
              // Replace streamed text with clean message field
              assistantText.textContent = event.data.message || assistantText.textContent;
              renderExtras(assistantRow, event.data);
              if (['quit', 'exit'].includes(text.toLowerCase())) {
                setTimeout(showProfile, 1800);
              }
            } else if (event.type === 'error') {
              showTyping(false);
              showError(event.text);
            }
          }
        }
      } catch (err) {
        showTyping(false);
        showError('Connection error: ' + err.message);
      }

      setInputEnabled(true);
      msgInput.focus();
    }

    // Enter key to send
    msgInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Profile fields: Enter to proceed
    document.getElementById('name-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') document.getElementById('prof-input').focus();
    });
    document.getElementById('prof-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') startChat();
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
