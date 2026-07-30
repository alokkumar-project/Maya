
from flask import Flask, request, jsonify, render_template
import traceback
import os
from datetime import datetime

CHAT_FOLDER = "chat_history"
os.makedirs(CHAT_FOLDER, exist_ok=True)

def save_chat(ip, user_msg, bot_msg, mode):
    filename = os.path.join(CHAT_FOLDER, f"{ip}.txt")

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"\n[{datetime.now()}]\n")
        file.write(f"MODE : {mode}\n")
        file.write(f"USER : {user_msg}\n")
        file.write(f"MAYA : {bot_msg}\n")
        file.write("-" * 50 + "\n")

TOKEN_FILE = "all_token.pkl"
MODEL_FILE = "maya_v1.pkl"

app = Flask(__name__)

chatbot = None
load_error = None

try:
    from chatbot_engine import load_chatbot
    chatbot = load_chatbot(token_file=TOKEN_FILE, model_file=MODEL_FILE)
except Exception as exc:
    load_error = str(exc)
    print("=" * 70)
    print("Could not load the chatbot model at startup:")
    traceback.print_exc()
    print("The website will still run, but /api/chat will return an error")
    print("until your model files and modules are placed in this folder.")
    print("=" * 70)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    mode = data.get("mode", "beam")
    if mode not in ("beam", "greedy"):
        mode = "beam"

    if chatbot is None:
        return jsonify({"error": f"Model not loaded: {load_error}"}), 503

    try:
        reply = chatbot.respond(message, mode=mode)

        # Get user IP
        ip = request.remote_addr

        # Save conversation
        save_chat(ip, message, reply, mode)

        return jsonify({"reply": reply, "mode": mode})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

@app.route("/api/health")
def health():
    return jsonify({"model_loaded": chatbot is not None, "error": load_error})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
