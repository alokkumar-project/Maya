

# =====================================================================
# 1. IMPORTS & CONFIGURATION
# =====================================================================
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import traceback
import os
import time
import threading
import queue
from functools import wraps
from datetime import datetime, date
from collections import deque, Counter

try:
    import psutil
except ImportError:
    psutil = None

try:
    import requests
except ImportError:
    requests = None

# ---- Chatbot model files (unchanged) ----
TOKEN_FILE = "all_token.pkl"
MODEL_FILE = "maya_v1.pkl"

# ---- Chat history storage (unchanged) ----
CHAT_FOLDER = "chat_history"
os.makedirs(CHAT_FOLDER, exist_ok=True)

# ---- Admin config (env vars only - never hardcode secrets) ----
ADMIN_PASSWORD = os.environ.get("MAYA_ADMIN_PASSWORD")
SECRET_KEY = os.environ.get("MAYA_SECRET_KEY")

if not ADMIN_PASSWORD:
    raise RuntimeError("MAYA_ADMIN_PASSWORD environment variable is not set.")

if not SECRET_KEY:
    raise RuntimeError("MAYA_SECRET_KEY environment variable is not set.")


# A user counts as "active" if they were seen within this window.
ACTIVE_WINDOW_SECONDS = 5 * 60

# How long to cache a resolved country for a given IP (avoids re-querying).
COUNTRY_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h


# =====================================================================
# 2. FLASK APP + CHATBOT MODEL LOADING (UNCHANGED)
# =====================================================================
app = Flask(__name__)
app.secret_key = SECRET_KEY

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


# =====================================================================
# 3. CLIENT IP HELPER (Render-safe)
# =====================================================================
def get_client_ip():
    """
    On Render (and most reverse proxies), request.remote_addr is the proxy's
    IP, not the visitor's. The real client IP is the first entry in
    X-Forwarded-For. Falls back to remote_addr for local/dev use.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


# =====================================================================
# 4. CHAT HISTORY PERSISTENCE (UNCHANGED)
# =====================================================================
def save_chat(ip, user_msg, bot_msg, mode):
    filename = os.path.join(CHAT_FOLDER, f"{ip}.txt")

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"\n[{datetime.now()}]\n")
        file.write(f"MODE : {mode}\n")
        file.write(f"USER : {user_msg}\n")
        file.write(f"MAYA : {bot_msg}\n")
        file.write("-" * 50 + "\n")


# =====================================================================
# 5. ADMIN STATS ENGINE
# =====================================================================
#
# Design goals (per requirements):
#   - Every update from the request path is O(1) and never blocks on network I/O.
#   - Country resolution (a network call) happens on a background worker thread,
#     fed by a queue. The chat request never waits for it.
#   - CPU/RAM reads only happen when the dashboard polls /api/admin/stats,
#     never during /api/chat.
#   - A single lock guards shared state; critical sections are kept tiny.
#   - Inactive users are swept out lazily (bounded work, only when computing
#     the active-user count), so the active_users dict never grows unbounded.
#
class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()

        self.total_visitors = set()          # unique IPs ever seen
        self.active_users = {}               # ip -> last_seen timestamp
        self.total_messages = 0
        self.messages_today = 0
        self._today = date.today()

        self.response_times = deque(maxlen=200)   # bounded, O(1) append
        self.recent_chats = deque(maxlen=50)       # bounded, O(1) append
        self.question_counter = Counter()
        self.country_counter = Counter()

        # country_cache: ip -> (country_name, resolved_at_timestamp)
        self._country_cache = {}
        self._country_queue = queue.Queue()

        # Background worker: resolves countries without ever blocking a request.
        self._worker = threading.Thread(target=self._country_worker, daemon=True)
        self._worker.start()

    # ---------------- fast path (called from the request cycle) ----------------

    def _roll_day_if_needed(self):
        today = date.today()
        if today != self._today:
            self._today = today
            self.messages_today = 0

    def record_visit(self, ip):
        """O(1). Called on every page/API hit to track visitors + active users."""
        now = time.time()
        with self._lock:
            self.total_visitors.add(ip)
            self.active_users[ip] = now

    def record_message(self, ip, message, response_time):
        """
        O(1), no network calls. Enqueues the IP for background country
        resolution instead of resolving it inline.
        """
        with self._lock:
            self._roll_day_if_needed()
            self.total_messages += 1
            self.messages_today += 1
            self.response_times.append(response_time)
            self.recent_chats.append({
                "time": datetime.now().strftime("%I:%M %p"),
                "ip": ip,
                "message": message,
            })
            clean = (message or "").strip().lower()
            if clean:
                self.question_counter[clean] += 1

        # Enqueue for async country resolution (non-blocking put on an
        # unbounded queue -> O(1), never touches the network here).
        self._country_queue.put(ip)

    # ---------------- background worker (runs off the request thread) ----------------

    def _country_worker(self):
        while True:
            ip = self._country_queue.get()
            try:
                country = self._resolve_country(ip)
                with self._lock:
                    self.country_counter[country] += 1
            except Exception:
                # Never let a bad lookup crash the worker thread.
                pass
            finally:
                self._country_queue.task_done()

    def _resolve_country(self, ip):
        """Runs only on the background thread. Cached, rate-limited, safe to be slow."""
        cached = self._country_cache.get(ip)
        if cached and (time.time() - cached[1] < COUNTRY_CACHE_TTL_SECONDS):
            return cached[0]

        is_private = (
            not ip
            or ip in ("unknown", "localhost")
            or ip.startswith("127.")
            or ip.startswith("192.168.")
            or ip.startswith("10.")
            or ip.startswith("172.16.")
        )

        country = "Local"
        if not is_private:
            country = "Unknown"
            if requests is not None:
                try:
                    r = requests.get(
                        f"http://ip-api.com/json/{ip}?fields=status,country,countryCode",
                        timeout=2,
                    )
                    data = r.json()
                    if data.get("status") == "success" and data.get("country"):
                        country = data["country"]
                except Exception:
                    pass

        self._country_cache[ip] = (country, time.time())
        return country

    # ---------------- read path (called only by /api/admin/stats) ----------------

    def _sweep_inactive(self):
        """Removes stale entries from active_users. Bounded by active_users size."""
        cutoff = time.time() - ACTIVE_WINDOW_SECONDS
        stale = [ip for ip, seen in self.active_users.items() if seen < cutoff]
        for ip in stale:
            del self.active_users[ip]

    def snapshot(self, model_loaded):
        with self._lock:
            self._roll_day_if_needed()
            self._sweep_inactive()

            active_count = len(self.active_users)
            total_visitors = len(self.total_visitors)
            total_messages = self.total_messages
            messages_today = self.messages_today
            avg_resp = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )
            top_questions = [q for q, _ in self.question_counter.most_common(6)]
            recent_chats = list(self.recent_chats)[-10:][::-1]
            countries = self.country_counter.most_common(8)

        # CPU/RAM reads happen outside the lock and only here (dashboard poll),
        # never in the /api/chat path.
        ram_mb, cpu_percent = None, None
        if psutil is not None:
            try:
                process = psutil.Process(os.getpid())
                ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)
                # interval=None -> non-blocking, uses time elapsed since the
                # last call. Never sleeps, so it can't slow anything down.
                cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                pass

        return {
            "status": "online",
            "active_users": active_count,
            "total_visitors": total_visitors,
            "messages_today": messages_today,
            "total_messages": total_messages,
            "model_loaded": model_loaded,
            "avg_response": round(avg_resp, 3),
            "ram_mb": ram_mb,
            "cpu_percent": cpu_percent,
            "uptime_seconds": int(time.time() - self.start_time),
            "recent_chats": recent_chats,
            "top_questions": top_questions,
            "countries": countries,
        }


stats = Stats()


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapper


# =====================================================================
# 6. PUBLIC ROUTES (chatbot contract is 100% unchanged)
# =====================================================================
@app.route("/")
def index():
    stats.record_visit(get_client_ip())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    mode = data.get("mode", "beam")
    if mode not in ("beam", "greedy"):
        mode = "beam"

    ip = get_client_ip()
    stats.record_visit(ip)

    if chatbot is None:
        return jsonify({"error": f"Model not loaded: {load_error}"}), 503

    try:
        start = time.time()
        reply = chatbot.respond(message, mode=mode)          # <-- untouched
        elapsed = time.time() - start

        # Save conversation (unchanged)
        save_chat(ip, message, reply, mode)

        # Record stats: O(1), no network calls on this thread.
        stats.record_message(ip, message, elapsed)

        return jsonify({"reply": reply, "mode": mode})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"model_loaded": chatbot is not None, "error": load_error})


# =====================================================================
# 7. ADMIN ROUTES
# =====================================================================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if password and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            next_url = request.args.get("next") or url_for("admin_dashboard")
            return redirect(next_url)
        error = "Incorrect password."
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template("admin.html")


@app.route("/api/admin/stats")
@admin_required
def admin_stats():
    return jsonify(stats.snapshot(model_loaded=chatbot is not None))


# =====================================================================
# 8. ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    # Render sets PORT automatically; default to 5000 for local dev.
    port = int(os.environ.get("PORT", 5500))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
