(() => {
  const STORAGE_KEY = "assistant_conversations_v1";

  const sidebar = document.getElementById("sidebar");
  const historyList = document.getElementById("historyList");
  const searchInput = document.getElementById("searchInput");
  const newChatBtn = document.getElementById("newChatBtn");
  const collapseBtn = document.getElementById("collapseBtn");
  const openSidebarBtn = document.getElementById("openSidebarBtn");
  const chatTitle = document.getElementById("chatTitle");
  const renameBtn = document.getElementById("renameBtn");
  const messagesEl = document.getElementById("messages");
  const emptyState = document.getElementById("emptyState");
  const composerForm = document.getElementById("composerForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const rowTemplate = document.getElementById("conversationRowTemplate");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const decodeToggle = document.getElementById("decodeToggle");
  const downloadBtn = document.getElementById("downloadBtn");

  const MODE_KEY = "assistant_decode_mode";
  let decodeMode = localStorage.getItem(MODE_KEY) || "beam";

  let conversations = [];
  let activeId = null;

  // ---------- persistence ----------
  function loadConversations() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      conversations = raw ? JSON.parse(raw) : [];
    } catch (e) {
      conversations = [];
    }
  }

  function saveConversations() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  }

  function createConversation() {
    const convo = {
      id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
      title: "New conversation",
      messages: [],
      updatedAt: Date.now(),
    };
    conversations.unshift(convo);
    activeId = convo.id;
    saveConversations();
    renderHistory();
    renderActiveConversation();
    messageInput.focus();
  }

  function getActive() {
    return conversations.find((c) => c.id === activeId) || null;
  }

  function deleteConversation(id, evt) {
    evt.stopPropagation();
    conversations = conversations.filter((c) => c.id !== id);
    if (activeId === id) {
      activeId = conversations.length ? conversations[0].id : null;
    }
    saveConversations();
    renderHistory();
    renderActiveConversation();
  }

  function selectConversation(id) {
    activeId = id;
    renderHistory();
    renderActiveConversation();
    if (window.innerWidth <= 820) sidebar.classList.remove("open");
  }

  // ---------- rendering ----------
  function renderHistory() {
    const query = searchInput.value.trim().toLowerCase();
    historyList.innerHTML = "";

    const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
    const filtered = query
      ? sorted.filter((c) => c.title.toLowerCase().includes(query))
      : sorted;

    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "history-label";
      empty.style.opacity = "0.7";
      empty.textContent = query ? "No matches" : "No conversations yet";
      historyList.appendChild(empty);
      return;
    }

    filtered.forEach((c, idx) => {
      const node = rowTemplate.content.cloneNode(true);
      const btn = node.querySelector(".history-item");
      const titleEl = node.querySelector(".history-item-title");
      const delBtn = node.querySelector(".history-item-delete");

      titleEl.textContent = c.title;
      btn.style.animationDelay = `${Math.min(idx * 30, 300)}ms`;
      if (c.id === activeId) btn.classList.add("active");

      btn.addEventListener("click", () => selectConversation(c.id));
      delBtn.addEventListener("click", (e) => deleteConversation(c.id, e));

      historyList.appendChild(node);
    });
  }

  function renderActiveConversation() {
    const convo = getActive();
    messagesEl.innerHTML = "";

    if (!convo) {
      chatTitle.textContent = "New conversation";
      messagesEl.appendChild(emptyState);
      return;
    }

    chatTitle.textContent = convo.title;

    if (!convo.messages.length) {
      messagesEl.appendChild(emptyState);
      return;
    }

    const list = document.createElement("div");
    list.className = "msg-list";

    convo.messages.forEach((m) => {
      list.appendChild(buildMessageEl(m.role, m.text, m.error));
    });

    messagesEl.appendChild(list);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function buildMessageEl(role, text, isError) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble" + (isError ? " error" : "");
    bubble.textContent = text;
    wrap.appendChild(bubble);
    return wrap;
  }

  function buildTypingEl() {
    const wrap = document.createElement("div");
    wrap.className = "msg bot";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
    wrap.appendChild(bubble);
    return wrap;
  }

  function appendNode(node) {
    let list = messagesEl.querySelector(".msg-list");
    if (!list) {
      messagesEl.innerHTML = "";
      list = document.createElement("div");
      list.className = "msg-list";
      messagesEl.appendChild(list);
    }
    list.appendChild(node);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return node;
  }

  // ---------- sending messages ----------
  async function sendMessage(text) {
    let convo = getActive();
    if (!convo) {
      createConversation();
      convo = getActive();
    }

    convo.messages.push({ role: "user", text, ts: Date.now() });
    if (convo.messages.length === 1) {
      convo.title = text.slice(0, 42) + (text.length > 42 ? "…" : "");
      chatTitle.textContent = convo.title;
    }
    convo.updatedAt = Date.now();
    saveConversations();
    renderHistory();
    appendNode(buildMessageEl("user", text));

    const typingEl = appendNode(buildTypingEl());
    sendBtn.disabled = true;
    sendBtn.classList.add("sent");
    setTimeout(() => sendBtn.classList.remove("sent"), 300);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, mode: decodeMode }),
      });
      const data = await res.json();

      typingEl.remove();

      if (!res.ok || data.error) {
        const errText = data.error || "Something went wrong.";
        appendNode(buildMessageEl("bot", errText, true));
        convo.messages.push({ role: "bot", text: errText, ts: Date.now(), error: true });
      } else {
        appendNode(buildMessageEl("bot", data.reply));
        convo.messages.push({ role: "bot", text: data.reply, ts: Date.now() });
      }
      convo.updatedAt = Date.now();
      saveConversations();
      renderHistory();
    } catch (err) {
      typingEl.remove();
      const errText = "Could not reach the server. Is app.py running?";
      appendNode(buildMessageEl("bot", errText, true));
      convo.messages.push({ role: "bot", text: errText, ts: Date.now(), error: true });
      saveConversations();
    } finally {
      sendBtn.disabled = false;
    }
  }

  // ---------- events ----------
  decodeToggle.querySelectorAll(".decode-option").forEach((btn) => {
    btn.addEventListener("click", () => {
      decodeMode = btn.dataset.mode;
      localStorage.setItem(MODE_KEY, decodeMode);
      applyDecodeModeUI();
    });
  });

  function applyDecodeModeUI() {
    decodeToggle.querySelectorAll(".decode-option").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === decodeMode);
    });
    decodeToggle.classList.toggle("mode-greedy", decodeMode === "greedy");
  }

  downloadBtn.addEventListener("click", () => {
    const convo = getActive();
    if (!convo || !convo.messages.length) return;

    const lines = [`Conversation: ${convo.title}`, `Exported: ${new Date().toLocaleString()}`, ""];
    convo.messages.forEach((m) => {
      const who = m.role === "user" ? "You" : "Assistant";
      lines.push(`${who}: ${m.text}`);
      lines.push("");
    });

    const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeTitle = convo.title.replace(/[^a-z0-9\-_ ]/gi, "").trim().slice(0, 40) || "conversation";
    a.href = url;
    a.download = `${safeTitle}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  newChatBtn.addEventListener("click", (e) => {
    const ripple = document.createElement("span");
    ripple.className = "ripple";
    const rect = newChatBtn.getBoundingClientRect();
    ripple.style.left = `${e.clientX - rect.left - 10}px`;
    ripple.style.top = `${e.clientY - rect.top - 10}px`;
    ripple.style.width = ripple.style.height = "20px";
    newChatBtn.appendChild(ripple);
    setTimeout(() => ripple.remove(), 500);
    createConversation();
  });

  collapseBtn.addEventListener("click", () => sidebar.classList.toggle("collapsed"));
  openSidebarBtn.addEventListener("click", () => sidebar.classList.toggle("open"));

  searchInput.addEventListener("input", renderHistory);

  renameBtn.addEventListener("click", () => {
    const convo = getActive();
    if (!convo) return;
    const name = prompt("Rename conversation", convo.title);
    if (name && name.trim()) {
      convo.title = name.trim();
      saveConversations();
      renderHistory();
      chatTitle.textContent = convo.title;
    }
  });

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text) return;
    messageInput.value = "";
    messageInput.style.height = "auto";
    sendMessage(text);
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerForm.requestSubmit();
    }
  });

  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
  });

  // ---------- health check ----------
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.model_loaded) {
        statusDot.className = "status-dot ok";
        statusText.textContent = "Model ready";
      } else {
        statusDot.className = "status-dot error";
        statusText.textContent = "Model not loaded";
      }
    } catch (e) {
      statusDot.className = "status-dot error";
      statusText.textContent = "Server unreachable";
    }
  }

  // ---------- init ----------
  applyDecodeModeUI();
  loadConversations();
  if (conversations.length) {
    activeId = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
  }
  renderHistory();
  renderActiveConversation();
  checkHealth();
})();
