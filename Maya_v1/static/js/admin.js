(() => {
  const REFRESH_MS = 5000;

  const el = {
    status: document.getElementById("s-status"),
    statusDot: document.getElementById("statusDotAdmin"),
    active: document.getElementById("s-active"),
    visitors: document.getElementById("s-visitors"),
    msgToday: document.getElementById("s-msg-today"),
    msgTotal: document.getElementById("s-msg-total"),
    model: document.getElementById("s-model"),
    avgResp: document.getElementById("s-avg-resp"),
    ram: document.getElementById("s-ram"),
    cpu: document.getElementById("s-cpu"),
    uptime: document.getElementById("s-uptime"),
    recentChats: document.getElementById("recentChats"),
    topQuestions: document.getElementById("topQuestions"),
    countries: document.getElementById("countries"),
  };

  // Best-effort country name -> flag emoji mapping for common cases.
  const COUNTRY_FLAGS = {
    "India": "🇮🇳", "United States": "🇺🇸", "United Kingdom": "🇬🇧",
    "Canada": "🇨🇦", "Australia": "🇦🇺", "Germany": "🇩🇪", "France": "🇫🇷",
    "Brazil": "🇧🇷", "China": "🇨🇳", "Japan": "🇯🇵", "Russia": "🇷🇺",
    "Pakistan": "🇵🇰", "Bangladesh": "🇧🇩", "Nigeria": "🇳🇬",
    "Indonesia": "🇮🇩", "Mexico": "🇲🇽", "Singapore": "🇸🇬",
    "Netherlands": "🇳🇱", "UAE": "🇦🇪", "Local": "🖥️", "Unknown": "🌐",
  };

  function flagFor(name) {
    return COUNTRY_FLAGS[name] || "🏳️";
  }

  function fmt(n) {
    if (n === null || n === undefined) return "—";
    return typeof n === "number" ? n.toLocaleString() : n;
  }

  function fmtUptime(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
  }

  function render(data) {
    el.status.textContent = data.status === "online" ? "Online" : "Offline";
    el.statusDot.style.background = data.status === "online" ? "var(--ok)" : "var(--danger)";

    el.active.textContent = fmt(data.active_users);
    el.visitors.textContent = fmt(data.total_visitors);
    el.msgToday.textContent = fmt(data.messages_today);
    el.msgTotal.textContent = fmt(data.total_messages);
    el.model.textContent = data.model_loaded ? "Yes" : "No";
    el.avgResp.textContent = `${data.avg_response.toFixed(3)} s`;
    el.ram.textContent = data.ram_mb !== null ? `${data.ram_mb} MB` : "N/A";
    el.cpu.textContent = data.cpu_percent !== null ? `${Math.round(data.cpu_percent)}%` : "N/A";
    el.uptime.textContent = fmtUptime(data.uptime_seconds);

    // Recent chats
    if (data.recent_chats && data.recent_chats.length) {
      el.recentChats.innerHTML = data.recent_chats.map((c) => `
        <div class="chat-row">
          <span class="chat-time">${escapeHtml(c.time)}</span>
          <span class="chat-ip">${escapeHtml(c.ip)}</span>
          <span class="chat-msg">${escapeHtml(c.message)}</span>
        </div>
      `).join("");
    } else {
      el.recentChats.innerHTML = `<div class="empty-row">No messages yet.</div>`;
    }

    // Top questions
    if (data.top_questions && data.top_questions.length) {
      el.topQuestions.innerHTML = data.top_questions.map((q) => `
        <div class="question-row">${escapeHtml(q)}</div>
      `).join("");
    } else {
      el.topQuestions.innerHTML = `<div class="empty-row">No data yet.</div>`;
    }

    // Countries
    if (data.countries && data.countries.length) {
      el.countries.innerHTML = data.countries.map(([name, count]) => `
        <div class="country-row">
          <span><span class="flag">${flagFor(name)}</span>${escapeHtml(name)}</span>
          <span>${fmt(count)}</span>
        </div>
      `).join("");
    } else {
      el.countries.innerHTML = `<div class="empty-row">No data yet.</div>`;
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch("/api/admin/stats");
      if (res.redirected) {
        window.location.href = res.url;
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      render(data);
    } catch (e) {
      // Silently ignore transient network errors; the next poll retries.
    }
  }

  fetchStats();
  setInterval(fetchStats, REFRESH_MS);
})();
