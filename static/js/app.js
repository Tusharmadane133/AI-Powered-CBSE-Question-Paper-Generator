const $ = id => document.getElementById(id);
const P = {
  logo: '<path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 2.5 9 2.5 12 0v-5"/>',
  sparkles: '<path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 016.5 17H20V3H6.5A2.5 2.5 0 004 5.5z"/><path d="M4 19.5A2.5 2.5 0 006.5 22H20v-5"/>',
  flask: '<path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 001.3 2h12.4a1.5 1.5 0 001.3-2L14 9V3"/>',
  math: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 7h8M8 12h2M14 12h2M8 16h2M14 16h2"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 010 18M12 3a14 14 0 000 18"/>',
  type: '<path d="M4 7V5h16v2M12 5v14M9 19h6"/>',
  lang: '<path d="M5 8l6 6M4 14l6-6 2-3M2 5h12M7 2v3M22 22l-5-10-5 10M14 18h6"/>',
  all: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 015 .5c0 1.5-2.5 2-2.5 3.5M12 17h.01"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/>'
};
const svg = (n, cls) => `<svg ${cls ? `class="${cls}" ` : ""}viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${P[n]}</svg>`;
const subjectIcon = s => { s = s.toLowerCase(); return s.includes("social") ? "globe" : s.includes("sci") ? "flask" : s.includes("math") ? "math" : s.includes("english") ? "type" : /hindi|marathi|sanskrit|urdu/.test(s) ? "lang" : "book"; };
/* ---------- safe storage helpers ---------- */
const store = {
  get(k, d = "") { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} }
};
let SUBJECTS = {}, chats = [], cur = null, msgsBox = null, stick = true;
const lives = new Map();
function attachResponseTimer(lv, col, label = "Thinking") {
  if (!lv.timerEl) {
    lv.timerEl = el("div", "muted small response-time", label + " · 0s");
    lv.timerEl.style.marginTop = "6px";
    lv.timerEl.style.fontSize = "12px";
    lv.timerEl.style.opacity = "0.72";
    lv.timerEl.style.lineHeight = "1.4";
  }
  if (lv.startedAt == null) lv.startedAt = performance.now();
  lv.timerLabel = label;
  if (col && lv.timerEl.parentElement !== col) col.appendChild(lv.timerEl);
  if (!lv.timerInterval) {
    lv.timerInterval = setInterval(() => {
      if (lv.timerEl && lv.startedAt != null) {
        const seconds = (performance.now() - lv.startedAt) / 1000;
        lv.timerEl.textContent = `${lv.timerLabel || "Thinking"} · ${Math.ceil(seconds)}s`;
      }
    }, 100);
  }
}
function stopResponseTimer(lv) {
  if (lv.timerInterval) {
    clearInterval(lv.timerInterval);
    lv.timerInterval = null;
  }
  if (lv.timerEl && lv.startedAt != null) {
    const seconds = (performance.now() - lv.startedAt) / 1000;
    lv.timerEl.textContent = `Thought for ${Math.max(1, Math.round(seconds))}s`;
  }
}
   // chat id -> running answer (each chat can generate on its own)
const BASE_TITLE = document.title;
let CLS = store.get("cls"), SUB = store.get("sub");
try { chats = JSON.parse(store.get("chats", "[]")); if (!Array.isArray(chats)) chats = []; } catch { chats = []; }
try { const s = JSON.parse(store.get("subjects", "{}")); if (s && typeof s === "object" && !Array.isArray(s)) SUBJECTS = s; } catch {}   // instant sidebar on load
try { history.scrollRestoration = "manual"; } catch {}
/* chat id like ChatGPT: long random UUID in the URL (/c/3f2b8c1e-....) */
const newId = () => (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
  : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === "x" ? r : (r & 3 | 8)).toString(16); });
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
async function* ndjson(res) {
  const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
  while (true) {
    const {done, value} = await reader.read(); if (done) break;
    buf += dec.decode(value, {stream: true});
    const lines = buf.split("\n"); buf = lines.pop();
    for (const l of lines) if (l.trim()) { try { yield JSON.parse(l); } catch {} }
  }
  buf += dec.decode();
  if (buf.trim()) { try { yield JSON.parse(buf); } catch {} }
}
const save = () => { store.set("chats", JSON.stringify(chats.slice(0, 40))); };
let unloading = false;   // reload/close in progress: do not mark running answers as failed
addEventListener("beforeunload", () => { unloading = true; });
addEventListener("pagehide", () => { unloading = true; });
/* URL per chat (like ChatGPT): /c/<id> -> refresh and back/forward keep the chat */
function pushUrl(id) { const want = id ? "/chat/" + id : "/"; if (location.pathname === want && !location.hash) return; history.pushState(null, "", want); }
function updateDocTitle() { document.title = cur && cur.title ? cur.title + " - " + BASE_TITLE : BASE_TITLE; }
function setBusy(b) { const s = $("send"); s.innerHTML = svg(b ? "stop" : "send"); s.title = b ? "Stop generating" : "Send"; s.setAttribute("aria-label", s.title); }
/* ---------- tiny safe markdown (bold, code, lists, headings) ---------- */
function inline(parent, text) {
  text.split(/(\*\*[^*]+\*\*|`[^`]+`|10<sup>[^<]*<\/sup>)/g).forEach(p => {
    if (!p) return;
    if (p.length > 4 && p.startsWith("**") && p.endsWith("**")) parent.appendChild(el("strong", "", p.slice(2, -2)));
    else if (p.length > 2 && p.startsWith("`") && p.endsWith("`")) parent.appendChild(el("code", "", p.slice(1, -1)));
    else {
      const exponent = p.match(/^10<sup>([^<]*)<\/sup>$/);
      if (exponent) {
        parent.appendChild(document.createTextNode("10"));
        parent.appendChild(el("sup", "", exponent[1]));
      } else parent.appendChild(document.createTextNode(p));
    }
  });
}
const MATH = {delimiters: [{left: "\\\\\\\\[", right: "\\\\\\\\]", display: true}, {left: "$$", right: "$$", display: true}, {left: "\\\\\\\\(", right: "\\\\\\\\)", display: false}], throwOnError: false};
function mathify(box) { if (window.renderMathInElement) { try { window.renderMathInElement(box, MATH); } catch {} } }
// list marker: "- x", "\* x", "• x", "1. x", "1) x" and also "3.The result" (no space, a letter follows)
const LIST_RE = /^\s*(?:([-*•])\s+|(\d{1,2})[.)](?:\s+|(?=[A-Za-z])))(.*)$/;
function normalizeScientificNotation(text) {
  return String(text)
    // Repair years broken across lines or blank paragraphs, including streamed/model formatting.
    .replace(/\b(17|18|19|20)(?:[ \t\u00a0]*\r?\n[ \t\u00a0]*)+(\d{2})(?=\b|[.,;:!?])/g, "$1$2")
    .replace(/\b(17|18|19|20)[ \t\u00a0]+(\d{2})(?=\b|[.,;:!?])/g, "$1$2")
    .replace(/10\s*[–−-]\s*(8|6|5|4|3|2|1|0)\b/g, (_, e) => `10<sup>−${e}</sup>`)
    .replace(/10\s*\^\s*\(?\s*−?([0-9]+)\s*\)?/g, (_, e) => `10<sup>−${e}</sup>`)
    .replace(/\b(10)(12|13|14|15|16|17)\b/g, (_, base, e) => `${base}<sup>${e}</sup>`);
}
function renderMd(box, text) {
  text = normalizeScientificNotation(text);
  box.replaceChildren(); let list = null, lt = null;
  text = text.replace(/(?<!\d)(\d{1,2})\.\s+(?=[A-Za-z])/g, "\n$1. "); // normalize numbered list items without splitting years like 1789.
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (m, a) => "\\[" + a.replace(/\n/g, " ") + "\\]").replace(/\$\$([\s\S]*?)\$\$/g, (m, a) => "$$" + a.replace(/\n/g, " ") + "$$");
  const lines = text.split("\n");
  const cells = line => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(x => x.trim());
  const isTableRow = line => /^\s*\|.*\|\s*$/.test(line);
  const isSeparator = line => isTableRow(line) && cells(line).length > 0 && cells(line).every(c => /^:?-{3,}:?$/.test(c));
  const renderTable = rows => {
    const table = el("table", "markdown-table");
    table.style.borderCollapse = "collapse";
    table.style.width = "100%";
    table.style.margin = "10px 0";
    table.style.fontSize = "0.95em";
    table.style.display = "block";
    table.style.overflowX = "auto";
    rows.forEach((row, ri) => {
      const tr = el("tr");
      row.forEach(value => {
        const cell = el(ri === 0 ? "th" : "td");
        cell.style.border = "1px solid var(--border, #d1d5db)";
        cell.style.padding = "8px 10px";
        cell.style.textAlign = "left";
        cell.style.verticalAlign = "top";
        if (ri === 0) {
          cell.style.fontWeight = "700";
          cell.style.background = "var(--surface, #eef2f7)";
        }
        inline(cell, value);
        tr.appendChild(cell);
      });
      table.appendChild(tr);
    });
    box.appendChild(table);
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Markdown table: header row followed by a separator row.
    if (isTableRow(line) && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      list = null; lt = null;
      const rows = [cells(line)];
      i += 2; // skip the header separator
      while (i < lines.length && isTableRow(lines[i])) {
        if (!isSeparator(lines[i])) rows.push(cells(lines[i]));
        i++;
      }
      i--;
      renderTable(rows);
      continue;
    }
    // Also support pipe-delimited comparison rows when the model omits Markdown's separator.
    if (isTableRow(line) && /^\s*\|\s*(feature|difference|property|criteria)\s*\|/i.test(line)) {
      const rows = [cells(line)];
      let j = i + 1;
      while (j < lines.length && isTableRow(lines[j])) {
        if (!isSeparator(lines[j])) rows.push(cells(lines[j]));
        j++;
      }
      if (rows.length > 1) {
        list = null; lt = null; renderTable(rows); i = j - 1; continue;
      }
    }
    const m = line.match(LIST_RE);
    if (m) {
      const t = m[2] ? "ol" : "ul";
      if (!list || lt !== t) {
        list = el(t); lt = t;
        if (t === "ol" && Number(m[2]) > 1) list.start = Number(m[2]);
        box.appendChild(list);
      }
      const li = el("li"); inline(li, m[3]); list.appendChild(li);
    } else {
      if (!line.trim()) continue;
      list = null; lt = null;
      const h = line.match(/^#{1,4}\s+(.\*)$/), p = el(h ? "h4" : "p");
      inline(p, h ? h[1] : line); box.appendChild(p);
    }
  }
  mathify(box);
}
/* ---------- theme / sidebar ---------- */
function setThemeIcon() { $("theme").innerHTML = svg(document.documentElement.dataset.theme === "dark" ? "sun" : "moon"); }
$("theme").onclick = () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = t; store.set("theme", t); setThemeIcon(); };
$("menu").onclick = () => document.body.classList.toggle("open");
$("scrim").onclick = () => document.body.classList.remove("open");
/* ---------- class / subject filters (remembered across refresh) ---------- */
function subjectsFor(cls) { return cls ? (SUBJECTS[cls] || []) : [...new Set(Object.values(SUBJECTS).flat())].sort(); }
function saveFilters() { store.set("cls", CLS); store.set("sub", SUB); }
function updateCtx() { $("ctxTitle").textContent = SUB || "All subjects"; $("ctxSub").textContent = CLS ? "Class " + CLS : "All classes"; }
function renderClasses() {
  const seg = $("classSeg"); seg.replaceChildren();
  const keys = Object.keys(SUBJECTS).sort((a, b) => a - b);
  [["", "All"], ...keys.map(k => [k, k])].forEach(([v, label]) => {
    const b = el("button", v === CLS ? "on" : "", label); b.type = "button";
    b.onclick = () => { CLS = v; if (SUB && !subjectsFor(CLS).includes(SUB)) SUB = ""; saveFilters(); renderClasses(); renderSubjects(); updateCtx(); };
    seg.appendChild(b);
  });
}
function renderSubjects() {
  const box = $("subjectList"); box.replaceChildren();
  [["", "All subjects", "all"], ...subjectsFor(CLS).map(s => [s, s, subjectIcon(s)])].forEach(([v, label, icon]) => {
    const b = el("button", "sub" + (v === SUB ? " on" : "")); b.type = "button";
    b.insertAdjacentHTML("afterbegin", svg(icon)); b.appendChild(el("span", "", label));
    b.onclick = () => { SUB = v; saveFilters(); renderSubjects(); updateCtx(); document.body.classList.remove("open"); };
    box.appendChild(b);
  });
}
/* ---------- chat list ---------- */
function renderHistory() {
  const h = $("history"); h.replaceChildren();
  if (!chats.length) { h.appendChild(el("div", "muted small", "No chats yet")).style.padding = "6px 10px"; return; }
  chats.forEach(c => {
    const it = el("div", "item" + (cur && cur.id === c.id ? " on" : "")); it.appendChild(el("span", "", c.title || "New chat"));
    const x = el("b", "", "✕"); x.title = "Delete";
    x.onclick = e => { e.stopPropagation(); chats = chats.filter(k => k.id !== c.id); save(); if (cur && cur.id === c.id) newChat(); else renderHistory(); };
    it.appendChild(x); it.onclick = () => openChat(c.id); h.appendChild(it);
  });
}
function emptyState() {
  $("chat").replaceChildren(); msgsBox = null;
  const e = el("div", "empty");
  e.insertAdjacentHTML("afterbegin", `<div class="brand-logo">${svg("logo")}</div>`);
  e.appendChild(el("h1", "", "What do you want to study today?"));
  e.appendChild(el("p", "", "Ask any question from your Class 8, 9 or 10 textbooks."));
  $("chat").appendChild(e);
}
function newChat(push = true) {
  cur = {id: newId(), title: "", messages: []};
  if (push) pushUrl(null);
  setBusy(false); emptyState(); renderHistory(); updateDocTitle();
  document.body.classList.remove("open"); $("q").focus();
}
function openChat(id, push = true) {
  const c = chats.find(k => String(k.id) === String(id));
  if (!c) return newChat(push);
  cur = c; if (push) pushUrl(c.id);
  $("chat").replaceChildren(); msgsBox = null;
  if (!cur.messages.length) emptyState();
  else cur.messages.forEach(m => {
    if (m.role === "user") return userMsg(m.text);
    const ui = botMsg(m.error ? "" : m.text, m.sources, m.partial || m.error ? undefined : m.q);
    if (m.error) { ui.body.classList.add("err"); ui.body.textContent = m.text; }
    const lv = lives.get(cur.id);
    if (m.partial && lv && lv.msg === m) { lv.ui = ui; attachResponseTimer(lv, ui.col, lv.timerLabel || "Response time"); if (!m.text) ui.body.classList.add("dots"); }
  });
  stick = true; setBusy(lives.has(cur.id)); renderHistory(); updateDocTitle(); document.body.classList.remove("open"); scrollEnd(true);
}
function route() {
  const m = location.pathname.match(/^\/(?:chat|c)\/([\w-]+)\/?$/) || location.hash.match(/^#\/(?:chat|c)\/([\w-]+)$/), id = m && m[1];
  const found = id && chats.find(c => String(c.id) === id);
  if (found) openChat(found.id, false);
  else { if (location.pathname !== "/" || location.hash) history.replaceState(null, "", "/"); newChat(false); }
}
window.addEventListener("popstate", route);
$("newChat").onclick = () => {
  if (location.pathname === "/" && !location.hash) location.reload();
  else location.href = "/";
};
/* ---------- messages ---------- */
function box() { if (!msgsBox) { $("chat").replaceChildren(); msgsBox = el("div", "msgs"); $("chat").appendChild(msgsBox); } return msgsBox; }
const scrollEnd = force => { if (force || stick) $("chat").scrollTop = $("chat").scrollHeight; };
$("chat").addEventListener("scroll", () => { const c = $("chat"); stick = c.scrollHeight - c.scrollTop - c.clientHeight < 80; });
function userMsg(text) { const r = el("div", "m-user"); r.appendChild(el("div", "", text)); box().appendChild(r); scrollEnd(true); }
function srcChips(srcs) {
  const s = el("div", "srcs");
  const seen = new Set();
  [...(srcs || [])]
    .sort((a, b) => String(a.book || "").localeCompare(String(b.book || "")) ||
      (Number(a.page) || 0) - (Number(b.page) || 0))
    .forEach(x => {
      const book = String(x.book || "").trim();
      if (!book) return;
      const hasPage = x.page !== undefined && x.page !== null && String(x.page).trim() !== "";
      const page = hasPage ? Math.max(1, parseInt(x.page, 10) || 1) : null;
      const label = page !== null ? `${book} · p.${page}` : book;
      if (seen.has(label)) return;
      seen.add(label);
      const c = el("a", "src");
      c.href = "/api/source?book=" + encodeURIComponent(book) +
        (page !== null ? "&page=" + page + "#page=" + page : "");
      c.target = "\_blank";
      c.rel = "noopener noreferrer";
      c.title = page !== null ? `Open ${book}, page ${page}` : `Open ${book} PDF`;
      c.style.cursor = "pointer";
      c.insertAdjacentHTML("afterbegin", svg("book"));
      c.appendChild(document.createTextNode(label));
      s.appendChild(c);
    });
  return s;
}
function actions(answer, q) {
  const a = el("div", "acts");
  const mk = (icon, label, fn) => { const b = el("button"); b.type = "button"; b.insertAdjacentHTML("afterbegin", svg(icon)); const t = el("span", "", label); b.appendChild(t); b.onclick = () => fn(t); a.appendChild(b); };
  mk("copy", "Copy", async t => { try { await navigator.clipboard.writeText(answer); t.textContent = "Copied"; setTimeout(() => t.textContent = "Copy", 1500); } catch { t.textContent = "Press Ctrl+C"; } });
  if (q) mk("help", "Quiz me", () => ask("Make 5 short quiz questions with answers on: " + q));
  return a;
}
function botMsg(text, srcs, q) {
  const r = el("div", "m-bot"); r.insertAdjacentHTML("afterbegin", `<div class="bot-av">${svg("sparkles")}</div>`);
  const col = el("div", "col"), body = el("div", "body"); renderMd(body, text); col.appendChild(body);
  if (srcs && srcs.length) col.appendChild(srcChips(srcs));
  else if (text && q !== undefined) col.appendChild(el("div", "muted small source-missing", "Textbook source not found for this answer."));
  if (text && q !== undefined) col.appendChild(actions(text, q));
  r.appendChild(col); box().appendChild(r); scrollEnd(); return {col, body};
}
/* ---------- streaming (server keeps generating; browser can reconnect after reload) ---------- */
async function runStream(chat, msg, lv, open) {
  let stopped = false, failed = false, errText = "", lastSave = 0;
  try {
    const res = await open(lv.ctrl.signal);
    if (!res.ok) throw new Error("HTTP " + res.status);
    for await (const m of ndjson(res)) {
      if (m.type === "job") { msg.job = m.data; save(); }
      else if (m.type === "sources") msg.sources = m.data;
      else if (m.type === "token") {
        if (lv.fresh) { msg.text = ""; lv.fresh = false; }   // resume: replay from start, old text stayed visible until now
        msg.text += m.data;
        if (!lv.raf) lv.raf = requestAnimationFrame(() => {
          lv.raf = 0;
          if (lv.ui) { lv.ui.body.classList.remove("dots"); renderMd(lv.ui.body, msg.text); }
          if (cur === chat) scrollEnd();
        });
        if (Date.now() - lastSave > 1000) { save(); lastSave = Date.now(); }   // survives refresh mid-answer
      }
      else if (m.type === "error") { failed = true; errText = m.data; }
    }
  } catch (e) { if (e.name === "AbortError") stopped = true; else { failed = true; errText = String(e); } }
 
  stopResponseTimer(lv); if (unloading) return;   // page is reloading: keep partial + job id so the answer resumes after reload
  msg.partial = false; delete msg.job;
  if (failed) { msg.error = true; msg.text = "Error: " + errText + (errText.includes("11434") || /connect/i.test(errText) ? "  — is Ollama running? (ollama serve)" : ""); }
  else if (stopped && !msg.text) { msg.text = "Stopped."; }
  const done = lv.ui; lives.delete(chat.id); if (cur === chat) setBusy(false);
  save(); renderHistory();
  if (cur === chat) {
    if (!done) { openChat(chat.id, false); return; }   // chat was opened after stream started: redraw it
    done.body.classList.remove("dots");
    if (msg.error) { done.body.classList.add("err"); done.body.textContent = msg.text; }
      else {
        renderMd(done.body, msg.text);
        if (msg.sources && msg.sources.length) done.col.appendChild(srcChips(msg.sources));
        else done.col.appendChild(el("div", "muted small source-missing", "Textbook source not found for this answer."));
        done.col.appendChild(actions(msg.text, msg.q));
      }
    $("q").focus(); scrollEnd();
  }
}
async function ask(override) {
  const text = (typeof override === "string" ? override : $("q").value).trim(); if (!text) return;
  if (!cur) newChat();
  const chat = cur;
  if (lives.has(chat.id)) return;
  const history = chat.messages.filter(m => !m.error && !m.partial && m.text).slice(-6).map(m => ({role: m.role === "user" ? "user" : "assistant", content: m.text}));
  if (!chat.messages.length) chat.title = text.slice(0, 48);
  chats = chats.filter(c => c !== chat); chats.unshift(chat);          // newest chat on top
  chat.messages.push({role: "user", text});
  const msg = {role: "bot", text: "", sources: [], q: text, partial: true};
  chat.messages.push(msg);
  pushUrl(chat.id); save(); renderHistory(); updateDocTitle();
  userMsg(text);
  if (typeof override !== "string") { $("q").value = ""; autosize(); }
  const ui = botMsg("", []); ui.body.classList.add("dots");
  const lv = {chat, msg, ui, ctrl: new AbortController()};
  attachResponseTimer(lv, ui.col, "Thinking");
  lives.set(chat.id, lv);
  stick = true; scrollEnd(true); setBusy(true);
  // book: null -> the server detects the book from the question itself (e.g. "Curiosity").
  // A fixed "Curiosity" here would filter every Class/Subject search (e.g. Class 8 Math) to Curiosity chapters.
  runStream(chat, msg, lv, signal => fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, signal,
    body: JSON.stringify({question: text, class: CLS, subject: SUB, book: null, history})}));
}
function autosize() { const q = $("q"); q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 180) + "px"; }
$("q").addEventListener("input", autosize);
$("send").onclick = () => { const lv = cur && lives.get(cur.id); lv ? lv.ctrl.abort() : ask(); };
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); } });
/* ---------- init (each call isolated so one failure does not break the rest) ---------- */
async function init() {
  setThemeIcon(); renderClasses(); renderSubjects(); updateCtx();
  setBusy(false);
  // after reload: if the server still has the job, resume the answer; otherwise mark it interrupted
  const resumed = [];
  chats.forEach(c => (c.messages || []).forEach(m => {
    if (!m.partial) return;
    if (m.job) {
      const lv = {chat: c, msg: m, ui: null, ctrl: new AbortController(), fresh: true};
       attachResponseTimer(lv, null, "Time since reconnect");
      lives.set(c.id, lv); resumed.push(lv);
    } else {
      m.partial = false;
      if (!m.text) { m.error = true; m.text = "Response was interrupted. Please ask again."; }
    }
  }));
  save(); route();   // restore the chat from the URL (/c/<id>)
  resumed.forEach(lv => runStream(lv.chat, lv.msg, lv, signal => fetch("/api/stream/" + lv.msg.job + "?from=0", {signal})));
  // health check runs in parallel: a slow/busy Ollama must not block loading classes & subjects
  fetch("/api/health").then(r => r.json()).then(h => {
    $("status").title = h.ollama ? "Ollama running: " + h.model : "Ollama is not running";
    $("status").classList.toggle("bad", !h.ollama);
  }).catch(() => { $("status").title = "Server not reachable"; $("status").classList.add("bad"); });
  loadSubjects();
}
// classes & subjects: retries by itself if the server is busy or still warming up
async function loadSubjects(attempt = 0) {
  try {
    const r = await fetch("/api/subjects"), d = await r.json();
    if (!r.ok) throw new Error(d.error || "HTTP " + r.status);
    SUBJECTS = d; store.set("subjects", JSON.stringify(d));
    // drop saved filters that no longer exist
    if (CLS && !SUBJECTS[CLS]) CLS = "";
    if (SUB && !subjectsFor(CLS).includes(SUB)) SUB = "";
    saveFilters(); renderClasses(); renderSubjects(); updateCtx();
    if (!Object.keys(SUBJECTS).length) {
      const p = document.querySelector(".empty p");
      if (p) { p.textContent = "No books indexed yet. Run: python -m src.ingestion.ingest"; p.style.color = "var(--bad)"; }
    }
  } catch (e) {
    const note = el("div", "muted small", "Could not load subjects" + (attempt < 10 ? ", retrying..." : ": " + e.message));
    note.style.padding = "6px 10px"; $("subjectList").appendChild(note);
    if (attempt < 10) setTimeout(() => loadSubjects(attempt + 1), 3000);
  }
}
// KaTeX scripts are deferred, so they load after this file: render math in already-drawn answers once ready
addEventListener("DOMContentLoaded", () => { document.querySelectorAll("#chat .body").forEach(mathify); scrollEnd(true); });
addEventListener("load", () => { document.querySelectorAll("#chat .body:not(.dots)").forEach(b => { if (!b.querySelector(".katex")) mathify(b); }); scrollEnd(true); });
init();
