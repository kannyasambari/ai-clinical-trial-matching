import { useState, useRef, useEffect } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:8000";

// ─── Utility ──────────────────────────────────────────────────────────────────
function formatTime(iso) {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function parseMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, '<code class="inline-code">$1</code>')
    .replace(/\n- /g, "\n• ")
    .replace(/\n/g, "<br/>");
}

// ─── API layer ────────────────────────────────────────────────────────────────
async function apiLogin(username, password) {
  const form = new URLSearchParams({ username, password });

  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));

    let message = "Login failed";

    if (Array.isArray(err.detail)) {
      message = err.detail.map((e) => e.msg).join(", ");
    } else if (typeof err.detail === "string") {
      message = err.detail;
    } else if (err.detail?.msg) {
      message = err.detail.msg;
    }

    throw new Error(message);
  }

  return res.json();
}

async function apiRegister(username, password, role) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, role }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(
      typeof err.detail === "string"
        ? err.detail
        : JSON.stringify(err.detail) || "Registration failed",
    );
  }
  return res.json();
}

async function* apiChatStream(token, message, sessionId) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId, stream: true }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {}
      }
    }
  }
}

// ─── Auth screen ──────────────────────────────────────────────────────────────
function AuthScreen({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("patient");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "login") {
        const data = await apiLogin(username, password);
        onAuth(data.access_token, username, data.role);
      } else {
        await apiRegister(username, password, role);
        const data = await apiLogin(username, password);
        onAuth(data.access_token, username, role);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-logo">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <path
                d="M14 2C7.373 2 2 7.373 2 14s5.373 12 12 12 12-5.373 12-12S20.627 2 14 2z"
                fill="url(#logoGrad)"
              />
              <path
                d="M14 7v7l5 3"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <defs>
                <linearGradient
                  id="logoGrad"
                  x1="2"
                  y1="2"
                  x2="26"
                  y2="26"
                  gradientUnits="userSpaceOnUse"
                >
                  <stop stopColor="#4FC3F7" />
                  <stop offset="1" stopColor="#0277BD" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div>
            <div className="logo-name">ARIA</div>
            <div className="logo-subtitle">Clinical Trial Assistant</div>
          </div>
        </div>

        <div className="auth-tabs">
          <button
            className={mode === "login" ? "tab active" : "tab"}
            onClick={() => setMode("login")}
          >
            Sign In
          </button>
          <button
            className={mode === "register" ? "tab active" : "tab"}
            onClick={() => setMode("register")}
          >
            Create Account
          </button>
        </div>

        <form onSubmit={submit} className="auth-form">
          <div className="field-group">
            <label>Username</label>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoFocus
              required
            />
          </div>
          <div className="field-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              required
            />
          </div>
          {mode === "register" && (
            <div className="field-group">
              <label>Role</label>
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="patient">Patient</option>
                <option value="doctor">Doctor</option>
                <option value="admin">Admin</option>
              </select>
            </div>
          )}
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? (
              <span className="spinner-small" />
            ) : mode === "login" ? (
              "Sign In"
            ) : (
              "Create Account"
            )}
          </button>
        </form>
        <p className="auth-disclaimer">
          This tool assists in finding clinical trials. Always consult your
          physician before making healthcare decisions.
        </p>
      </div>
    </div>
  );
}

// ─── Profile panel ────────────────────────────────────────────────────────────
function ProfilePanel({ profile, searchCount }) {
  const hasProfile = profile && (profile.age || profile.conditions?.length);
  return (
    <div className="profile-panel">
      <div className="panel-header">
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
        Patient Profile
      </div>
      {!hasProfile ? (
        <div className="profile-empty">
          Profile will be populated as you describe your medical situation.
        </div>
      ) : (
        <div className="profile-data">
          {profile.age && (
            <div className="profile-row">
              <span className="profile-label">Age</span>
              <span className="profile-value">{profile.age}</span>
            </div>
          )}
          {profile.sex && (
            <div className="profile-row">
              <span className="profile-label">Sex</span>
              <span className="profile-value">{profile.sex}</span>
            </div>
          )}
          {profile.conditions?.length > 0 && (
            <div className="profile-row column">
              <span className="profile-label">Conditions</span>
              <div className="tag-list">
                {profile.conditions.map((c) => (
                  <span key={c} className="tag tag-condition">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
          {profile.symptoms?.length > 0 && (
            <div className="profile-row column">
              <span className="profile-label">Symptoms</span>
              <div className="tag-list">
                {profile.symptoms.map((s) => (
                  <span key={s} className="tag tag-symptom">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}
          {profile.medications?.length > 0 && (
            <div className="profile-row column">
              <span className="profile-label">Medications</span>
              <div className="tag-list">
                {profile.medications.map((m) => (
                  <span key={m} className="tag tag-med">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {searchCount > 0 && (
        <div className="search-count">
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
          {searchCount} search{searchCount !== 1 ? "es" : ""} performed
        </div>
      )}
      <div className="panel-section-header">How to use Aria</div>
      <ul className="usage-tips">
        <li>Describe your condition naturally</li>
        <li>Mention your age and symptoms</li>
        <li>Answer Aria's follow-up questions for better matches</li>
        <li>Add details in follow-up messages</li>
      </ul>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="avatar aria-avatar">A</div>
      <div className="bubble typing-bubble">
        <span className="dot" style={{ animationDelay: "0ms" }} />
        <span className="dot" style={{ animationDelay: "160ms" }} />
        <span className="dot" style={{ animationDelay: "320ms" }} />
      </div>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      {!isUser && <div className="avatar aria-avatar">A</div>}
      <div className="bubble-wrapper">
        <div
          className={`bubble ${isUser ? "user-bubble" : "aria-bubble"}`}
          dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
        />
        <div className="msg-time">{formatTime(msg.timestamp)}</div>
      </div>
      {isUser && (
        <div className="avatar user-avatar">
          {msg.username?.[0]?.toUpperCase() || "U"}
        </div>
      )}
    </div>
  );
}

// ── v4 NEW: Gap-question banner shown above the input when Aria is waiting ────
function GapBanner({ onDismiss }) {
  return (
    <div className="gap-banner">
      <div className="gap-banner-icon">💬</div>
      <div className="gap-banner-text">
        <strong>Aria needs a bit more info</strong>
        <span>
          Answer her questions above to get personalised trial recommendations
        </span>
      </div>
      <button
        className="gap-banner-dismiss"
        onClick={onDismiss}
        title="Dismiss"
      >
        ×
      </button>
    </div>
  );
}

function WelcomeScreen() {
  const suggestions = [
    "I'm a 45-year-old man recently diagnosed with type 2 diabetes and high blood pressure.",
    "28-year-old woman with PCOS, irregular periods, and suspected insulin resistance.",
    "I have stage 2 breast cancer. I'm 52, female, currently on tamoxifen.",
    "My doctor thinks I have Crohn's disease. I'm 34, male, not on medication yet.",
  ];
  return (
    <div className="welcome">
      <div className="welcome-icon">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
          <circle cx="20" cy="20" r="20" fill="url(#wGrad)" />
          <path
            d="M20 10v10l7 4"
            stroke="white"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
          <defs>
            <linearGradient
              id="wGrad"
              x1="0"
              y1="0"
              x2="40"
              y2="40"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="#4FC3F7" />
              <stop offset="1" stopColor="#0277BD" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <h2 className="welcome-title">Hi, I'm Aria</h2>
      <p className="welcome-subtitle">
        Describe your medical situation and I'll search real clinical trials for
        you. No forms — just tell me what's going on.
      </p>
      <div className="suggestions">
        {suggestions.map((s, i) => (
          <div key={i} className="suggestion-chip" data-suggestion={s}>
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            {s.length > 70 ? s.slice(0, 70) + "…" : s}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [token, setToken] = useState(
    () => localStorage.getItem("aria_token") || "",
  );
  const [username, setUsername] = useState(
    () => localStorage.getItem("aria_user") || "",
  );
  const [userRole, setUserRole] = useState(
    () => localStorage.getItem("aria_role") || "",
  );
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [profile, setProfile] = useState(null);
  const [searchCount, setSearchCount] = useState(0);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // ── v4 new state ────────────────────────────────────────────────────────────
  // true while Aria has asked gap questions and is waiting for the user to answer
  const [awaitingGapAnswers, setAwaitingGapAnswers] = useState(false);
  const [showGapBanner, setShowGapBanner] = useState(false);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, loading]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  // Suggestion chip click handler
  useEffect(() => {
    const handler = (e) => {
      const chip = e.target.closest("[data-suggestion]");
      if (chip) {
        setInput(chip.dataset.suggestion);
        textareaRef.current?.focus();
      }
    };
    document.addEventListener("click", handler);
    return () => document.removeEventListener("click", handler);
  }, []);

  function handleAuth(tok, uname, role) {
    setToken(tok);
    setUsername(uname);
    setUserRole(role);
    localStorage.setItem("aria_token", tok);
    localStorage.setItem("aria_user", uname);
    localStorage.setItem("aria_role", role);
  }

  function logout() {
    setToken("");
    setUsername("");
    setUserRole("");
    setMessages([]);
    setSessionId(null);
    setProfile(null);
    setAwaitingGapAnswers(false);
    setShowGapBanner(false);
    localStorage.removeItem("aria_token");
    localStorage.removeItem("aria_user");
    localStorage.removeItem("aria_role");
  }

  function resetChat() {
    setMessages([]);
    setSessionId(null);
    setProfile(null);
    setSearchCount(0);
    setInput("");
    setAwaitingGapAnswers(false);
    setShowGapBanner(false);
  }

  async function sendMessage(text) {
    if (!text.trim() || loading) return;
    setInput("");
    // Clear gap state when user sends their answer
    setAwaitingGapAnswers(false);
    setShowGapBanner(false);

    const userMsg = {
      id: Date.now(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
      username,
    };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setStreamingContent("");

    try {
      let fullContent = "";
      let finalMeta = null;

      for await (const event of apiChatStream(token, text, sessionId)) {
        if (event.type === "token") {
          fullContent += event.content;
          setStreamingContent(fullContent);
        } else if (event.type === "done") {
          finalMeta = event;
          if (event.session_id) setSessionId(event.session_id);
          if (event.patient_profile) setProfile(event.patient_profile);
          if (typeof event.trials_found === "number") {
            setSearchCount((prev) => prev + (event.search_performed ? 1 : 0));
          }

          // ── v4: handle gap-asking state ──────────────────────────────────
          if (event.awaiting_gap_answers) {
            setAwaitingGapAnswers(true);
            setShowGapBanner(true);
          } else {
            setAwaitingGapAnswers(false);
            setShowGapBanner(false);
          }
        }
      }

      // Non-streaming done event
      if (!fullContent && finalMeta?.reply) {
        fullContent = finalMeta.reply;
        if (finalMeta.session_id) setSessionId(finalMeta.session_id);
        if (finalMeta.patient_profile) setProfile(finalMeta.patient_profile);
        if (finalMeta.awaiting_gap_answers) {
          setAwaitingGapAnswers(true);
          setShowGapBanner(true);
        }
      }

      if (fullContent) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: "assistant",
            content: fullContent,
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          content: `**Connection error:** ${err.message}\n\nPlease check that the backend is running at \`${API_BASE}\`.`,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
      setStreamingContent("");
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  // Dynamic input placeholder depending on conversation state
  const inputPlaceholder = awaitingGapAnswers
    ? "Answer Aria's questions above to unlock your trial results…"
    : "Describe your condition, age, symptoms… (Shift+Enter for new line)";

  if (!token) return <AuthScreen onAuth={handleAuth} />;

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-logo">
          <svg width="22" height="22" viewBox="0 0 28 28" fill="none">
            <path
              d="M14 2C7.373 2 2 7.373 2 14s5.373 12 12 12 12-5.373 12-12S20.627 2 14 2z"
              fill="url(#sGrad)"
            />
            <path
              d="M14 7v7l5 3"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <defs>
              <linearGradient
                id="sGrad"
                x1="2"
                y1="2"
                x2="26"
                y2="26"
                gradientUnits="userSpaceOnUse"
              >
                <stop stopColor="#4FC3F7" />
                <stop offset="1" stopColor="#0277BD" />
              </linearGradient>
            </defs>
          </svg>
          <span className="sidebar-title">ARIA</span>
        </div>

        <button className="new-chat-btn" onClick={resetChat}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
          New Conversation
        </button>

        <ProfilePanel profile={profile} searchCount={searchCount} />

        <div className="sidebar-footer">
          <div className="user-info">
            <div className="user-dot" />
            <div>
              <div className="user-name">{username}</div>
              <div className="user-role">{userRole}</div>
            </div>
          </div>
          <button className="logout-btn" onClick={logout} title="Sign out">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />
            </svg>
          </button>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        {/* Topbar */}
        <header className="topbar">
          <button
            className="toggle-sidebar"
            onClick={() => setSidebarOpen((o) => !o)}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
          </button>
          <div className="topbar-title">
            {/* v4: pulsing dot turns amber while waiting for gap answers */}
            <span
              className={`aria-dot ${awaitingGapAnswers ? "waiting" : ""}`}
            />
            Aria · Clinical Trial Assistant
          </div>
          <div className="topbar-badge">Beta</div>
        </header>

        {/* Messages */}
        <div className="messages-area">
          {messages.length === 0 && !loading && <WelcomeScreen />}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}

          {loading && !streamingContent && <TypingIndicator />}
          {streamingContent && (
            <div className="message assistant">
              <div className="avatar aria-avatar">A</div>
              <div className="bubble-wrapper">
                <div
                  className="bubble aria-bubble streaming"
                  dangerouslySetInnerHTML={{
                    __html: parseMarkdown(streamingContent),
                  }}
                />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* ── v4: gap banner appears above input when Aria is waiting ── */}
        {showGapBanner && (
          <GapBanner onDismiss={() => setShowGapBanner(false)} />
        )}

        {/* Input */}
        <div className="input-area">
          <div
            className={`input-wrapper ${awaitingGapAnswers ? "gap-mode" : ""}`}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={inputPlaceholder}
              rows={1}
              disabled={loading}
            />
            <button
              className={`send-btn ${input.trim() && !loading ? "active" : ""}`}
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || loading}
            >
              {loading ? (
                <span className="spinner-small" />
              ) : (
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                >
                  <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
                </svg>
              )}
            </button>
          </div>
          <div className="input-hint">
            {awaitingGapAnswers
              ? "Your answers help Aria find more accurate trial matches 💙"
              : "Aria uses real ClinicalTrials.gov data · Not medical advice · Always consult your physician"}
          </div>
        </div>
      </main>
    </div>
  );
}
