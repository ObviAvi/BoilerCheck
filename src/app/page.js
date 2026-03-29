"use client";

import { useMemo, useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [error, setError] = useState("");

  const canAsk = useMemo(() => query.trim().length > 0 && !loading, [query, loading]);

  async function handleAsk() {
    if (!query.trim()) return;

    setLoading(true);
    setData(null);
    setError("");

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `Backend returned status ${res.status}`);
      }

      const json = await res.json();

      setData({
        answer: json?.answer ?? "No answer returned.",
        documents: Array.isArray(json?.documents) ? json.documents : [],
      });
    } catch (err) {
      console.error("Failed to fetch backend response:", err);
      setError(err.message || "Could not connect to backend. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  const theme = {
    page: darkMode ? "bg-[#0B0B0D] text-white" : "bg-[#F8F5EE] text-[#171717]",
    header: darkMode
      ? "border-white/10 bg-[#0B0B0D]/80"
      : "border-black/10 bg-[#F8F5EE]/85",
    card: darkMode
      ? "border-white/10 bg-white/[0.04]"
      : "border-black/10 bg-white/90",
    cardInner: darkMode
      ? "border-white/10 bg-[#101115]"
      : "border-black/10 bg-[#FCFAF6]",
    muted: darkMode ? "text-white/60" : "text-black/55",
    softText: darkMode ? "text-white/75" : "text-black/75",
    faintText: darkMode ? "text-white/40" : "text-black/40",
    input: darkMode
      ? "border-white/10 bg-[#090A0F] text-white placeholder:text-white/35"
      : "border-black/10 bg-white text-black placeholder:text-black/35",
    secondaryButton: darkMode
      ? "border-white/10 bg-white/[0.04] text-white hover:bg-white/[0.08]"
      : "border-black/10 bg-white text-black hover:bg-black/[0.04]",
    skeleton: darkMode ? "bg-white/10" : "bg-black/10",
    errorBox: darkMode
      ? "border-red-400/30 bg-red-500/10 text-red-200"
      : "border-red-300 bg-red-50 text-red-700",
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${theme.page}`}>
      {darkMode && (
        <div className="pointer-events-none fixed inset-0 overflow-hidden">
          <div className="absolute left-1/2 top-[-100px] h-[260px] w-[700px] -translate-x-1/2 rounded-full bg-[#D6B15E]/10 blur-3xl" />
          <div className="absolute bottom-[-120px] right-[-80px] h-[260px] w-[260px] rounded-full bg-[#D6B15E]/[0.06] blur-3xl" />
        </div>
      )}

      <header
        className={`sticky top-0 z-20 border-b backdrop-blur-xl transition-colors duration-300 ${theme.header}`}
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-3">
            <div
              className={`grid h-11 w-11 place-items-center rounded-2xl border shadow-sm ${
                darkMode ? "border-white/10 bg-white/[0.04]" : "border-black/10 bg-white"
              }`}
            >
              <span className="text-base font-bold tracking-wide text-[#D6B15E]">BC</span>
            </div>

            <div>
              <h1 className="text-[26px] font-semibold tracking-tight">BoilerCheck</h1>
              <p className={`text-sm ${theme.muted}`}>Answers with sources</p>
            </div>
          </div>

          <button
            onClick={() => setDarkMode((prev) => !prev)}
            className={`rounded-xl border px-4 py-2 text-sm font-medium transition ${theme.secondaryButton}`}
          >
            {darkMode ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
      </header>

      <main className="relative mx-auto max-w-7xl px-5 py-8">
        <section
          className={`rounded-3xl border p-5 shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-colors duration-300 ${theme.card}`}
        >
          <div className="mb-3">
            <label className={`text-sm font-medium ${theme.muted}`}>
              Ask a Purdue policy
            </label>
          </div>

          <div className="flex flex-col gap-3">
            <input
              className={`w-full rounded-2xl border px-5 py-4 text-[15px] outline-none transition focus:border-[#D6B15E]/60 focus:ring-4 focus:ring-[#D6B15E]/10 ${theme.input}`}
              placeholder='Example: "What appliances are allowed in my dorm?"'
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAsk();
              }}
            />

            <div className="flex gap-3">
              <button
                onClick={handleAsk}
                disabled={!canAsk}
                className="rounded-2xl bg-[#D6B15E] px-6 py-3 text-sm font-semibold text-black transition hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Searching..." : "Ask"}
              </button>

              <button
                onClick={() => {
                  setQuery("");
                  setData(null);
                  setError("");
                }}
                className={`rounded-2xl border px-6 py-3 text-sm font-medium transition ${theme.secondaryButton}`}
              >
                Clear
              </button>
            </div>
          </div>
        </section>

        {error && (
          <div className={`mt-6 rounded-2xl border px-4 py-3 text-sm ${theme.errorBox}`}>
            {error}
          </div>
        )}

        <section className="mt-8 grid gap-6 lg:grid-cols-[1.65fr_1fr]">
          <div
            className={`rounded-3xl border p-6 shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-colors duration-300 ${theme.card}`}
          >
            <div className="mb-5">
              <p className={`text-xs font-semibold tracking-[0.22em] ${theme.muted}`}>
                ANSWER
              </p>
            </div>

            {!data && !loading && !error && (
              <div
                className={`rounded-2xl border p-6 transition-colors duration-300 ${theme.cardInner}`}
              >
                <h2 className="text-[30px] font-semibold tracking-tight">Type a question above.</h2>
                <p className={`mt-3 max-w-2xl text-[15px] leading-7 ${theme.softText}`}>
                  Your answer will appear here on the left, and the supporting source cards will
                  appear on the right.
                </p>
              </div>
            )}

            {loading && (
              <div className="space-y-4">
                <div className={`h-5 w-2/5 animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-full animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-[92%] animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-[85%] animate-pulse rounded ${theme.skeleton}`} />
              </div>
            )}

            {data && (
              <div
                className={`rounded-2xl border p-6 transition-colors duration-300 ${theme.cardInner}`}
              >
                <h2 className="text-[26px] font-semibold tracking-tight">Response</h2>
                <div className={`mt-5 text-[16px] leading-8 ${theme.softText}`}>
                  {data.answer}
                </div>
              </div>
            )}
          </div>

          <aside
            className={`rounded-3xl border p-6 shadow-[0_8px_30px_rgba(0,0,0,0.06)] transition-colors duration-300 ${theme.card}`}
          >
            <div className="mb-5">
              <p className={`text-xs font-semibold tracking-[0.22em] ${theme.muted}`}>
                SOURCES
              </p>
            </div>

            {!data && !loading && !error && (
              <div
                className={`rounded-2xl border p-5 transition-colors duration-300 ${theme.cardInner}`}
              >
                <p className={`text-[15px] leading-7 ${theme.softText}`}>
                  Source cards will appear here after you ask a question.
                </p>
              </div>
            )}

            {data && (
              <div className="space-y-4">
                {data.documents.length === 0 ? (
                  <div
                    className={`rounded-2xl border p-5 transition-colors duration-300 ${theme.cardInner}`}
                  >
                    <p className={`text-[15px] leading-7 ${theme.softText}`}>
                      No source documents were returned by the backend.
                    </p>
                  </div>
                ) : (
                  data.documents.map((doc, index) => (
                    <a
                      key={doc.document_id || index}
                      href={doc.url || "#"}
                      target="_blank"
                      rel="noreferrer"
                      className={`group block rounded-2xl border p-5 transition ${theme.cardInner} ${
                        darkMode
                          ? "hover:border-[#D6B15E]/35"
                          : "hover:border-[#D6B15E]/50"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="text-[18px] font-semibold leading-7">
                            {doc.title || "Untitled Source"}
                          </h3>
                          <p className={`mt-1 text-xs ${theme.faintText}`}>
                            {(doc.domain || "unknown")} • effective {doc.effective_date || "N/A"}
                          </p>
                        </div>

                        <span className="mt-1 text-xs font-medium text-[#D6B15E] opacity-0 transition group-hover:opacity-100">
                          Open ↗
                        </span>
                      </div>

                      <div className="mt-4 space-y-3">
                        {Array.isArray(doc.sections) &&
                          doc.sections.map((section, secIndex) => (
                            <div
                              key={secIndex}
                              className={`rounded-xl border p-3 ${
                                darkMode
                                  ? "border-white/10 bg-white/[0.03]"
                                  : "border-black/10 bg-white/70"
                              }`}
                            >
                              <div className="text-sm font-semibold">
                                {section.section_title || "Untitled Section"}
                              </div>
                              <div className={`mt-1 text-sm leading-6 ${theme.softText}`}>
                                {section.text || "No text provided."}
                              </div>
                            </div>
                          ))}
                      </div>

                      <div className={`mt-4 text-xs ${theme.faintText}`}>
                        {doc.url ? new URL(doc.url).hostname : "No URL"}
                      </div>
                    </a>
                  ))
                )}
              </div>
            )}
          </aside>
        </section>
      </main>
    </div>
  );
}