"use client";

import { useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Home() {
  const [query, setQuery] = useState("");
  /** null = retrieval not finished yet; array = ready to render (maybe empty) */
  const [documents, setDocuments] = useState(null);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [error, setError] = useState("");
  const [expandedSourceSections, setExpandedSourceSections] = useState({});
  const abortRef = useRef(null);

  const SOURCE_TEXT_PREVIEW_CHARS = 220;

  const canAsk = useMemo(() => query.trim().length > 0 && !loading, [query, loading]);
  const retrievalPending = loading && documents === null;
  const showAnswerPanel = !retrievalPending && (documents !== null || streamingAnswer.length > 0);
  /** Stagger outer source cards slightly; section sub-boxes fade in one-by-one (global order) */
  const sourceCardStaggerMs = 120;
  const sourceSectionStaggerMs = 480;
  const sectionGlobalOffset = useMemo(() => {
    if (!documents?.length) return [];
    const prefix = [];
    let acc = 0;
    for (const doc of documents) {
      prefix.push(acc);
      acc += Array.isArray(doc.sections) ? doc.sections.length : 0;
    }
    return prefix;
  }, [documents]);

  const imageEntries = useMemo(() => {
    if (!documents?.length) return [];

    const flattened = [];
    for (const doc of documents) {
      if (!Array.isArray(doc.images)) continue;
      for (const image of doc.images) {
        if (!image?.source_url) continue;
        flattened.push({
          ...image,
          document_id: doc.document_id,
          document_title: doc.title || "Untitled Source",
          document_url: doc.url || "",
        });
      }
    }

    return flattened;
  }, [documents]);
  const hasImageMatches = imageEntries.length > 0;

  /** Strip legacy placeholder tokens if the model mis-cites; normalize spacing */
  const answerMarkdown = useMemo(
    () => streamingAnswer.replace(/\s*\[CITATION\]\s*/gi, " ").replace(/  +/g, " "),
    [streamingAnswer]
  );
  const waitingForFirstToken =
    loading && documents !== null && streamingAnswer.length === 0;

  const toggleSourceSection = (key) => {
    setExpandedSourceSections((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  async function handleAsk() {
    if (!query.trim()) return;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setLoading(true);
    setDocuments(null);
    setStreamingAnswer("");
    setError("");

    try {
      const res = await fetch("http://127.0.0.1:8000/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: ac.signal,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `Backend returned status ${res.status}`);
      }

      if (!res.body) {
        throw new Error("No response body from stream.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const consumePayload = (payload) => {
        if (payload.type === "documents") {
          setDocuments(Array.isArray(payload.documents) ? payload.documents : []);
        } else if (payload.type === "token" && payload.text) {
          setStreamingAnswer((prev) => prev + payload.text);
        } else if (payload.type === "error") {
          throw new Error(payload.message || "Stream error");
        }
      };

      const parseSseBlock = (text) => {
        for (const line of text.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            consumePayload(JSON.parse(line.slice(6)));
          } catch (e) {
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          parseSseBlock(block);
        }
      }

      if (buffer.trim()) {
        parseSseBlock(buffer);
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      console.error("Failed to fetch backend response:", err);
      setDocuments(null);
      setStreamingAnswer("");
      setError(err.message || "Could not connect to backend. Make sure FastAPI is running on port 8000.");
    } finally {
      if (abortRef.current === ac) abortRef.current = null;
      setLoading(false);
    }
  }

  const theme = {
    page: darkMode
      ? "bg-[#09090B] text-zinc-100"
      : "bg-[#F4F4F5] text-zinc-900",
    header: darkMode
      ? "border-white/[0.06] bg-[#09090B]/75"
      : "border-black/[0.06] bg-[#F4F4F5]/80",
    card: darkMode
      ? "border-white/[0.06] bg-[#18181B] shadow-[0_1px_0_rgba(255,255,255,0.04)_inset,0_12px_40px_rgba(0,0,0,0.45)]"
      : "border-black/[0.05] bg-white shadow-[0_1px_0_rgba(0,0,0,0.03)_inset,0_8px_32px_rgba(0,0,0,0.06)]",
    cardInner: darkMode
      ? "border-white/[0.06] bg-[#141416]"
      : "border-black/[0.05] bg-[#FAFAFA]",
    muted: darkMode ? "text-zinc-400" : "text-zinc-500",
    softText: darkMode ? "text-zinc-300" : "text-zinc-600",
    faintText: darkMode ? "text-zinc-500" : "text-zinc-400",
    inputOuter: darkMode
      ? "from-violet-500/25 via-[#D6B15E]/35 to-sky-500/25"
      : "from-violet-400/35 via-[#D6B15E]/45 to-sky-400/40",
    inputInner: darkMode
      ? "border-0 bg-[#0C0C0E] text-zinc-100 placeholder:text-zinc-500"
      : "border-0 bg-white text-zinc-900 placeholder:text-zinc-400",
    secondaryButton: darkMode
      ? "border-white/[0.08] bg-white/[0.04] text-zinc-200 hover:bg-white/[0.07]"
      : "border-black/[0.08] bg-white text-zinc-800 shadow-sm hover:bg-zinc-50",
    skeleton: darkMode ? "bg-zinc-700/50" : "bg-zinc-200/80",
    errorBox: darkMode
      ? "border-red-500/25 bg-red-950/40 text-red-200"
      : "border-red-200 bg-red-50 text-red-800",
    themePillTrack: darkMode ? "bg-zinc-800/90" : "bg-zinc-200/90",
    themePillActive: darkMode
      ? "bg-zinc-950 text-zinc-100 shadow-[0_1px_3px_rgba(0,0,0,0.5)]"
      : "bg-white text-zinc-900 shadow-[0_1px_3px_rgba(0,0,0,0.08)]",
    themePillIdle: darkMode ? "text-zinc-500 hover:text-zinc-300" : "text-zinc-500 hover:text-zinc-700",
  };

  return (
    <div className={`min-h-screen transition-colors duration-500 ${theme.page}`}>
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        {darkMode ? (
          <>
            <div className="absolute left-1/2 top-[-120px] h-[320px] w-[720px] -translate-x-1/2 rounded-full bg-[#D6B15E]/[0.07] blur-3xl" />
            <div className="absolute bottom-[-140px] right-[-100px] h-[280px] w-[300px] rounded-full bg-sky-500/[0.04] blur-3xl" />
            <div className="absolute left-[-80px] top-1/3 h-[200px] w-[200px] rounded-full bg-violet-500/[0.05] blur-3xl" />
          </>
        ) : (
          <>
            <div className="absolute left-1/2 top-[-100px] h-[280px] w-[640px] -translate-x-1/2 rounded-full bg-sky-200/35 blur-3xl" />
            <div className="absolute right-[-60px] top-1/4 h-[220px] w-[220px] rounded-full bg-violet-200/30 blur-3xl" />
            <div className="absolute bottom-[-80px] left-1/4 h-[200px] w-[360px] rounded-full bg-[#D6B15E]/12 blur-3xl" />
          </>
        )}
      </div>

      <header
        className={`sticky top-0 z-20 border-b backdrop-blur-2xl backdrop-saturate-150 transition-colors duration-500 ${theme.header}`}
      >
        <div className="mx-auto flex w-full max-w-[1760px] items-center justify-between px-5 py-3.5 sm:py-4">
          <div className="flex items-center gap-3">
            <div
              className={`grid h-10 w-10 place-items-center rounded-2xl border shadow-sm sm:h-11 sm:w-11 ${
                darkMode
                  ? "border-white/[0.08] bg-gradient-to-br from-white/[0.08] to-white/[0.02]"
                  : "border-black/[0.06] bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)]"
              }`}
            >
              <span className="text-sm font-bold tracking-tight text-[#D6B15E] sm:text-base">BC</span>
            </div>

            <div>
              <h1 className="text-xl font-semibold tracking-tight sm:text-[26px]">BoilerCheck</h1>
              <p className={`text-xs sm:text-sm ${theme.muted}`}>Purdue policy, cited sources</p>
            </div>
          </div>

          <div
            className={`flex rounded-full p-1 transition-colors duration-300 ${theme.themePillTrack}`}
            role="group"
            aria-label="Theme"
          >
            <button
              type="button"
              onClick={() => setDarkMode(false)}
              className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition sm:px-4 sm:text-sm ${
                !darkMode ? theme.themePillActive : theme.themePillIdle
              }`}
            >
              Light
            </button>
            <button
              type="button"
              onClick={() => setDarkMode(true)}
              className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition sm:px-4 sm:text-sm ${
                darkMode ? theme.themePillActive : theme.themePillIdle
              }`}
            >
              Dark
            </button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto w-full max-w-[1760px] px-4 py-8 sm:px-6">
        <section
          className={`rounded-[1.35rem] border p-5 transition-colors duration-500 sm:rounded-3xl sm:p-6 ${theme.card}`}
        >
          <div className="mb-4">
            <label className={`text-sm font-medium ${theme.muted}`}>Ask about policy</label>
          </div>

          <div className="flex flex-col gap-4">
            <div
              className={`rounded-2xl bg-gradient-to-r p-[1.25px] shadow-[0_0_0_1px_rgba(0,0,0,0.03)] transition-shadow focus-within:shadow-[0_0_0_1px_rgba(214,177,94,0.25),0_0_24px_rgba(214,177,94,0.12)] dark:focus-within:shadow-[0_0_0_1px_rgba(214,177,94,0.2),0_0_28px_rgba(214,177,94,0.08)] ${theme.inputOuter}`}
            >
              <input
                className={`w-full rounded-[0.9rem] px-4 py-3.5 text-[15px] outline-none ring-0 transition sm:px-5 sm:py-4 ${theme.inputInner}`}
                placeholder='e.g. “What appliances are allowed in my dorm?”'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleAsk();
                }}
              />
            </div>

            <div className="flex flex-wrap gap-2 sm:gap-3">
              <button
                type="button"
                onClick={handleAsk}
                disabled={!canAsk}
                className={`rounded-2xl px-6 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                  darkMode
                    ? "bg-[#D6B15E] text-zinc-950 hover:brightness-110"
                    : "bg-zinc-900 text-white shadow-sm hover:bg-zinc-800"
                }`}
              >
                {loading ? (documents === null ? "Searching…" : "Answering…") : "Ask"}
              </button>

              <button
                type="button"
                onClick={() => {
                  abortRef.current?.abort();
                  setQuery("");
                  setDocuments(null);
                  setStreamingAnswer("");
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
          <div
            className={`mt-6 rounded-2xl border px-4 py-3 text-sm transition-colors duration-300 ${theme.errorBox}`}
          >
            {error}
          </div>
        )}

        <section
          className={`mt-8 grid gap-5 ${
            hasImageMatches ? "xl:grid-cols-[1.8fr_1.08fr_1.08fr]" : "lg:grid-cols-[1.8fr_1.1fr]"
          } lg:gap-7 xl:gap-7`}
        >
          <div
            className={`rounded-[1.35rem] border p-5 transition-colors duration-500 sm:rounded-3xl sm:p-6 ${theme.card}`}
          >
            <div className="mb-5">
              <p className={`text-[11px] font-semibold uppercase tracking-[0.2em] ${theme.muted}`}>
                Answer
              </p>
            </div>

            {!showAnswerPanel && !loading && !error && (
              <div
                className={`rounded-2xl border p-6 transition-colors duration-300 sm:p-8 ${theme.cardInner}`}
              >
                <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl sm:leading-tight">
                  What do you want to know?
                </h2>
                <p className={`mt-3 max-w-xl text-[15px] leading-relaxed ${theme.softText}`}>
                  Ask in plain language. The model answers from retrieved Purdue policy only, with
                  sources you can open on the right.
                </p>
              </div>
            )}

            {retrievalPending && (
              <div className="space-y-4">
                <div className={`h-5 w-2/5 animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-full animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-[92%] animate-pulse rounded ${theme.skeleton}`} />
                <div className={`h-4 w-[85%] animate-pulse rounded ${theme.skeleton}`} />
              </div>
            )}

            {showAnswerPanel && (
              <div
                className={`rounded-2xl border p-6 transition-colors duration-300 sm:p-7 ${theme.cardInner}`}
              >
                <h2 className="text-lg font-semibold tracking-tight sm:text-xl">Response</h2>
                {waitingForFirstToken && (
                  <p
                    className={`mt-3 flex items-center gap-2 text-sm ${theme.muted}`}
                    aria-live="polite"
                  >
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#D6B15E] opacity-40" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-[#D6B15E]" />
                    </span>
                    Drafting your answer from the sources…
                  </p>
                )}
                <div className={`mt-4 text-[16px] leading-8 ${theme.softText}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                      ul: ({ children }) => (
                        <ul className="mb-0 list-disc space-y-2.5 pl-5 marker:text-[#D6B15E]/80 [&_li>p]:mb-0 [&_li>p]:inline">
                          {children}
                        </ul>
                      ),
                      ol: ({ children }) => (
                        <ol className="mb-0 list-decimal space-y-2.5 pl-5 marker:font-medium marker:text-[#D6B15E]/90 [&_li>p]:mb-0 [&_li>p]:inline">
                          {children}
                        </ol>
                      ),
                      li: ({ children }) => <li className="leading-7">{children}</li>,
                      strong: ({ children }) => (
                        <strong className="font-semibold text-current">{children}</strong>
                      ),
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-medium text-[#D6B15E] underline decoration-[#D6B15E]/50 underline-offset-[3px] transition hover:decoration-[#D6B15E]"
                        >
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {answerMarkdown}
                  </ReactMarkdown>
                  {loading && documents !== null && !waitingForFirstToken && (
                    <span
                      className={`ml-0.5 inline-block h-[1.1em] w-0.5 translate-y-px animate-pulse rounded-sm ${
                        darkMode ? "bg-zinc-400" : "bg-zinc-500"
                      }`}
                      aria-hidden
                    />
                  )}
                </div>
              </div>
            )}
          </div>

          <aside
            className={`rounded-[1.35rem] border p-5 transition-colors duration-500 sm:rounded-3xl sm:p-6 ${theme.card}`}
          >
            <div className="mb-5 flex items-baseline justify-between gap-2">
              <p className={`text-[11px] font-semibold uppercase tracking-[0.2em] ${theme.muted}`}>
                Sources
              </p>
              {documents !== null && documents.length > 0 && (
                <span className={`text-xs font-medium tabular-nums ${theme.faintText}`}>
                  {documents.length}
                </span>
              )}
            </div>

            {documents === null && !loading && !error && (
              <div
                className={`rounded-2xl border p-5 transition-colors duration-300 ${theme.cardInner}`}
              >
                <p className={`text-[15px] leading-7 ${theme.softText}`}>
                  Source cards will appear here after you ask a question.
                </p>
              </div>
            )}

            {retrievalPending && (
              <div className="space-y-4">
                <div className={`h-14 animate-pulse rounded-2xl ${theme.skeleton}`} />
                <div className={`h-14 animate-pulse rounded-2xl ${theme.skeleton}`} />
                <div className={`h-14 animate-pulse rounded-2xl ${theme.skeleton}`} />
              </div>
            )}

            {documents !== null && (
              <div className="space-y-4">
                {documents.length === 0 ? (
                  <div
                    className={`rounded-2xl border p-5 transition-colors duration-300 ${theme.cardInner}`}
                  >
                    <p className={`text-[15px] leading-7 ${theme.softText}`}>
                      No source documents were returned by the backend.
                    </p>
                  </div>
                ) : (
                  documents.map((doc, index) => (
                    <a
                      key={doc.document_id || index}
                      href={doc.url || "#"}
                      target="_blank"
                      rel="noreferrer"
                      style={{ animationDelay: `${index * sourceCardStaggerMs}ms` }}
                      className={`source-card-enter group block rounded-2xl border p-5 shadow-sm transition-colors duration-300 ${theme.cardInner} ${
                        darkMode
                          ? "hover:border-[#D6B15E]/30 hover:shadow-[0_0_0_1px_rgba(214,177,94,0.12)]"
                          : "hover:border-[#D6B15E]/45 hover:shadow-md"
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
                              style={{
                                animationDelay: `${
                                  (sectionGlobalOffset[index] + secIndex) * sourceSectionStaggerMs
                                }ms`,
                              }}
                              className={`source-section-enter rounded-xl border p-3 ${
                                darkMode
                                  ? "border-white/10 bg-white/[0.03]"
                                  : "border-black/10 bg-white/70"
                              }`}
                            >
                              <div className="text-sm font-semibold">
                                {section.section_title || "Untitled Section"}
                              </div>
                              {(() => {
                                const rawText = section.text || "No text provided.";
                                const sectionKey = `${doc.document_id || index}-${secIndex}`;
                                const expanded = !!expandedSourceSections[sectionKey];
                                const shouldTruncate = rawText.length > SOURCE_TEXT_PREVIEW_CHARS;
                                const previewText = shouldTruncate
                                  ? rawText.slice(0, SOURCE_TEXT_PREVIEW_CHARS).trimEnd()
                                  : rawText;

                                return (
                                  <>
                                    <div className={`mt-1 text-sm leading-6 ${theme.softText}`}>
                                      {expanded || !shouldTruncate
                                        ? rawText
                                        : `${previewText}...`}
                                    </div>
                                    {shouldTruncate && (
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.preventDefault();
                                          toggleSourceSection(sectionKey);
                                        }}
                                        className={`mt-2 text-xs font-medium ${
                                          darkMode
                                            ? "text-[#D6B15E] hover:text-[#E3C983]"
                                            : "text-zinc-700 hover:text-zinc-900"
                                        }`}
                                      >
                                        {expanded ? "Show less" : "Show more"}
                                      </button>
                                    )}
                                  </>
                                );
                              })()}
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

          {hasImageMatches && (
            <aside
              className={`rounded-[1.35rem] border p-5 transition-colors duration-500 sm:rounded-3xl sm:p-6 ${theme.card}`}
            >
              <div className="mb-5 flex items-baseline justify-between gap-2">
                <p className={`text-[11px] font-semibold uppercase tracking-[0.2em] ${theme.muted}`}>
                  Images
                </p>
                <span className={`text-xs font-medium tabular-nums ${theme.faintText}`}>
                  {imageEntries.length}
                </span>
              </div>

              <div className="space-y-4">
                {imageEntries.map((image, index) => (
                  <a
                    key={`${image.source_url}-${image.md5 || index}`}
                    href={image.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className={`group block overflow-hidden rounded-2xl border p-3 shadow-sm transition-colors duration-300 ${theme.cardInner} ${
                      darkMode
                        ? "hover:border-[#D6B15E]/30 hover:shadow-[0_0_0_1px_rgba(214,177,94,0.12)]"
                        : "hover:border-[#D6B15E]/45 hover:shadow-md"
                    }`}
                  >
                    <img
                      src={image.source_url}
                      alt={image.description || image.filename || "Source image"}
                      loading="lazy"
                      className="h-40 w-full rounded-xl object-cover"
                    />

                    <div className="mt-3 space-y-1">
                      <p className="text-sm font-semibold leading-6">{image.document_title}</p>
                      {image.description && (
                        <p className={`line-clamp-3 text-xs leading-5 ${theme.softText}`}>
                          {image.description}
                        </p>
                      )}
                      <div className={`text-[11px] ${theme.faintText}`}>
                        {image.filename || "image"}
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            </aside>
          )}
        </section>
      </main>
    </div>
  );
}