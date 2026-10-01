import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const SAMPLE_QUESTIONS = [
  "Which region has the highest total revenue, and how much?",
  "Average units sold per category, as a bar chart.",
  "What share of revenue comes from the Enterprise segment?",
  "Which month had the most sales?",
];

function Header({ models, dataset }) {
  return (
    <header className="masthead">
      <div className="masthead__name">
        <span className="masthead__word">Ask Your Data</span>
        <span className="masthead__rule" />
        <span className="masthead__tag">a local analyst that writes its own code</span>
      </div>
      <div className="masthead__chips">
        {dataset && <span className="chip chip--data">{dataset}</span>}
        {models?.coder_model && <span className="chip">coder · {models.coder_model}</span>}
        {models?.reason_model && <span className="chip">reason · {models.reason_model}</span>}
      </div>
    </header>
  );
}

function StepRow({ step, index }) {
  const kind = step.ok === false ? "err" : step.title.startsWith("Check") ? "check" : "ok";
  return (
    <li className={`step step--${kind}`} style={{ animationDelay: `${index * 140}ms` }}>
      <span className="step__idx">{String(index + 1).padStart(2, "0")}</span>
      <div className="step__body">
        <span className="step__title">
          {step.title}
          {step.chart && <span className="step__badge">chart</span>}
        </span>
        <pre className="step__detail">{step.detail}</pre>
      </div>
    </li>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [models, setModels] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`).then((r) => r.json()).then(setModels).catch(() => {});
    fetch(`${API_BASE}/sample`).then((r) => r.json()).then((d) => setDataset(d.name)).catch(() => {});
    // "see a demo" mode: load a pre-run answer so visitors can see the output
    if (new URLSearchParams(window.location.search).has("demo")) {
      fetch("/demo.json")
        .then((r) => r.json())
        .then((d) => { setResult(d); setQuestion(d.question || ""); })
        .catch(() => {});
    }
  }, []);

  const ask = useCallback(async (q) => {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append("question", text);
      if (file) body.append("file", file);
      const res = await fetch(`${API_BASE}/ask`, { method: "POST", body });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Server said ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [question, file, loading]);

  const onFile = (f) => {
    setFile(f);
    setDataset(f ? f.name : "sales.csv");
  };

  return (
    <div className="desk">
      <Header models={models} dataset={dataset} />

      <main className="work">
        {/* left: the ask + the run log */}
        <section className="col col--left">
          <div className="ask-card">
            <label className="ask-card__label">Ask a question about the data</label>
            <textarea
              className="ask-card__input"
              rows={3}
              placeholder="e.g. which region sells the most, and by how much?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
              }}
            />
            <div className="ask-card__row">
              <div className="ask-card__source">
                <button className="link" onClick={() => fileRef.current?.click()}>
                  {file ? "change CSV" : "use your own CSV"}
                </button>
                {file && (
                  <button className="link link--muted" onClick={() => onFile(null)}>
                    reset to sample
                  </button>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv,text/csv"
                  hidden
                  onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
                />
              </div>
              <button className="ask-btn" onClick={() => ask()} disabled={loading || !question.trim()}>
                {loading ? "working…" : "Ask"}
                <span className="ask-btn__hint">⌘⏎</span>
              </button>
            </div>
            <div className="suggests">
              {SAMPLE_QUESTIONS.map((s) => (
                <button key={s} className="suggest" onClick={() => { setQuestion(s); ask(s); }}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="log">
            <div className="log__head">
              <span>RUN LOG</span>
              {result && <span className="log__meta">{result.attempts} attempt{result.attempts > 1 ? "s" : ""}</span>}
            </div>
            {loading && (
              <div className="log__working">
                <span className="pulse" /> planning, writing code, running it…
              </div>
            )}
            {!loading && !result && !error && (
              <p className="log__idle">The steps the agent takes show up here, one by one.</p>
            )}
            {error && <div className="banner banner--err">{error}</div>}
            {result && (
              <ul className="steps">
                {result.steps.map((s, i) => <StepRow key={i} step={s} index={i} />)}
              </ul>
            )}
          </div>
        </section>

        {/* right: the answer, the chart, the code */}
        <section className="col col--right">
          {!result && !loading && (
            <div className="placeholder">
              <p className="placeholder__lead">
                This is a working analyst, not a chatbot. It reads your data, writes
                pandas code, runs it in a locked sandbox, checks the result, and
                fixes its own mistakes before answering.
              </p>
              <p className="placeholder__sub">Ask something on the left to see it work.</p>
            </div>
          )}

          {result && (
            <>
              <div className={`answer ${result.success ? "" : "answer--weak"}`}>
                <span className="answer__eyebrow">{result.success ? "ANSWER" : "BEST EFFORT"}</span>
                <p className="answer__text">{result.answer}</p>
              </div>

              {result.chart_data_uri && (
                <figure className="chart">
                  <img src={result.chart_data_uri} alt="chart the agent drew" />
                </figure>
              )}

              {result.code && (
                <div className="code">
                  <div className="code__head">
                    <span>THE CODE IT RAN</span>
                    <button
                      className="link link--muted"
                      onClick={() => navigator.clipboard?.writeText(result.code)}
                    >
                      copy
                    </button>
                  </div>
                  <pre className="code__body"><code>{result.code}</code></pre>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
