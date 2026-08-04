import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CircleDot,
  Loader2,
  LockKeyhole,
  Radar,
  ScanSearch,
  Send,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { AnimatePresence, MotionConfig, motion, useReducedMotion } from "motion/react";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { getHealth, getMetadata, predictMessage } from "./api";
import type { Health, Metadata, Prediction, RiskLevel, ThreatLabel } from "./types";

const LABELS: ThreatLabel[] = ["ham", "phish", "spam"];
const EXAMPLES: Array<{ label: ThreatLabel; text: string }> = [
  { label: "ham", text: "Hi Jordan, can we move our project sync to 2 PM tomorrow? Thanks." },
  { label: "phish", text: "Urgent password reset required verify account." },
  { label: "spam", text: "Limited offer coupon savings buy now." },
];

const LABEL_COPY: Record<ThreatLabel, string> = {
  ham: "Legitimate",
  phish: "Phishing",
  spam: "Spam",
};

const RISK_COPY: Record<RiskLevel, string> = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
};

const API_DOWN: Health = {
  status: "error",
  model_loaded: false,
  model_path: null,
  detail: "API unavailable.",
};

function formatPercent(value?: number): string {
  if (value === undefined) {
    return "Unavailable";
  }
  return new Intl.NumberFormat("en", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

function formatCount(value?: number): string {
  if (value === undefined) {
    return "Unavailable";
  }
  return new Intl.NumberFormat("en").format(value);
}

function labelTone(label: ThreatLabel): string {
  const tones = {
    ham: "border-emerald-400/35 bg-emerald-400/10 text-emerald-100",
    phish: "border-rose-400/35 bg-rose-400/10 text-rose-100",
    spam: "border-amber-400/35 bg-amber-400/10 text-amber-100",
  };
  return tones[label];
}

function riskTone(risk: RiskLevel): string {
  const tones = {
    low: "text-emerald-200",
    medium: "text-amber-200",
    high: "text-rose-200",
  };
  return tones[risk];
}

function statusLabel(health: Health | null): string {
  if (health === null) {
    return "Checking model";
  }
  return health.model_loaded ? "Model ready" : "Model offline";
}

function statusClasses(health: Health | null): string {
  if (health === null) {
    return "border-cyan-300/25 bg-cyan-300/10 text-cyan-100";
  }
  if (health.model_loaded) {
    return "border-emerald-300/30 bg-emerald-300/10 text-emerald-100";
  }
  return "border-rose-300/30 bg-rose-300/10 text-rose-100";
}

function probabilityFor(prediction: Prediction, label: ThreatLabel): number | undefined {
  return prediction.probabilities?.[label];
}

function bestConfidence(prediction: Prediction | null): number | undefined {
  if (!prediction) {
    return undefined;
  }
  return probabilityFor(prediction, prediction.label);
}

function App() {
  const prefersReducedMotion = useReducedMotion();
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const textLimit = metadata?.max_text_chars ?? 5000;
  const trimmed = message.trim();
  const canSubmit = trimmed.length > 0 && trimmed.length <= textLimit && !isLoading;
  const testMetrics = metadata?.metrics?.metrics?.test;
  const confidence = bestConfidence(prediction);

  useEffect(() => {
    let cancelled = false;

    Promise.allSettled([getHealth(), getMetadata()]).then(([healthResult, metadataResult]) => {
      if (cancelled) {
        return;
      }

      setHealth(healthResult.status === "fulfilled" ? healthResult.value : API_DOWN);
      if (metadataResult.status === "fulfilled") {
        setMetadata(metadataResult.value);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const privacyCopy = useMemo(
    () => metadata?.privacy ?? "Messages are analyzed for the current request only and are not stored.",
    [metadata?.privacy],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictMessage(trimmed);
      setPrediction(result);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction failed.");
    } finally {
      setIsLoading(false);
    }
  }

  function chooseExample(text: string) {
    setMessage(text);
    setPrediction(null);
    setError(null);
  }

  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-screen overflow-x-hidden bg-zinc-950 text-zinc-50">
        <div className="fixed inset-0 -z-10 bg-[linear-gradient(135deg,rgba(24,24,27,0.96),rgba(9,9,11,1)_55%,rgba(28,25,23,0.94))]" />
        <div className="fixed inset-0 -z-10 bg-[linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.035)_1px,transparent_1px)] bg-[size:48px_48px] opacity-25" />

        <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-5 sm:px-6 lg:px-8">
          <header className="mb-5 flex flex-col gap-4 sm:mb-7 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid h-11 w-11 shrink-0 place-items-center rounded-md border border-cyan-300/25 bg-zinc-900/80 shadow-soft">
                <ShieldCheck className="h-5 w-5 text-cyan-200" aria-hidden="true" />
              </div>
              <div className="min-w-0">
                <h1 className="text-2xl font-semibold tracking-normal text-white sm:text-3xl">
                  ThreatLens
                </h1>
                <p className="mt-1 text-sm text-zinc-400">Spam and phishing message classifier</p>
              </div>
            </div>
            <div
              className={`inline-flex w-fit items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium ${statusClasses(
                health,
              )}`}
            >
              <CircleDot className="h-4 w-4" aria-hidden="true" />
              <span>{statusLabel(health)}</span>
            </div>
          </header>

          <div className="grid flex-1 gap-5 lg:grid-cols-[minmax(0,1fr)_24rem] xl:grid-cols-[minmax(0,1fr)_27rem]">
            <section className="rounded-lg border border-white/10 bg-zinc-900/78 p-4 shadow-soft backdrop-blur-xl sm:p-6">
              <form className="flex h-full flex-col gap-5" onSubmit={handleSubmit}>
                <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                  <div>
                    <label htmlFor="message" className="text-sm font-medium text-zinc-100">
                      Message
                    </label>
                    <p className="mt-1 text-sm text-zinc-500">
                      {formatCount(trimmed.length)} / {formatCount(textLimit)} characters
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLES.map((example) => (
                      <button
                        type="button"
                        key={example.label}
                        onClick={() => chooseExample(example.text)}
                        className={`rounded-md border px-3 py-2 text-sm font-medium transition hover:-translate-y-0.5 hover:border-white/25 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-200 ${labelTone(
                          example.label,
                        )}`}
                      >
                        {example.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="relative min-h-[18rem] flex-1 overflow-hidden rounded-lg border border-white/10 bg-black/30">
                  <textarea
                    id="message"
                    value={message}
                    maxLength={textLimit}
                    onChange={(event) => setMessage(event.target.value)}
                    placeholder="Urgent password reset required verify account."
                    className="h-full min-h-[18rem] w-full resize-none bg-transparent p-4 text-base leading-7 text-zinc-50 outline-none transition placeholder:text-zinc-600 focus:ring-2 focus:ring-cyan-300/50 sm:p-5"
                  />
                  {isLoading && (
                    <div className="pointer-events-none absolute inset-0 overflow-hidden">
                      <motion.div
                        className="h-full w-1/3 bg-gradient-to-r from-transparent via-cyan-200/10 to-transparent"
                        initial={{ x: "-120%" }}
                        animate={{ x: "320%" }}
                        transition={{
                          duration: prefersReducedMotion ? 0 : 1.2,
                          repeat: prefersReducedMotion ? 0 : Infinity,
                          ease: "easeInOut",
                        }}
                      />
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-sm text-zinc-500">{privacyCopy}</p>
                  <button
                    type="submit"
                    disabled={!canSubmit}
                    className="inline-flex h-12 items-center justify-center gap-2 rounded-md bg-cyan-200 px-5 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-100 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400 sm:min-w-36"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    ) : (
                      <Send className="h-4 w-4" aria-hidden="true" />
                    )}
                    Analyze
                  </button>
                </div>
              </form>
            </section>

            <aside className="flex flex-col gap-5">
              <section className="min-h-[20rem] rounded-lg border border-white/10 bg-zinc-900/78 p-4 shadow-soft backdrop-blur-xl sm:p-5">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <ScanSearch className="h-5 w-5 text-cyan-200" aria-hidden="true" />
                    <h2 className="text-lg font-semibold text-white">Analysis</h2>
                  </div>
                  {prediction && (
                    <span
                      data-testid="prediction-label"
                      className={`rounded-md border px-2.5 py-1 text-xs font-semibold ${labelTone(
                        prediction.label,
                      )}`}
                    >
                      {LABEL_COPY[prediction.label]}
                    </span>
                  )}
                </div>

                <AnimatePresence mode="wait">
                  {isLoading ? (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: prefersReducedMotion ? 0 : -8 }}
                      className="space-y-5"
                    >
                      <div className="flex items-center gap-3 rounded-md border border-cyan-300/15 bg-cyan-300/5 p-4">
                        <Radar className="h-5 w-5 animate-pulse text-cyan-200" aria-hidden="true" />
                        <span className="text-sm text-cyan-100">Scanning message structure</span>
                      </div>
                      {LABELS.map((label) => (
                        <div key={label} className="space-y-2">
                          <div className="flex justify-between text-sm text-zinc-500">
                            <span>{label}</span>
                            <span>...</span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-white/10">
                            <motion.div
                              className="h-full w-1/2 rounded-full bg-cyan-200/50"
                              animate={{ x: prefersReducedMotion ? 0 : ["-60%", "180%"] }}
                              transition={{
                                duration: 1.1,
                                repeat: prefersReducedMotion ? 0 : Infinity,
                                ease: "easeInOut",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  ) : prediction ? (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, y: prefersReducedMotion ? 0 : 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: prefersReducedMotion ? 0 : -8 }}
                      className="space-y-5"
                    >
                      <div>
                        <p className={`text-sm font-semibold ${riskTone(prediction.risk_level)}`}>
                          <span data-testid="prediction-risk">
                            {RISK_COPY[prediction.risk_level]}
                          </span>
                        </p>
                        <p className="mt-2 text-4xl font-semibold tracking-normal text-white">
                          {formatPercent(confidence)}
                        </p>
                        <p className="mt-1 text-sm text-zinc-500">Model confidence</p>
                      </div>

                      <div className="space-y-3">
                        {LABELS.map((label) => {
                          const value = probabilityFor(prediction, label) ?? 0;
                          return (
                            <div key={label} className="space-y-2">
                              <div className="flex items-center justify-between gap-3 text-sm">
                                <span className="font-medium text-zinc-200">{label}</span>
                                <span className="text-zinc-500">{formatPercent(value)}</span>
                              </div>
                              <div className="h-2 overflow-hidden rounded-full bg-white/10">
                                <motion.div
                                  className={`h-full rounded-full ${
                                    label === "ham"
                                      ? "bg-emerald-300"
                                      : label === "phish"
                                        ? "bg-rose-300"
                                        : "bg-amber-300"
                                  }`}
                                  initial={{ width: prefersReducedMotion ? `${value * 100}%` : 0 }}
                                  animate={{ width: `${value * 100}%` }}
                                  transition={{ duration: 0.7, ease: "easeOut" }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      <div className="space-y-3 text-sm leading-6 text-zinc-300">
                        <p>{prediction.explanation}</p>
                        <p className="text-zinc-100">{prediction.suggested_action}</p>
                      </div>
                    </motion.div>
                  ) : error ? (
                    <motion.div
                      key="error"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="rounded-md border border-rose-300/20 bg-rose-300/10 p-4 text-sm leading-6 text-rose-100"
                    >
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
                        <p>{error}</p>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex min-h-56 flex-col justify-center gap-4 text-zinc-500"
                    >
                      <ShieldAlert className="h-9 w-9 text-zinc-600" aria-hidden="true" />
                      <p className="max-w-sm text-sm leading-6">Ready for model output.</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </section>

              <section className="rounded-lg border border-white/10 bg-zinc-900/78 p-4 shadow-soft backdrop-blur-xl sm:p-5">
                <div className="mb-4 flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-emerald-200" aria-hidden="true" />
                  <h2 className="text-lg font-semibold text-white">Model Results</h2>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="Test accuracy" value={formatPercent(testMetrics?.accuracy)} />
                  <Metric label="Macro F1" value={formatPercent(testMetrics?.f1_macro)} />
                  <Metric label="Rows modeled" value={formatCount(metadata?.metrics?.data_summary?.modeling_rows)} />
                  <Metric label="Artifact" value={metadata?.model.artifact ?? "Unavailable"} />
                </div>
                <div className="mt-4 flex items-start gap-3 rounded-md border border-white/10 bg-black/20 p-3 text-sm leading-6 text-zinc-400">
                  <LockKeyhole className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500" aria-hidden="true" />
                  <p>{privacyCopy}</p>
                </div>
              </section>

              <section className="rounded-lg border border-white/10 bg-zinc-900/78 p-4 shadow-soft backdrop-blur-xl sm:p-5">
                <div className="flex items-start gap-3">
                  {health?.model_loaded ? (
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-200" aria-hidden="true" />
                  ) : (
                    <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-200" aria-hidden="true" />
                  )}
                  <p className="text-sm leading-6 text-zinc-400">
                    {health?.detail ?? "Connecting to the API."}
                  </p>
                </div>
              </section>
            </aside>
          </div>
        </main>
      </div>
    </MotionConfig>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-black/20 p-3">
      <p className="text-xs font-medium uppercase tracking-normal text-zinc-500">{label}</p>
      <p className="mt-2 truncate text-base font-semibold text-zinc-100" title={value}>
        {value}
      </p>
    </div>
  );
}

export default App;
