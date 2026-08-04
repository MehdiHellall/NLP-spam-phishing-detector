import {
  AlertTriangle,
  BarChart3,
  CircleDot,
  Loader2,
  LockKeyhole,
  ScanSearch,
  Send,
  ShieldCheck,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { getMetadata, getReadiness, predictMessage } from "./api";
import type { Metadata, Prediction, Readiness, RiskLevel, ThreatLabel } from "./types";

const LABELS: ThreatLabel[] = ["ham", "phish", "spam"];

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

const API_DOWN: Readiness = {
  status: "error",
  model_loaded: false,
  model_path: null,
  detail: "The API is unavailable. Check the service and try again.",
};

function formatPercent(value?: number | null): string {
  if (value == null) {
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

function metadataString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function labelTone(label: ThreatLabel): string {
  return {
    ham: "border-emerald-400/35 bg-emerald-400/10 text-emerald-100",
    phish: "border-rose-400/35 bg-rose-400/10 text-rose-100",
    spam: "border-amber-400/35 bg-amber-400/10 text-amber-100",
  }[label];
}

function riskTone(risk: RiskLevel): string {
  return {
    low: "text-emerald-200",
    medium: "text-amber-200",
    high: "text-rose-200",
  }[risk];
}

function statusLabel(readiness: Readiness | null): string {
  if (readiness === null) {
    return "Checking model";
  }
  return readiness.model_loaded ? "Model ready" : "Model offline";
}

function statusClasses(readiness: Readiness | null): string {
  if (readiness === null) {
    return "border-zinc-600 bg-zinc-800 text-zinc-200";
  }
  if (readiness.model_loaded) {
    return "border-emerald-300/30 bg-emerald-300/10 text-emerald-100";
  }
  return "border-rose-300/30 bg-rose-300/10 text-rose-100";
}

function App() {
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const textLimit = metadata?.max_text_chars ?? 5000;
  const trimmedMessage = message.trim();
  const canSubmit =
    trimmedMessage.length > 0 && trimmedMessage.length <= textLimit && !isLoading;
  const modelOutput = prediction?.model_outputs.tfidf_logreg;
  const testMetrics = metadata?.metrics?.metrics?.test;
  const artifact =
    prediction?.artifact_metadata.artifact ?? metadata?.model.artifact ?? "Unavailable";
  const modelName =
    prediction?.artifact_metadata.model_name ??
    metadataString(metadata?.model.metadata.model_name) ??
    "TF-IDF + Logistic Regression";

  useEffect(() => {
    let cancelled = false;

    Promise.allSettled([getReadiness(), getMetadata()]).then(
      ([readinessResult, metadataResult]) => {
        if (cancelled) {
          return;
        }

        setReadiness(
          readinessResult.status === "fulfilled" ? readinessResult.value : API_DOWN,
        );
        if (metadataResult.status === "fulfilled") {
          setMetadata(metadataResult.value);
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, []);

  const privacyCopy = useMemo(
    () =>
      metadata?.privacy ??
      "Messages are analyzed for the current request only and are not stored.",
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
      const result = await predictMessage(trimmedMessage);
      setPrediction(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The message could not be analyzed. Try again.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-5 sm:px-6 lg:px-8">
        <header className="mb-5 flex flex-col gap-4 sm:mb-7 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-11 w-11 shrink-0 place-items-center rounded-md border border-cyan-300/25 bg-zinc-900">
              <ShieldCheck className="h-5 w-5 text-cyan-200" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h1 className="text-2xl font-semibold text-white sm:text-3xl">ThreatLens</h1>
              <p className="mt-1 text-sm text-zinc-400">Message threat classifier</p>
            </div>
          </div>
          <div
            role="status"
            className={
              "inline-flex w-fit items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium " +
              statusClasses(readiness)
            }
          >
            <CircleDot className="h-4 w-4" aria-hidden="true" />
            <span>{statusLabel(readiness)}</span>
          </div>
        </header>

        <div className="grid flex-1 gap-5 lg:grid-cols-[minmax(0,1fr)_25rem]">
          <section className="rounded-lg border border-white/10 bg-zinc-900 p-4 sm:p-6">
            <form className="flex h-full flex-col gap-5" onSubmit={handleSubmit}>
              <div>
                <label htmlFor="message" className="text-sm font-medium text-zinc-100">
                  Message
                </label>
                <p className="mt-1 text-sm text-zinc-500">
                  {formatCount(message.length)} / {formatCount(textLimit)} characters
                </p>
              </div>

              <textarea
                id="message"
                value={message}
                maxLength={textLimit}
                onChange={(event) => {
                  setMessage(event.target.value);
                  setError(null);
                }}
                placeholder="Paste the message you want to analyze."
                className="min-h-[20rem] flex-1 resize-none rounded-lg border border-white/10 bg-black/30 p-4 text-base leading-7 text-zinc-50 outline-none transition placeholder:text-zinc-600 focus:border-cyan-300/40 focus:ring-2 focus:ring-cyan-300/40 sm:p-5"
              />

              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex max-w-2xl items-start gap-2 text-sm leading-6 text-zinc-500">
                  <LockKeyhole
                    className="mt-1 h-4 w-4 shrink-0 text-zinc-500"
                    aria-hidden="true"
                  />
                  <p>{privacyCopy}</p>
                </div>
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
            <section
              className="min-h-[20rem] rounded-lg border border-white/10 bg-zinc-900 p-4 sm:p-5"
              aria-labelledby="analysis-heading"
            >
              <div className="mb-5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <ScanSearch className="h-5 w-5 text-cyan-200" aria-hidden="true" />
                  <h2 id="analysis-heading" className="text-lg font-semibold text-white">
                    Analysis
                  </h2>
                </div>
                {prediction && (
                  <span
                    data-testid="prediction-label"
                    className={
                      "rounded-md border px-2.5 py-1 text-xs font-semibold " +
                      labelTone(prediction.final_label)
                    }
                  >
                    {LABEL_COPY[prediction.final_label]}
                  </span>
                )}
              </div>

              {isLoading ? (
                <div className="flex items-center gap-3 rounded-md border border-cyan-300/15 bg-cyan-300/5 p-4">
                  <Loader2
                    className="h-5 w-5 animate-spin text-cyan-200"
                    aria-hidden="true"
                  />
                  <span className="text-sm text-cyan-100">Analyzing message…</span>
                </div>
              ) : prediction && modelOutput ? (
                <div className="space-y-5">
                  <div>
                    <p
                      data-testid="prediction-risk"
                      className={"text-sm font-semibold " + riskTone(prediction.final_risk_level)}
                    >
                      {RISK_COPY[prediction.final_risk_level]}
                    </p>
                    <p
                      data-testid="prediction-confidence"
                      className="mt-2 text-4xl font-semibold text-white"
                    >
                      {formatPercent(prediction.final_confidence)}
                    </p>
                    <p className="mt-1 text-sm text-zinc-500">Model confidence</p>
                  </div>

                  <div className="space-y-3" aria-label="Class probabilities">
                    {LABELS.map((label) => {
                      const probability = modelOutput.probabilities?.[label];
                      return (
                        <div key={label} className="space-y-2">
                          <div className="flex items-center justify-between gap-3 text-sm">
                            <span className="font-medium text-zinc-200">
                              {LABEL_COPY[label]}
                            </span>
                            <span
                              data-testid={"probability-" + label}
                              className="text-zinc-400"
                            >
                              {formatPercent(probability)}
                            </span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-white/10">
                            <div
                              className={
                                "h-full rounded-full " +
                                (label === "ham"
                                  ? "bg-emerald-300"
                                  : label === "phish"
                                    ? "bg-rose-300"
                                    : "bg-amber-300")
                              }
                              style={{
                                width:
                                  probability === undefined
                                    ? "0%"
                                    : String(probability * 100) + "%",
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="border-t border-white/10 pt-4 text-sm leading-6">
                    <h3 className="font-medium text-zinc-100">Explanation</h3>
                    <p
                      data-testid="prediction-explanation"
                      className="mt-1 text-zinc-400"
                    >
                      {prediction.explanation}
                    </p>
                  </div>

                  <div className="text-sm leading-6">
                    <h3 className="font-medium text-zinc-100">Suggested action</h3>
                    <p data-testid="suggested-action" className="mt-1 text-zinc-300">
                      {prediction.suggested_action}
                    </p>
                  </div>
                </div>
              ) : error ? (
                <div
                  role="alert"
                  className="rounded-md border border-rose-300/20 bg-rose-300/10 p-4 text-sm leading-6 text-rose-100"
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle
                      className="mt-0.5 h-5 w-5 shrink-0"
                      aria-hidden="true"
                    />
                    <p>{error}</p>
                  </div>
                </div>
              ) : (
                <p className="text-sm leading-6 text-zinc-500">
                  Enter a message and select Analyze to view the classifier result.
                </p>
              )}
            </section>

            <section
              className="rounded-lg border border-white/10 bg-zinc-900 p-4 sm:p-5"
              aria-labelledby="model-heading"
            >
              <div className="mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-emerald-200" aria-hidden="true" />
                <h2 id="model-heading" className="text-lg font-semibold text-white">
                  Model
                </h2>
              </div>
              <dl className="grid grid-cols-2 gap-3">
                <Metric label="Model" value={modelName} />
                <Metric
                  label="Status"
                  value={metadata?.model.loaded ? "Ready" : "Unavailable"}
                />
                <Metric
                  label="Artifact"
                  value={artifact}
                  testId="model-artifact"
                />
                {testMetrics?.accuracy !== undefined ? (
                  <Metric
                    label="Test accuracy"
                    value={formatPercent(testMetrics.accuracy)}
                  />
                ) : null}
                {testMetrics?.f1_macro !== undefined ? (
                  <Metric label="Macro F1" value={formatPercent(testMetrics.f1_macro)} />
                ) : null}
              </dl>
              {readiness && !readiness.model_loaded ? (
                <p
                  role="alert"
                  className="mt-4 rounded-md border border-rose-300/20 bg-rose-300/10 p-3 text-sm leading-6 text-rose-100"
                >
                  {readiness.detail}
                </p>
              ) : null}
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  testId,
}: {
  label: string;
  value: string;
  testId?: string;
}) {
  return (
    <div className="min-w-0 rounded-md border border-white/10 bg-black/20 p-3">
      <dt className="text-xs font-medium uppercase text-zinc-500">{label}</dt>
      <dd
        data-testid={testId}
        className="mt-2 truncate text-sm font-semibold text-zinc-100"
        title={value}
      >
        {value}
      </dd>
    </div>
  );
}

export default App;
