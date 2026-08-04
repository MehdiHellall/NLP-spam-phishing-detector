export type ThreatLabel = "ham" | "phish" | "spam";
export type RiskLevel = "low" | "medium" | "high";

export type ModelPrediction = {
  label: ThreatLabel;
  confidence: number | null;
  probabilities: Record<ThreatLabel, number> | null;
};

export type Prediction = {
  final_label: ThreatLabel;
  final_risk_level: RiskLevel;
  final_confidence: number | null;
  model_outputs: {
    tfidf_logreg: ModelPrediction;
  };
  explanation: string;
  suggested_action: string;
  artifact_metadata: {
    artifact: string | null;
    model_name: string | null;
    metrics_file: string | null;
  };
};

export type Readiness = {
  status: "ok" | "error";
  model_loaded: boolean;
  model_path: string | null;
  detail: string;
};

export type MetricSplit = {
  accuracy?: number;
  f1_macro?: number;
  precision_macro?: number;
  recall_macro?: number;
  per_label?: Record<ThreatLabel, { f1: number; precision: number; recall: number; support: number }>;
};

export type Metadata = {
  app_name: string;
  labels: ThreatLabel[];
  max_text_chars: number;
  model: {
    loaded: boolean;
    artifact: string | null;
    metadata: Record<string, unknown>;
    status: string | null;
  };
  metrics: {
    model_name?: string;
    data_summary?: {
      modeling_rows?: number;
    };
    metrics?: {
      test?: MetricSplit;
      validation?: MetricSplit;
    };
  } | null;
  privacy: string;
};
