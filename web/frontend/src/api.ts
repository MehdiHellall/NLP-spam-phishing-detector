import type { Health, Metadata, Prediction } from "./types";

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL;

if (!rawApiBaseUrl) {
  throw new Error("VITE_API_BASE_URL must be set to the ThreatLens API origin.");
}

const API_BASE_URL = rawApiBaseUrl.replace(/\/$/, "");

type ApiErrorBody = {
  detail?: string;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  const body = (await response.json().catch(() => ({}))) as ApiErrorBody;
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return body as T;
}

export function getHealth(): Promise<Health> {
  return fetch(`${API_BASE_URL}/health`)
    .then(async (response) => {
      const body = (await response.json().catch(() => ({}))) as Health | ApiErrorBody;
      if ("model_loaded" in body && "status" in body) {
        return body;
      }
      if (!response.ok) {
        throw new Error(body.detail ?? `Request failed with ${response.status}`);
      }
      return body as Health;
    });
}

export function getMetadata(): Promise<Metadata> {
  return requestJson<Metadata>("/metadata");
}

export function predictMessage(text: string): Promise<Prediction> {
  return requestJson<Prediction>("/predict", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}
