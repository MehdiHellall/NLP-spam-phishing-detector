import type { Metadata, Prediction, Readiness } from "./types";

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL;

if (!rawApiBaseUrl) {
  throw new Error("VITE_API_BASE_URL must be set to the ThreatLens API origin.");
}

const API_BASE_URL = rawApiBaseUrl.replace(/\/$/, "");

type ApiErrorBody = {
  detail?: string;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = init?.body
    ? {
        "Content-Type": "application/json",
      }
    : init?.headers;

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });

  const body = (await response.json().catch(() => ({}))) as ApiErrorBody;
  if (!response.ok) {
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return body as T;
}

export function getReadiness(): Promise<Readiness> {
  return fetch(`${API_BASE_URL}/v1/ready`)
    .then(async (response) => {
      const body = (await response.json().catch(() => ({}))) as Readiness | ApiErrorBody;
      if ("model_loaded" in body && "status" in body) {
        return body;
      }
      if (!response.ok) {
        throw new Error(body.detail ?? `Request failed with ${response.status}`);
      }
      return body as Readiness;
    });
}

export function getMetadata(): Promise<Metadata> {
  return requestJson<Metadata>("/v1/metadata");
}

export function predictMessage(text: string): Promise<Prediction> {
  return requestJson<Prediction>("/v1/predict", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}
