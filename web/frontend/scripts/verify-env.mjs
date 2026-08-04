const apiBaseUrl = process.env.VITE_API_BASE_URL;

if (!apiBaseUrl) {
  console.error("VITE_API_BASE_URL must be set before building ThreatLens.");
  process.exit(1);
}

try {
  const parsedUrl = new URL(apiBaseUrl);
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new Error("unsupported protocol");
  }
} catch {
  console.error("VITE_API_BASE_URL must be a valid http(s) URL.");
  process.exit(1);
}
