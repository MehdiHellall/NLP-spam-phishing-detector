const apiBaseUrl = process.env.VITE_API_BASE_URL;

if (!apiBaseUrl) {
  console.error("VITE_API_BASE_URL must be set before building ThreatLens.");
  process.exit(1);
}

function isLoopbackHost(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

try {
  const parsedUrl = new URL(apiBaseUrl);
  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    throw new Error("unsupported protocol");
  }
  if (parsedUrl.protocol === "http:" && !isLoopbackHost(parsedUrl.hostname)) {
    throw new Error("non-loopback http");
  }
  if (parsedUrl.pathname !== "/" || parsedUrl.search || parsedUrl.hash) {
    throw new Error("not an origin");
  }
} catch {
  console.error(
    "VITE_API_BASE_URL must be an http(s) origin, and non-loopback production origins must use https.",
  );
  process.exit(1);
}
