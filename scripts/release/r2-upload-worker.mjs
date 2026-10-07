// Source of upload.infernux-engine.com (existing Worker: infernux-r2-upload-040).
// Explicit release destinations; keep published versions available. Secrets are bindings.
const ALLOWED_KEYS = new Set([
  "plugins/infernux.mcp/0.1.12/infernux.mcp.inxpkg",
  "plugins/infernux.mcp/0.1.5/infernux.mcp.inxpkg",
  "plugins/infernux.platform-windows/0.2.2/infernux.platform-windows.inxpkg",
  "plugins/infernux.platform-linux/0.2.2/infernux.platform-linux.inxpkg",
  "plugins/infernux.platform-android/0.2.4/infernux.platform-android.inxpkg",
  "plugins/infernux.platform-web/0.2.2/infernux.platform-web.inxpkg",
  "plugins/infernux.mcp/0.1.2/infernux.mcp.inxpkg",
  "plugins/infernux.platform-windows/0.2.1/infernux.platform-windows.inxpkg",
  "plugins/infernux.platform-linux/0.2.1/infernux.platform-linux.inxpkg",
  "plugins/infernux.platform-android/0.2.3/infernux.platform-android.inxpkg",
  "plugins/infernux.platform-web/0.2.1/infernux.platform-web.inxpkg",
  "android-support/0.1.0/infernux-android-support-0.1.0-linux-x64.inxkit",
  "android-support/0.1.0/infernux-android-support-0.1.0-windows-x64.inxkit",
  "plugins/infernux.mcp/0.1.1/infernux.mcp.inxpkg",
  "plugins/infernux.platform-android/0.2.2/infernux.platform-android.inxpkg",
  "plugins/infernux.platform-linux/0.2.0/infernux.platform-linux.inxpkg",
  "plugins/infernux.platform-web/0.2.0/infernux.platform-web.inxpkg",
  "plugins/infernux.platform-windows/0.2.0/infernux.platform-windows.inxpkg",
  "plugins/official-registry.json",
  "hub/0.4.0/InfernuxHub-0.4.0-linux-x64-full.zip",
  "hub/0.4.0/InfernuxHub-0.4.0-windows-x64-full.zip",
  "hub/0.4.0/InfernuxHub-linux-x64-manifest.json",
  "hub/0.4.0/InfernuxHub-windows-x64-manifest.json",
  "hub/0.4.0/InfernuxHubInstaller-0.4.0-linux-x64",
  "hub/0.4.0/InfernuxHubInstaller-0.4.0-windows-x64.exe",
  "hub/0.4.0/build-2/InfernuxHub-0.4.0-linux-x64-full.zip",
  "hub/0.4.0/build-2/InfernuxHub-0.4.0-windows-x64-full.zip",
  "hub/0.4.0/build-2/InfernuxHub-linux-x64-manifest.json",
  "hub/0.4.0/build-2/InfernuxHub-windows-x64-manifest.json",
  "hub/0.4.0/build-2/InfernuxHubInstaller-0.4.0-linux-x64",
  "hub/0.4.0/build-2/InfernuxHubInstaller-0.4.0-windows-x64.exe"
]);
const RELEASE_SOURCES = new Map([
  ["plugins/infernux.mcp/0.1.12/infernux.mcp.inxpkg", "https://github.com/ChenlizheMe/infernux_mcp/releases/download/v0.1.12/infernux.mcp.inxpkg"],
  ["plugins/infernux.mcp/0.1.5/infernux.mcp.inxpkg", "https://github.com/ChenlizheMe/infernux_mcp/releases/download/v0.1.5/infernux.mcp.inxpkg"],
  ["plugins/infernux.platform-windows/0.2.2/infernux.platform-windows.inxpkg", "https://github.com/ChenlizheMe/infernux_windows/releases/download/v0.2.2/infernux.platform-windows.inxpkg"],
  ["plugins/infernux.platform-linux/0.2.2/infernux.platform-linux.inxpkg", "https://github.com/ChenlizheMe/infernux_linux/releases/download/v0.2.2/infernux.platform-linux.inxpkg"],
  ["plugins/infernux.platform-android/0.2.4/infernux.platform-android.inxpkg", "https://github.com/ChenlizheMe/infernux_android/releases/download/v0.2.4/infernux.platform-android.inxpkg"],
  ["plugins/infernux.platform-web/0.2.2/infernux.platform-web.inxpkg", "https://github.com/ChenlizheMe/infernux_web/releases/download/v0.2.2/infernux.platform-web.inxpkg"],
  ["plugins/infernux.mcp/0.1.2/infernux.mcp.inxpkg", "https://github.com/ChenlizheMe/infernux_mcp/releases/download/v0.1.2/infernux.mcp.inxpkg"],
  ["plugins/infernux.platform-windows/0.2.1/infernux.platform-windows.inxpkg", "https://github.com/ChenlizheMe/infernux_windows/releases/download/v0.2.1/infernux.platform-windows.inxpkg"],
  ["plugins/infernux.platform-linux/0.2.1/infernux.platform-linux.inxpkg", "https://github.com/ChenlizheMe/infernux_linux/releases/download/v0.2.1/infernux.platform-linux.inxpkg"],
  ["plugins/infernux.platform-android/0.2.3/infernux.platform-android.inxpkg", "https://github.com/ChenlizheMe/infernux_android/releases/download/v0.2.3/infernux.platform-android.inxpkg"],
  ["plugins/infernux.platform-web/0.2.1/infernux.platform-web.inxpkg", "https://github.com/ChenlizheMe/infernux_web/releases/download/v0.2.1/infernux.platform-web.inxpkg"],
  [
    "plugins/infernux.mcp/0.1.1/infernux.mcp.inxpkg",
    "https://github.com/ChenlizheMe/infernux_mcp/releases/download/v0.1.1/infernux.mcp.inxpkg"
  ],
  [
    "plugins/infernux.platform-windows/0.2.0/infernux.platform-windows.inxpkg",
    "https://github.com/ChenlizheMe/infernux_windows/releases/download/v0.2.0/infernux.platform-windows.inxpkg"
  ],
  [
    "plugins/infernux.platform-linux/0.2.0/infernux.platform-linux.inxpkg",
    "https://github.com/ChenlizheMe/infernux_linux/releases/download/v0.2.0/infernux.platform-linux.inxpkg"
  ],
  [
    "plugins/infernux.platform-web/0.2.0/infernux.platform-web.inxpkg",
    "https://github.com/ChenlizheMe/infernux_web/releases/download/v0.2.0/infernux.platform-web.inxpkg"
  ],
  [
    "plugins/infernux.platform-android/0.2.2/infernux.platform-android.inxpkg",
    "https://github.com/ChenlizheMe/infernux_android/releases/download/v0.2.2/infernux.platform-android.inxpkg"
  ],
  [
    "android-support/0.1.0/infernux-android-support-0.1.0-windows-x64.inxkit",
    "https://github.com/ChenlizheMe/Infernux/releases/download/v0.4.0/infernux-android-support-0.1.0-windows-x64.inxkit"
  ],
  [
    "android-support/0.1.0/infernux-android-support-0.1.0-linux-x64.inxkit",
    "https://github.com/ChenlizheMe/Infernux/releases/download/v0.4.0/infernux-android-support-0.1.0-linux-x64.inxkit"
  ],
  ...[
    "InfernuxHub-0.4.0-linux-x64-full.zip",
    "InfernuxHub-0.4.0-windows-x64-full.zip",
    "InfernuxHub-linux-x64-manifest.json",
    "InfernuxHub-windows-x64-manifest.json",
    "InfernuxHubInstaller-0.4.0-linux-x64",
    "InfernuxHubInstaller-0.4.0-windows-x64.exe"
  ].map((name) => [
    `hub/0.4.0/${name}`,
    `https://github.com/ChenlizheMe/Infernux/releases/download/v0.4.0/${name}`
  ]),
  ...[
    "InfernuxHub-0.4.0-linux-x64-full.zip",
    "InfernuxHub-0.4.0-windows-x64-full.zip",
    "InfernuxHub-linux-x64-manifest.json",
    "InfernuxHub-windows-x64-manifest.json",
    "InfernuxHubInstaller-0.4.0-linux-x64",
    "InfernuxHubInstaller-0.4.0-windows-x64.exe"
  ].map((name) => [
    `hub/0.4.0/build-2/${name}`,
    `https://github.com/ChenlizheMe/Infernux/releases/download/v0.4.0/${name}`
  ])
]);
const encoder = new TextEncoder();
async function isAuthorized(request, secret) {
  const expected = encoder.encode(`Bearer ${secret}`);
  const provided = encoder.encode(request.headers.get("Authorization") ?? "");
  if (expected.byteLength !== provided.byteLength) {
    return false;
  }
  return crypto.subtle.timingSafeEqual(expected, provided);
}
function json(document, status = 200) {
  return Response.json(document, { status });
}
function requireKey(key) {
  if (!ALLOWED_KEYS.has(key) && !isHubReleaseKey(key)) {
    throw new RangeError("Object key is not authorized");
  }
  return key;
}
function isHubReleaseKey(key) {
  const match = /^hub\/(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\/build-(\d+)\/(.+)$/.exec(key ?? "");
  if (!match) {
    return false;
  }
  const [, version, build, filename] = match;
  if (!/^[1-9]\d*$/.test(build)) return false;
  const hubVersion = build === "1" ? version : `${version}-${build}`;
  return (new Set([
    `InfernuxHubInstaller-${hubVersion}-windows-x64.exe`,
    `InfernuxHubInstaller-${hubVersion}-linux-x64`,
    `InfernuxHub-${hubVersion}-windows-x64-full.zip`,
    `InfernuxHub-${hubVersion}-linux-x64-full.zip`,
    "InfernuxHub-windows-x64-manifest.json",
    "InfernuxHub-linux-x64-manifest.json"
  ])).has(filename);
}
export default {
  async fetch(request, env) {
    if (!await isAuthorized(request, env.UPLOAD_TOKEN)) {
      return json({ error: "unauthorized" }, 401);
    }
    const url = new URL(request.url);
    try {
      if (request.method === "POST" && url.pathname === "/start") {
        const body = await request.json();
        const key = requireKey(body.key);
        const upload = await env.BUCKET.createMultipartUpload(key, {
          httpMetadata: {
            contentType: body.contentType,
            cacheControl: body.cacheControl,
            contentDisposition: body.contentDisposition
          }
        });
        return json({ key: upload.key, uploadId: upload.uploadId });
      }
      if (request.method === "POST" && url.pathname === "/mirror") {
        const body = await request.json();
        const key = requireKey(body.key);
        const source = RELEASE_SOURCES.get(key);
        if (!source) {
          return json({ error: "release source is not authorized" }, 400);
        }
        const response = await fetch(source, {
          headers: { "User-Agent": "Infernux-Release-Publisher/0.4" },
          redirect: "follow"
        });
        if (!response.ok || response.body === null) {
          console.error(JSON.stringify({
            event: "mirror_source_error",
            key,
            status: response.status
          }));
          return json({ error: "release source failed", status: response.status }, 502);
        }
        const filename = key.slice(key.lastIndexOf("/") + 1);
        const contentType = filename.endsWith(".json") ? "application/json" : "application/octet-stream";
        const object = await env.BUCKET.put(key, response.body, {
          httpMetadata: {
            contentType,
            cacheControl: "public, max-age=31536000, immutable",
            contentDisposition: `attachment; filename="${filename}"`
          }
        });
        return json({ key: object.key, size: object.size, etag: object.etag });
      }
      if (request.method === "PUT" && url.pathname === "/part") {
        const key = requireKey(url.searchParams.get("key"));
        const uploadId = url.searchParams.get("uploadId");
        const partNumber = Number(url.searchParams.get("partNumber"));
        const contentLength = Number(request.headers.get("Content-Length"));
        if (!uploadId || !Number.isInteger(partNumber) || partNumber < 1 || partNumber > 1e4 || !Number.isInteger(contentLength) || contentLength < 1 || contentLength > 64 * 1024 * 1024 || request.body === null) {
          return json({ error: "invalid part" }, 400);
        }
        const upload = env.BUCKET.resumeMultipartUpload(key, uploadId);
        const part = await upload.uploadPart(partNumber, request.body);
        return json({ etag: part.etag, partNumber: part.partNumber });
      }
      if (request.method === "POST" && url.pathname === "/complete") {
        const body = await request.json();
        const key = requireKey(body.key);
        if (!body.uploadId || !Array.isArray(body.parts) || body.parts.length < 1) {
          return json({ error: "invalid completion" }, 400);
        }
        const upload = env.BUCKET.resumeMultipartUpload(key, body.uploadId);
        const object = await upload.complete(body.parts);
        return json({ key: object.key, size: object.size, etag: object.etag });
      }
      if (request.method === "POST" && url.pathname === "/abort") {
        const body = await request.json();
        const key = requireKey(body.key);
        const upload = env.BUCKET.resumeMultipartUpload(key, body.uploadId);
        await upload.abort();
        return json({ aborted: true });
      }
      return json({ error: "not found" }, 404);
    } catch (error) {
      console.error(JSON.stringify({ event: "upload_error", message: error.message }));
      if (error instanceof RangeError) {
        return json({ error: error.message }, 400);
      }
      return json({ error: "upload failed" }, 500);
    }
  }
};
