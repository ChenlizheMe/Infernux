import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const docsRoot = path.resolve("docs");
const snapshotsRoot = path.join(docsRoot, "api-snapshots");
const check = process.argv.includes("--check");
const recordCurrent = process.argv.includes("--record-current");

const manifest = JSON.parse(await readFile(path.join(docsRoot, "docs-manifest.json"), "utf8"));
const apiIndex = JSON.parse(await readFile(path.join(docsRoot, "api-index.json"), "utf8"));
const release = manifest.documented_release;

function json(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function canonicalSymbols(snapshot) {
  // Published snapshots retain their original bytes. The package migration
  // changes an implementation import spelling, not the existing public API.
  return snapshot.symbols.map((symbol) => ({
    ...symbol,
    symbol_key: symbol.symbol_key.replace(/^Infernux\./, "infernux."),
    module: symbol.module.replace(/^Infernux\./, "infernux."),
    signatures: symbol.signatures.map((signature) => signature.replace(/\bInfernux\./g, "infernux.")),
  })).sort((a, b) => a.symbol_key.localeCompare(b.symbol_key, "en"));
}

function comparableSymbols(index) {
  return index.symbols
    .filter((symbol) => symbol.language === "en")
    .map((symbol) => ({
      symbol_key: symbol.symbol_key,
      module: symbol.module,
      symbol: symbol.symbol,
      kind: symbol.kind,
      signatures: symbol.signatures,
      status: symbol.status,
      since: symbol.since,
      canonical_url: symbol.canonical_url,
    }))
    .sort((a, b) => a.symbol_key.localeCompare(b.symbol_key, "en"));
}

function fingerprint(symbols) {
  return createHash("sha256").update(json(symbols)).digest("hex");
}

function snapshotFromIndex() {
  const symbols = comparableSymbols(apiIndex);
  return {
    schema_version: 1,
    release,
    symbol_count: symbols.length,
    fingerprint_sha256: fingerprint(symbols),
    symbols,
  };
}

function versionParts(version) {
  return version.split(/[.-]/).map((part) => /^\d+$/.test(part) ? Number(part) : part);
}

function compareVersions(a, b) {
  const left = versionParts(a);
  const right = versionParts(b);
  for (let i = 0; i < Math.max(left.length, right.length); i += 1) {
    const av = left[i] ?? 0;
    const bv = right[i] ?? 0;
    if (av === bv) continue;
    if (typeof av === "number" && typeof bv === "number") return av - bv;
    return String(av).localeCompare(String(bv), "en");
  }
  return 0;
}

async function readSnapshot(version) {
  return JSON.parse(await readFile(path.join(snapshotsRoot, `${version}.json`), "utf8"));
}

async function availableReleases() {
  return (await readdir(snapshotsRoot).catch(() => []))
    .filter((name) => /^\d+(?:\.\d+)+(?:[-.][\w-]+)?\.json$/.test(name))
    .map((name) => name.slice(0, -5))
    .sort(compareVersions);
}

function buildDiff(previous, current) {
  if (!previous) {
    return {
      schema_version: 1,
      current_release: current.release,
      previous_release: null,
      comparison_available: false,
      reason: "0.2.1 is the first recorded API snapshot; no earlier authoritative snapshot exists.",
      counts: { added: 0, removed: 0, changed: 0 },
      added: [],
      removed: [],
      changed: [],
    };
  }

  const before = new Map(canonicalSymbols(previous).map((symbol) => [symbol.symbol_key, symbol]));
  const after = new Map(canonicalSymbols(current).map((symbol) => [symbol.symbol_key, symbol]));
  const added = [...after.keys()].filter((key) => !before.has(key)).sort();
  const removed = [...before.keys()].filter((key) => !after.has(key)).sort();
  const changed = [];

  for (const key of [...after.keys()].filter((item) => before.has(item)).sort()) {
    const oldSymbol = before.get(key);
    const newSymbol = after.get(key);
    const fields = ["module", "symbol", "kind", "signatures", "status", "since"]
      .filter((field) => JSON.stringify(oldSymbol[field]) !== JSON.stringify(newSymbol[field]));
    if (fields.length) {
      changed.push({
        symbol_key: key,
        changed_fields: fields,
        before: Object.fromEntries(fields.map((field) => [field, oldSymbol[field]])),
        after: Object.fromEntries(fields.map((field) => [field, newSymbol[field]])),
        canonical_url: newSymbol.canonical_url,
      });
    }
  }

  return {
    schema_version: 1,
    current_release: current.release,
    previous_release: previous.release,
    comparison_available: true,
    counts: { added: added.length, removed: removed.length, changed: changed.length },
    added,
    removed,
    changed,
  };
}

await mkdir(snapshotsRoot, { recursive: true });
const expected = snapshotFromIndex();
const snapshotFile = path.join(snapshotsRoot, `${release}.json`);
const expectedText = json(expected);
let currentText = await readFile(snapshotFile, "utf8").catch(() => "");

if (recordCurrent) {
  if (currentText && currentText.replace(/\r\n/g, "\n") !== expectedText) {
    throw new Error(`Refusing to overwrite immutable API snapshot ${release}. Bump docs-manifest.json documented_release and release.json before recording a new release baseline.`);
  }
  if (!currentText) {
    await writeFile(snapshotFile, expectedText, "utf8");
    currentText = expectedText;
  }
}

const samePublishedApi = currentText && (() => {
  const recorded = JSON.parse(currentText);
  return recorded.release === expected.release
    && recorded.schema_version === expected.schema_version
    && json(canonicalSymbols(recorded)) === json(canonicalSymbols(expected));
})();
if (currentText.replace(/\r\n/g, "\n") !== expectedText && !samePublishedApi) {
  if (!currentText) {
    throw new Error(`API snapshot ${release} is missing. Record it only when ${release} is a new, intentional release baseline: node docs/tools/build-api-diff.mjs --record-current`);
  }
  const recorded = JSON.parse(currentText);
  const drift = buildDiff(recorded, expected);
  const changedPreview = drift.changed.slice(0, 12)
    .map((item) => `${item.symbol_key} (${item.changed_fields.join(", ")})`)
    .join("; ");
  const addedPreview = drift.added.slice(0, 12).join(", ");
  const removedPreview = drift.removed.slice(0, 12).join(", ");
  const details = [
    changedPreview && `changed: ${changedPreview}`,
    addedPreview && `added: ${addedPreview}`,
    removedPreview && `removed: ${removedPreview}`,
  ].filter(Boolean).join(" | ");
  throw new Error(
    `API index drifted from immutable release ${release}: ${drift.counts.added} added, ${drift.counts.removed} removed, ${drift.counts.changed} changed. ` +
    `Do not overwrite the published snapshot. Regenerate the API index from the ${release} release source, or bump documented_release and release.json before recording a new snapshot.` +
    (details ? ` ${details}` : "")
  );
}

const releases = await availableReleases();
const previousRelease = releases.filter((version) => compareVersions(version, release) < 0).at(-1) || null;
const previous = previousRelease ? await readSnapshot(previousRelease) : null;
const output = buildDiff(previous, expected);
const outputFile = path.join(docsRoot, "api-changes.json");
const outputText = json(output);

if (check) {
  const existing = await readFile(outputFile, "utf8").catch(() => "");
  if (existing.replace(/\r\n/g, "\n") !== outputText) {
    throw new Error("api-changes.json is stale. Run: node docs/tools/build-api-diff.mjs");
  }
  console.log(`Verified immutable API snapshot ${release} and its version comparison.`);
} else {
  await writeFile(outputFile, outputText, "utf8");
  console.log(previousRelease
    ? `Compared API ${release} with ${previousRelease}: ${output.counts.added} added, ${output.counts.removed} removed, ${output.counts.changed} changed.`
    : `Recorded API ${release} as the first comparison baseline; no earlier snapshot exists.`);
}
