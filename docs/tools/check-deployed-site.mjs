import process from "node:process";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { buildWebsiteHealthReport, renderWebsiteHealthSummary } from "./website-health-report.mjs";

const baseArg = process.argv.indexOf("--base-url");
const base = new URL(baseArg >= 0 ? process.argv[baseArg + 1] : "https://infernux-engine.com/");
const reportArg = process.argv.indexOf("--report");
const reportPath = reportArg >= 0 ? path.resolve(process.argv[reportArg + 1]) : null;
const failures = [];
const healthResults = [];
const startedAt = new Date();
let deployedManifest = null;
const requestAttempts = 3;
const requestTimeoutMs = 20_000;

// Availability and public information only; markup/classes are design choices.
const checks = [
    { route: "/" },
    { route: "/tutorials.html" },
    { route: "/download.html" },
    { route: "/roadmap.html" },
    { route: "/wiki/site/en/api/index.html" },
    { route: "/wiki/site/zh/api/index.html" },
    { route: "/release.json", jsonKey: "version" },
    { route: "/release-notes.json", jsonKey: "version" },
    { route: "/api-index.json", jsonKey: "symbols" },
    { route: "/docs-manifest.json", jsonKey: "documented_release" },
    { route: "/hub-catalog.json", jsonKey: "stable" },
    { route: "/platform-support.json", jsonKey: "released_version" },
];
const information = new Map();

function record(id, target, status, started, detail = null) {
    healthResults.push({
        id,
        kind: "route",
        target,
        status,
        duration_ms: Math.max(0, Math.round(performance.now() - started)),
        detail,
    });
}

function isRetryableStatus(status) {
    return status === 408 || status === 425 || status === 429 || status >= 500;
}

function wait(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchText(target) {
    let lastError = null;

    for (let attempt = 1; attempt <= requestAttempts; attempt += 1) {
        try {
            const response = await fetch(target, {
                headers: { "user-agent": "Infernux-website-health/1.0" },
                signal: AbortSignal.timeout(requestTimeoutMs),
            });
            if (!response.ok) {
                const error = new Error(`HTTP ${response.status}`);
                error.retryable = isRetryableStatus(response.status);
                throw error;
            }

            return { response, body: await response.text(), attempt };
        } catch (error) {
            lastError = error;
            const retryable = error.retryable !== false;
            if (!retryable || attempt === requestAttempts) break;

            console.warn(`RETRY ${target.pathname}: ${error.message} (attempt ${attempt}/${requestAttempts})`);
            await wait(attempt * 1_000);
        }
    }

    throw lastError;
}

for (const check of checks) {
    const target = new URL(check.route, base);
    const started = performance.now();
    try {
        const { response, body, attempt } = await fetchText(target);
        if (check.jsonKey) {
            const data = JSON.parse(body);
            if (!(check.jsonKey in data)) throw new Error(`JSON is missing '${check.jsonKey}'`);
            information.set(check.route, data);
            if (check.route === "/docs-manifest.json") deployedManifest = data;
        }
        const attemptDetail = attempt > 1 ? ` after ${attempt} attempts` : "";
        console.log(`PASS ${check.route}${attemptDetail}`);
        record(check.route, target.toString(), "passed", started, `HTTP ${response.status}${attemptDetail}`);
    } catch (error) {
        failures.push(`${check.route}: ${error.message}`);
        console.error(`FAIL ${check.route}: ${error.message}`);
        record(check.route, target.toString(), "failed", started, error.message);
    }
}

const published = information.get("/release.json");
if (published) {
    const started = performance.now();
    const mismatches = [];
    for (const [route, key] of [["/release-notes.json", "version"], ["/api-index.json", "generated_for_release"],
        ["/docs-manifest.json", "documented_release"], ["/platform-support.json", "released_version"]]) {
        if (information.has(route) && information.get(route)[key] !== published.version) mismatches.push(route);
    }
    const notes = information.get("/release-notes.json");
    if (notes && notes.tag !== published.tag) mismatches.push("release-notes tag");
    const catalog = information.get("/hub-catalog.json");
    const stable = catalog?.releases?.find(release => release.version === catalog.stable);
    if (catalog && stable?.release_url !== published.release_url) mismatches.push("Hub stable release");
    const detail = mismatches.length ? `Published information differs: ${mismatches.join(", ")}` : `Release ${published.tag}`;
    if (mismatches.length) failures.push(detail);
    record("release-consistency", new URL("/release.json", base).toString(),
        mismatches.length ? "failed" : "passed", started, detail);
}

const finishedAt = new Date();
const repository = process.env.GITHUB_REPOSITORY || "ChenlizheMe/Infernux";
const serverUrl = process.env.GITHUB_SERVER_URL || "https://github.com";
const runUrl = process.env.GITHUB_RUN_ID ? `${serverUrl}/${repository}/actions/runs/${process.env.GITHUB_RUN_ID}` : null;
const report = buildWebsiteHealthReport({
    checkedAt: process.env.WEBSITE_HEALTH_CHECKED_AT || finishedAt.toISOString(),
    baseUrl: base,
    startedAt,
    finishedAt,
    checks: healthResults,
    manifest: deployedManifest,
    pagesBuild: null,
    environment: {
        repository,
        checkoutCommit: process.env.GITHUB_SHA,
        workflow: process.env.GITHUB_WORKFLOW,
        runId: process.env.GITHUB_RUN_ID,
        runAttempt: process.env.GITHUB_RUN_ATTEMPT,
        runUrl,
    },
});

if (reportPath) {
    await mkdir(path.dirname(reportPath), { recursive: true });
    await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}
if (process.env.GITHUB_STEP_SUMMARY) {
    await appendFile(process.env.GITHUB_STEP_SUMMARY, renderWebsiteHealthSummary(report), "utf8");
}
if (failures.length) {
    console.error(`Deployed website health failed with ${failures.length} issue(s).`);
    process.exitCode = 1;
} else {
    console.log(`Deployed website health passed for ${base.origin}.`);
}
