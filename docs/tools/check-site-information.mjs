import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Check published facts and destinations, independently of the site's design.
export async function checkSiteInformation(repoRoot = process.cwd()) {
    const docs = path.join(repoRoot, "docs");
    const json = async name => JSON.parse(await readFile(path.join(docs, name), "utf8"));
    const [release, notes, manifest, index, changes, catalog, platforms] = await Promise.all(
        ["release.json", "release-notes.json", "docs-manifest.json", "api-index.json",
            "api-changes.json", "hub-catalog.json", "platform-support.json"].map(json)
    );
    const errors = [];
    const require = (condition, message) => { if (!condition) errors.push(message); };
    const version = release.version;
    require(/^\d+\.\d+\.\d+$/.test(version), "release.json: invalid release version");
    const tag = release.tag?.match(/^v(\d+\.\d+\.\d+)(?:-v([1-9]\d*))?$/);
    require(tag?.[1] === version, "release.json: tag/version mismatch");
    for (const [label, actual] of Object.entries({
        "release notes": notes.version, "documentation": manifest.documented_release,
        "API index": index.generated_for_release, "API changes": changes.current_release,
        "platform release": platforms.released_version,
    })) require(actual === version, `${label}: version ${actual} differs from ${version}`);
    require(notes.tag === release.tag, "release notes: tag mismatch");
    const project = await readFile(path.join(repoRoot, "pyproject.toml"), "utf8");
    const engineVersion = project.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
    require(platforms.development_version === engineVersion, "platform development version differs from pyproject.toml");
    if (engineVersion !== version) {
        const parts = value => String(value).split(".").map(Number);
        const current = parts(engineVersion), published = parts(version);
        const firstDifference = current.findIndex((value, i) => value !== published[i]);
        require(current.length === 3 && firstDifference >= 0 && current[firstDifference] > published[firstDifference],
            "development version must not precede the published version");
        for (const name of ["UpdateLog.md", "UpdateLog-zh.md"]) {
            const text = await readFile(path.join(repoRoot, name), "utf8");
            require(text.startsWith(`# Infernux v${engineVersion} `), `${name}: missing upcoming version heading`);
        }
    }

    function downloadUrl(value, name, label) {
        try {
            const url = new URL(value);
            require(url.protocol === "https:" && !url.username && !url.password,
                `${label}: download must use HTTPS without credentials`);
            require(decodeURIComponent(url.pathname.split("/").pop()) === name,
                `${label}: URL filename differs from ${name}`);
        } catch { errors.push(`${label}: invalid download URL`); }
    }
    const assets = release.assets || [];
    require(assets.length > 0, "release.json: download assets are empty");
    require(new Set(assets.map(asset => asset.name)).size === assets.length, "release.json: duplicate asset names");
    for (const asset of assets) {
        require(asset.name?.includes(version), `${asset.name}: wrong release version`);
        require(Number.isInteger(asset.size_bytes) && asset.size_bytes > 0, `${asset.name}: missing download size`);
        downloadUrl(asset.url, asset.name, asset.name);
        if (asset.fallback_url) {
            downloadUrl(asset.fallback_url, asset.name, `${asset.name} alternate`);
            require(new URL(asset.fallback_url).pathname.includes(`/releases/download/${release.tag}/`),
                `${asset.name}: alternate download targets another release`);
        }
    }
    const stable = catalog.releases?.find(item => item.version === catalog.stable);
    require(Boolean(stable), "Hub catalog: stable release is missing");
    const build = Number(tag?.[2] || 1);
    require(catalog.stable === (build === 1 ? version : `${version}-${build}`), "Hub stable version differs from release tag");
    require(stable?.release_url === release.release_url && notes.release_url === release.release_url,
        "release notes/Hub release destinations differ");
    require(release.release_url?.endsWith(`/releases/tag/${release.tag}`), "release URL differs from release tag");
    for (const platform of platforms.platforms || []) {
        if (!platform.released || !platform.editor) continue;
        const installer = stable?.platforms?.[platform.id]?.installer;
        const asset = assets.find(item => item.kind === "hub-installer" && item.name === installer?.name);
        require(Boolean(asset) && asset.url === installer?.url && asset.fallback_url === installer?.fallback_url,
            `${platform.id}: catalog installer and release asset differ`);
        const wheel = platform.id === "windows-x64" ? "win_amd64" : platform.id === "linux-x64" ? "manylinux" : platform.id;
        require(assets.some(item => item.kind === "python-wheel" && item.name.includes(wheel)
            && item.name.includes(platforms.python_abi)), `${platform.id}: released editor wheel/ABI missing`);
    }

    const snapshot = await json(`api-snapshots/${version}.json`);
    require(snapshot.release === version, "API snapshot: wrong release");
    const symbols = index.symbols || [];
    require(symbols.length > 0 && index.symbol_count === symbols.length, "API index: invalid symbol count");
    require(new Set(symbols.map(symbol => symbol.id)).size === symbols.length, "API index: duplicate identifiers");
    const exists = async file => stat(file).then(info => info.isFile()).catch(() => false);
    for (const [folder, language] of [["en", "en"], ["zh", "zh-CN"]]) {
        const localized = symbols.filter(symbol => symbol.language === language);
        const keys = new Set(localized.map(symbol => symbol.symbol_key));
        const routes = new Set(localized.map(symbol => symbol.url));
        const sourceRoot = path.join(docs, "wiki/docs", folder, "api");
        const files = (await readdir(sourceRoot)).filter(name => name.endsWith(".md") && name !== "index.md");
        require(files.length === localized.length && keys.size === localized.length,
            `${language}: API sources and index entries are incomplete or duplicated`);
        for (const symbol of snapshot.symbols || []) {
            // A published API is identified by its document route. Import
            // spelling in prose may change without removing the API itself.
            const route = new URL(symbol.canonical_url).pathname.replace("/en/api/", `/${folder}/api/`);
            require(routes.has(route), `${language}: missing published API ${symbol.symbol_key}`);
        }
        const sources = new Set();
        for (const symbol of localized) {
            sources.add(path.basename(symbol.source || ""));
            require(symbol.source?.startsWith(`docs/wiki/docs/${folder}/api/`), `${symbol.id}: invalid source location`);
            require(await exists(path.resolve(repoRoot, symbol.source || "")), `${symbol.id}: source document missing`);
            for (const route of [symbol.url, symbol.counterpart_url]) {
                require(/^\/wiki\/site\/(en|zh)\/api\/[^/]+\.html$/.test(route || ""), `${symbol.id}: invalid API route`);
                require(await exists(path.join(docs, String(route).replace(/^\//, ""))), `${symbol.id}: API destination missing: ${route}`);
            }
        }
        require(files.every(file => sources.has(file)), `${language}: API documents missing from the index`);
    }
    if (errors.length) throw new Error(`Website information errors:\n${errors.join("\n")}`);
    return { version, tag: release.tag, apiSymbols: symbols.length, downloads: assets.length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
    console.log("Website information verified:", await checkSiteInformation());
}
