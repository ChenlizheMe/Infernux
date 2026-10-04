import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("./r2-upload-worker.mjs", import.meta.url), "utf8");
const context = vm.createContext({ TextEncoder });
vm.runInContext(source.replace("export default", "globalThis.worker =") + "\nglobalThis.validateKey = requireKey; globalThis.releaseSources = RELEASE_SOURCES;", context);
let packages;
if (process.argv[2]) {
  const catalog = JSON.parse(await readFile(process.argv[2], "utf8"));
  assert.equal(catalog.$schema, "infernux.official_plugin_registry");
  packages = catalog.packages;
} else {
  const catalog = JSON.parse(await readFile(new URL("../../external/plugins/plugins.json", import.meta.url), "utf8"));
  packages = await Promise.all(catalog.plugins.map(async (entry) => ({
    ...JSON.parse(await readFile(new URL(`../../external/plugins/${entry.path}/package/inx_package.json`, import.meta.url), "utf8")),
    repository: entry.repository,
  })));
}
assert.ok(Array.isArray(packages) && packages.length > 0);
for (const manifest of packages) {
  const artifact = `${manifest.reference.replaceAll("/", ".")}.inxpkg`;
  const key = `plugins/${manifest.reference.replaceAll("/", ".")}/${manifest.version}/${artifact}`;
  assert.equal(context.validateKey(key), key);
  assert.equal(context.releaseSources.get(key), `${manifest.repository}/releases/download/v${manifest.version}/${artifact}`);
  assert.throws(() => context.validateKey(key.replace(`/${manifest.version}/`, "/unpublished/")), /not authorized/);
}
for (const key of [
  "hub/0.4.1/build-1/InfernuxHubInstaller-0.4.1-linux-x64",
  "hub/0.4.1/build-2/InfernuxHubInstaller-0.4.1-2-linux-x64",
  "hub/0.4.1/build-2/InfernuxHub-0.4.1-2-windows-x64-full.zip",
  "hub/0.4.1/build-2/InfernuxHub-linux-x64-manifest.json",
  "hub/0.4.0/build-2/InfernuxHubInstaller-0.4.0-linux-x64",
]) assert.equal(context.validateKey(key), key);
for (const key of [
  "hub/0.4.1/build-2/InfernuxHubInstaller-0.4.1-linux-x64",
  "hub/0.4.1/build-2/InfernuxHubInstaller-0.4.1-3-linux-x64",
  "hub/0.4.1/build-0/InfernuxHubInstaller-0.4.1-0-linux-x64",
  "hub/0.4.1/build-2/../../unrelated",
]) assert.throws(() => context.validateKey(key), /not authorized/);
console.log("Current official plugin releases and Hub upload keys verified.");
