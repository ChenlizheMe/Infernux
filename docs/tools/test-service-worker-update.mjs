import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../js/main.js', import.meta.url), 'utf8');
assert.match(source, /navigator\.serviceWorker\.register\("\/sw\.js"/);
assert.match(source, /function registerOfflineShell\(\)/);
assert.doesNotMatch(source, /site-update-notice|SKIP_WAITING|monitorServiceWorkerUpdates|serviceWorkerReloadRequested/);
console.log('Service Worker shell test passed: offline registration remains quiet and there is no update prompt.');
