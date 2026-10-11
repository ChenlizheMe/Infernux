"use strict";

// Exercise public gameplay APIs, then inspect pixels and the native attachment
// sample count. The script-created effect has no GUID and must remain editable.
async function verifyWebRenderSettings(page, readCanvas) {
  const checks = [];
  await page.locator("canvas").focus();
  for (const [key, samples, expected] of [
    ["5", 1, [108, 22, 8]], ["6", 4, [255, 73, 39]],
    ["7", 4, [204, 51, 26]], ["8", 4, null],
  ]) {
    const applied = page.waitForEvent("console", {
      predicate: message => message.text().includes(`INFERNUX_RENDER_SETTINGS_PROBE ${key}`),
      timeout: 10000,
    });
    await page.keyboard.down(key);
    try { await applied; } finally { await page.keyboard.up(key); }
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const image = await readCanvas();
    const pixels = [0.75, 0.9].map(x => {
      const offset = (Math.floor(image.height * .08) * image.width + Math.floor(image.width * x)) * 4;
      return [...image.data.subarray(offset, offset + 3)];
    });
    const actualSamples = await page.evaluate(() => Module.ccall(
      "InfernuxWebGetRuntimeDiagnostic", "number", ["number", "number"], [15, 0],
    ));
    const passed = actualSamples === samples && pixels.every(color => expected
      ? color.every((value, channel) => Math.abs(value - expected[channel]) <= 3)
      : color[2] > color[0] + 10 && color[2] > 30);
    checks.push({key, samples: actualSamples, expectedSamples: samples, pixels, expected, passed});
  }
  return {passed: checks.every(check => check.passed), checks};
}

module.exports = {verifyWebRenderSettings};
