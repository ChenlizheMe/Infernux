"use strict";

// These keys belong to the authored multiplatform fixture, not a renderer
// diagnostic override. Verify the presented pixels after normal gameplay edits.
async function verifyWebCamera(page, readCanvas) {
  const checks = [];
  const sample = async () => {
    const image = await readCanvas();
    return [0.75, 0.9].map((x) => {
      const offset = (Math.floor(image.height * 0.08) * image.width + Math.floor(image.width * x)) * 4;
      return [...image.data.subarray(offset, offset + 3)];
    });
  };
  await page.locator("canvas").focus();
  const sky = await sample();
  for (const [key, expected] of [["1", [204, 51, 26]], ["2", [26, 89, 179]], ["3", [0, 0, 0]], ["4", null]]) {
    const applied = page.waitForEvent("console", {
      predicate: (message) => message.text().includes(`INFERNUX_CAMERA_PROBE ${key}`),
      timeout: 10000,
    });
    await page.keyboard.down(key);
    try { await applied; } finally { await page.keyboard.up(key); }
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const pixels = await sample();
    const target = expected ? [expected, expected] : sky;
    // The sky exposure can continue adapting during the input sequence. Its
    // restored gradient must be visible, without requiring time-identical pixels.
    const passed = expected
      ? pixels.every((color, i) => color.every((value, j) => Math.abs(value - target[i][j]) <= 3))
      : pixels.every((color) => color[2] > color[0] + 10 && color[2] > 30);
    checks.push({key, pixels, expected: target, passed});
  }
  return {passed: checks.every((check) => check.passed), sky, checks};
}

module.exports = { verifyWebCamera };
