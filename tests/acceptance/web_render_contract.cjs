"use strict";

function sampleBackdrop(image) {
  return [0.75, 0.9].map(x => {
    const offset = (Math.floor(image.height * 0.08) * image.width + Math.floor(image.width * x)) * 4;
    return [...image.data.subarray(offset, offset + 3)];
  });
}

function matchesBackdrop(pixels, expected) {
  return pixels.every((color, point) =>
    color.every((value, channel) => Math.abs(value - expected[point][channel]) <= 3));
}

// The script acknowledges the edit during Update. The view can present it on
// the next World publication, independently of the browser's animation clock.
async function verifyEdits(page, readCanvas, marker, edits, sky, timeout) {
  const checks = [];
  await page.locator("canvas").focus();
  for (const [key, expected, expectedSamples] of edits) {
    await page.keyboard.down(key);
    const receipt = `${marker} ${key}`;
    try {
      await page.waitForFunction(value => {
        const diagnostics = JSON.parse(document.querySelector("canvas").dataset.infernuxDiagnostics);
        return diagnostics.some(message => message.includes(value));
      }, receipt, {polling: 100, timeout});
    } catch (error) {
      throw new Error(`The running game did not acknowledge "${receipt}"`, {cause: error});
    } finally {
      await page.keyboard.up(key);
    }

    let check;
    for (let attempt = 0; attempt < 3; ++attempt) {
      const frame = await page.evaluate(() =>
        Module.ccall("InfernuxWebGetRuntimeDiagnostic", "number", ["number", "number"], [16, 0]));
      const pixels = sampleBackdrop(await readCanvas());
      const samples = expectedSamples === undefined ? undefined : await page.evaluate(() =>
        Module.ccall("InfernuxWebGetRuntimeDiagnostic", "number", ["number", "number"], [15, 0]));
      const expectedPixels = expected ? [expected, expected] : sky;
      check = {key, frame, pixels, expected: expectedPixels, samples, expectedSamples,
               passed: samples === expectedSamples && matchesBackdrop(pixels, expectedPixels)};
      if (check.passed || attempt === 2) break;
      await page.waitForFunction(previous =>
        Module.ccall("InfernuxWebGetRuntimeDiagnostic", "number", ["number", "number"], [16, 0]) > previous,
        frame, {polling: 100, timeout});
    }
    checks.push(check);
    process.stdout.write(`${receipt} ${JSON.stringify(check)}\n`);
    if (!check.passed) break;
  }
  return {passed: checks.every(check => check.passed), checks};
}

async function verifyWebCamera(page, readCanvas, timeout) {
  const sky = sampleBackdrop(await readCanvas());
  if (!sky.every(([red, , blue]) => blue > red + 10 && blue > 30))
    throw new Error(`The fixture did not start with its blue sky: ${JSON.stringify(sky)}`);
  const result = await verifyEdits(page, readCanvas, "INFERNUX_CAMERA_PROBE", [
    ["1", [204, 51, 26]], ["2", [26, 89, 179]], ["3", [0, 0, 0]], ["4", null],
  ], sky, timeout);
  return {...result, sky};
}

async function verifyWebRenderSettings(page, readCanvas, timeout) {
  const sky = sampleBackdrop(await readCanvas());
  return verifyEdits(page, readCanvas, "INFERNUX_RENDER_SETTINGS_PROBE", [
    ["5", [108, 22, 8], 1], ["6", [255, 73, 39], 4], ["7", [204, 51, 26], 4],
    ["8", null, 4], ["9", [26, 89, 179], 4], ["0", null, 4],
  ], sky, timeout);
}

module.exports = {verifyWebCamera, verifyWebRenderSettings};
