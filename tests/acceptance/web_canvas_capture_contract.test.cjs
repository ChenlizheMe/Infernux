"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {chromium} = require("playwright");
const {PNG} = require("pngjs");
const {captureCanvas, readCanvasFrame} = require("./web_mobile_input_smoke.cjs");

function pixel(image, x, y) {
  const offset = (y * image.width + x) * 4;
  return [...image.data.subarray(offset, offset + 4)];
}

function near(actual, expected) {
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) <= 1,
    `${actual} differs from ${expected}`));
}

test("canvas pixels preserve alpha/sRGB while screen capture retains CSS geometry and compositing",
  {timeout: 30000}, async () => {
    const browser = await chromium.launch({channel: "chromium", headless: true, args: [
      "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    ]});
    try {
      const page = await browser.newPage({viewport: {width: 412, height: 315}, deviceScaleFactor: 2});
      await page.setContent(`<style>
        body { margin: 0; background: rgb(18,52,86); }
        canvas { position: absolute; left: 17px; top: 23px; width: 96px; height: 64px;
          transform: translate(5px,7px) scale(1.5); transform-origin: 0 0; }
        </style><canvas width="192" height="128"></canvas>`);
      await page.evaluate(() => {
        const canvas = document.querySelector("canvas");
        const context = canvas.getContext("2d", {colorSpace: "srgb"});
        context.fillStyle = "rgba(128,224,64,0.5)";
        context.fillRect(0, 0, 192, 128);
        context.fillStyle = "#d13f80";
        context.fillRect(96, 0, 96, 128);
      });
      const canvas = page.locator("canvas");
      const bitmap = await readCanvasFrame(canvas);
      const screen = PNG.sync.read(await captureCanvas(canvas));
      assert.deepEqual([bitmap.width, bitmap.height], [192, 128]);
      near(pixel(bitmap, 48, 64), [128, 224, 64, 128]);
      near(pixel(bitmap, 144, 64), [209, 63, 128, 255]);
      assert.deepEqual(await canvas.boundingBox(), {x: 22, y: 30, width: 144, height: 96});
      assert.deepEqual([screen.width, screen.height], [288, 192]);
      near(pixel(screen, 72, 96), [73, 138, 75, 255]);
      near(pixel(screen, 216, 96), [209, 63, 128, 255]);

      // Each read follows an explicitly acknowledged author frame. It reads
      // that canvas bitmap without asking the compositor for additional frames.
      for (const color of ["#cc331a", "#1a59b3"]) {
        await page.evaluate(value => new Promise(resolve => requestAnimationFrame(() => {
          const context = document.querySelector("canvas").getContext("2d");
          context.fillStyle = value;
          context.fillRect(0, 0, 192, 128);
          resolve();
        })), color);
        const frame = await readCanvasFrame(canvas);
        assert.deepEqual(pixel(frame, 48, 64), [...Buffer.from(color.slice(1), "hex"), 255]);
      }
    } finally {
      await browser.close();
    }
  });
