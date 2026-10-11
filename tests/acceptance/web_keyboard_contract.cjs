"use strict";

// Drive DOM keyboard events and read both the native and public Python Input
// APIs. A rendered frame alone cannot demonstrate that these keys are usable.
async function verifyWebKeyboard(page) {
  const checks = [];
  await page.locator("canvas").focus();
  const read = (scan) => page.evaluate((value) => ({
    native: Module.ccall("InfernuxWebGetKeyState", "number", ["number"], [value]),
    python: Module.ccall("InfernuxWebGetRuntimeDiagnostic", "number", ["number", "number"], [0, value]),
  }), scan);
  for (const [key, scan] of [
    ["a", 4], ["ShiftLeft", 225], ["ShiftRight", 229],
    ["ControlLeft", 224], ["ControlRight", 228], ["AltLeft", 226], ["AltRight", 230],
    ["MetaLeft", 227], ["MetaRight", 231], ["F1", 58], ["F12", 69],
    ["Delete", 76], ["Home", 74], ["PageDown", 78], ["Numpad1", 89],
    ["NumpadEnter", 88], ["NumpadDecimal", 99], [";", 51],
  ]) {
    await page.keyboard.down(key);
    const down = await read(scan);
    await page.keyboard.up(key);
    const up = await read(scan);
    checks.push({ key, scan, down, up });
  }
  await page.keyboard.down("ShiftLeft");
  await page.keyboard.down("ShiftRight");
  await page.keyboard.up("ShiftLeft");
  const left = await read(225);
  const right = await read(229);
  await page.keyboard.up("ShiftRight");
  const passed = checks.every(({down, up}) => (
    down.native === 1 && down.python === 1 && up.native === 0 && up.python === 0
  )) && left.native === 0 && left.python === 0 && right.native === 1 && right.python === 1;
  return { passed, checks, independentModifiers: { left, right } };
}

module.exports = { verifyWebKeyboard };
