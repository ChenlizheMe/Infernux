"use strict";

// The caller serves a deliberately invalid game package. A Console error alone
// is insufficient: the host must stop loading and never advertise a ready game.
async function verifyWebStartupFailure(page, url, expectedDiagnostic, timeoutMs = 45000) {
  const messages = [];
  const pageErrors = [];
  page.on("console", message => messages.push({ type: message.type(), text: message.text() }));
  page.on("pageerror", error => pageErrors.push(String(error)));
  await page.goto(url);
  await page.waitForFunction(
    () => document.querySelector("canvas")?.dataset.infernuxState === "aborted",
    null, { timeout: timeoutMs },
  );
  const state = await page.evaluate(() => ({
    canvas: document.querySelector("canvas").dataset.infernuxState,
    status: document.querySelector("#status").textContent,
    hint: document.querySelector("#infernux-loader-hint").textContent,
    loaderVisible: !document.querySelector("#infernux-loader").classList.contains("hidden"),
  }));
  const passed = state.loaderVisible && state.status.includes("Player stopped:")
    && messages.some(message => message.text.includes(expectedDiagnostic))
    && !messages.some(message => message.text.includes("INFERNUX_WEB_SCENE_READY")
      || message.text.includes("INFERNUX_WEB_FIRST_FRAME_READY"));
  return { passed, state, messages, pageErrors };
}

module.exports = { verifyWebStartupFailure };
