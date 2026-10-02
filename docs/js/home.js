(() => {
    "use strict";

    const HOME_COPY_RESET_MS = 2200;

    function homeLanguage() {
        return document.documentElement.lang?.toLowerCase().startsWith("zh") ? "zh" : "en";
    }

    function homeCopy(key) {
        const messages = {
            en: {
                idle: "Copy starter component",
                success: "Starter component copied",
                failure: "Could not copy the starter component. Select the code manually."
            },
            zh: {
                idle: "复制起步组件",
                success: "已复制起步组件",
                failure: "无法复制起步组件，请手动选择代码。"
            }
        };
        return messages[homeLanguage()][key];
    }

    function extractStarterCode(button) {
        const code = button?.closest(".code-preview")?.querySelector("code");
        const source = typeof code?.textContent === "string" ? code.textContent.replace(/\r\n?/g, "\n").trim() : "";
        return source ? `${source}\n` : "";
    }

    function fallbackHomeCopy(text) {
        const textarea = document.createElement("textarea");
        textarea.className = "home-copy-fallback";
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        document.body.appendChild(textarea);
        textarea.select();
        let copied = false;
        try {
            copied = typeof document.execCommand === "function" && document.execCommand("copy") === true;
        } catch {
            copied = false;
        }
        textarea.remove();
        return copied;
    }

    async function copyHomeText(value) {
        const text = String(value || "");
        if (!text.trim()) return false;
        try {
            if (globalThis.navigator?.clipboard?.writeText) {
                await globalThis.navigator.clipboard.writeText(text);
                return true;
            }
        } catch {}
        return fallbackHomeCopy(text);
    }

    function setHomeCopyLabel(button, key) {
        const label = homeCopy(key);
        const visible = button.querySelector("span");
        if (visible) visible.textContent = label;
        button.setAttribute("aria-label", label);
        button.title = label;
    }

    async function activateHomeCodeCopy(button) {
        if (!button || button.disabled) return;
        const source = extractStarterCode(button);
        if (!source) return;
        button.disabled = true;
        const copied = await copyHomeText(source);
        button.disabled = false;
        button.dataset.state = copied ? "success" : "failure";
        const icon = button.querySelector("i");
        if (icon) icon.className = copied ? "fas fa-check" : "fas fa-copy";
        const status = document.getElementById("home-code-copy-status");
        if (status) status.textContent = homeCopy(copied ? "success" : "failure");
        setHomeCopyLabel(button, copied ? "success" : "failure");
        if (button.homeCopyTimer) window.clearTimeout(button.homeCopyTimer);
        button.homeCopyTimer = window.setTimeout(() => {
            button.dataset.state = "idle";
            const restoredIcon = button.querySelector("i");
            if (restoredIcon) restoredIcon.className = "fas fa-copy";
            setHomeCopyLabel(button, "idle");
            button.homeCopyTimer = null;
        }, HOME_COPY_RESET_MS);
    }

    function syncHomeCopy() {
        document.querySelectorAll("[data-home-code-copy]").forEach((button) => {
            if (button.dataset.state !== "success" && button.dataset.state !== "failure") setHomeCopyLabel(button, "idle");
        });
    }

    function syncOrbitControl(button, spinning) {
        const zh = homeLanguage() === "zh";
        const labels = spinning
            ? (zh ? ["停止预览", "预览 / 运行"] : ["Stop preview", "PREVIEW / RUNNING"])
            : (zh ? ["旋转预览", "预览 / 空闲"] : ["Spin preview", "PREVIEW / IDLE"]);
        const status = button.closest(".hero-panel")?.querySelector("[data-orbit-status]");
        button.textContent = labels[0];
        button.setAttribute("aria-pressed", String(spinning));
        if (status) status.textContent = labels[1];
    }

    function syncOrbitLanguage() {
        document.querySelectorAll("[data-orbit-action=spin]").forEach((button) => {
            syncOrbitControl(button, button.getAttribute("aria-pressed") === "true");
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("[data-home-code-copy]").forEach((button) => {
            button.dataset.state = "idle";
            setHomeCopyLabel(button, "idle");
            button.addEventListener("click", () => activateHomeCodeCopy(button));
        });

        document.querySelectorAll("[data-orbit-action=spin]").forEach((button) => {
            syncOrbitControl(button, false);
            button.addEventListener("click", () => {
                const art = button.closest(".hero-panel")?.querySelector("[data-orbit-art]");
                const spinning = art?.classList.toggle("is-spinning") || false;
                syncOrbitControl(button, spinning);
            });
        });

    });
    document.addEventListener("site:language-changed", syncHomeCopy);
    document.addEventListener("site:language-changed", syncOrbitLanguage);

    if (globalThis.__INFERNUX_HOME_TEST__) {
        globalThis.__infernuxHomeTest = { homeCopy, extractStarterCode, copyHomeText };
    }
})();
