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

    async function loadGithubStars() {
        const deck = document.querySelector("[data-github-stars]");
        const value = deck?.querySelector("[data-github-stars-value]");
        const caption = deck?.querySelector("[data-github-stars-caption]");
        if (!deck || !value || typeof globalThis.fetch !== "function") return;

        const zh = homeLanguage() === "zh";
        deck.dataset.githubStarsState = "loading";
        value.textContent = "…";
        if (caption) caption.textContent = zh ? "正在同步 GitHub" : "syncing GitHub";
        try {
            const response = await globalThis.fetch("https://api.github.com/repos/ChenlizheMe/Infernux", {
                headers: { Accept: "application/vnd.github+json" },
                cache: "no-store"
            });
            if (!response.ok) throw new Error(`GitHub responded ${response.status}`);
            const repository = await response.json();
            const stars = Number(repository.stargazers_count);
            if (!Number.isSafeInteger(stars) || stars < 0) throw new Error("GitHub returned an invalid star count");
            value.textContent = new Intl.NumberFormat(zh ? "zh-CN" : "en-US").format(stars);
            if (caption) caption.textContent = zh ? "GitHub 实时 Star" : "live GitHub stars";
            deck.dataset.githubStarsState = "ready";
        } catch (error) {
            value.textContent = "—";
            if (caption) caption.textContent = zh ? "暂时无法读取 GitHub" : "GitHub signal unavailable";
            deck.dataset.githubStarsState = "error";
            console.warn("Infernux GitHub star signal could not be loaded.", error);
        }
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("[data-home-code-copy]").forEach((button) => {
            button.dataset.state = "idle";
            setHomeCopyLabel(button, "idle");
            button.addEventListener("click", () => activateHomeCodeCopy(button));
        });
        loadGithubStars();
    });
    document.addEventListener("site:language-changed", () => {
        syncHomeCopy();
        loadGithubStars();
    });

    if (globalThis.__INFERNUX_HOME_TEST__) {
        globalThis.__infernuxHomeTest = { homeCopy, extractStarterCode, copyHomeText, loadGithubStars };
    }
})();
