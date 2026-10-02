(() => {
    "use strict";
    const deck = document.querySelector("[data-example-deck]");
    if (!deck) return;
    const tabs = [...deck.querySelectorAll("[data-example-filter]")];
    const cards = [...deck.querySelectorAll("[data-example-target]")];
    const apply = (target) => {
        tabs.forEach(tab => {
            const active = tab.dataset.exampleFilter === target;
            tab.classList.toggle("is-active", active);
            tab.setAttribute("aria-selected", String(active));
        });
        cards.forEach(card => {
            const visible = target === "all" || card.dataset.exampleTarget.split(" ").includes(target);
            card.hidden = !visible;
        });
    };
    tabs.forEach(tab => tab.addEventListener("click", () => apply(tab.dataset.exampleFilter)));
})();
