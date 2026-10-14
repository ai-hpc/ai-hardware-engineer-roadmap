/* Learner-controlled layout. Mobile drawers remain owned by Material. */
(() => {
  "use strict";

  const root = document.documentElement;
  const toolbar = document.querySelector("[data-reading-controls]");
  if (!toolbar) return;

  const storageKey = "ai-hardware-engineer-roadmap.reading-layout.v1";
  // toolbar: false collapses the panel to a single "Reading view" button.
  const preferences = { navigation: true, contents: true, wide: false, toolbar: true };
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    for (const key of Object.keys(preferences)) {
      if (typeof saved?.[key] === "boolean") preferences[key] = saved[key];
    }
  } catch {
    // Private browsing or disabled storage must not disable the controls.
  }

  const navigation = document.querySelector(".md-sidebar--primary");
  const contents = document.querySelector(".md-sidebar--secondary");
  const main = document.querySelector(".md-main");
  const header = document.querySelector(".md-header");
  const status = toolbar.querySelector("[data-reading-status]");
  const button = action => toolbar.querySelector(`[data-reading-action="${action}"]`);
  const fullscreenButton = button("fullscreen");
  let focusView = false;
  let pendingSearch = false;

  toolbar.dataset.readingHasNavigation = String(!!navigation && !navigation.hidden);
  toolbar.dataset.readingHasToc = String(
    !!contents && !contents.hidden &&
    !!contents.querySelector(".md-nav__link") && !document.querySelector(".home-hero")
  );

  for (const [action, element] of [["navigation", navigation], ["contents", contents], ["wide", main], ["fullscreen", main]]) {
    if (!element) continue;
    element.id ||= `reading-${action}`;
    button(action)?.setAttribute("aria-controls", element.id);
  }

  const label = (action, text) => {
    const control = button(action);
    if (!control) return;
    control.querySelector("[data-reading-label]").textContent = text;
  };

  const measure = () => {
    root.style.setProperty("--reading-header-height", `${header?.getBoundingClientRect().height || 0}px`);
    // A collapsed panel is not sticky, so headings need no room for it.
    const toolbarHeight = preferences.toolbar ? toolbar.getBoundingClientRect().height : 0;
    root.style.setProperty("--reading-toolbar-height", `${toolbarHeight}px`);
  };

  const render = () => {
    root.dataset.readingNav = preferences.navigation ? "visible" : "hidden";
    root.dataset.readingToc = preferences.contents ? "visible" : "hidden";
    root.dataset.readingWide = String(preferences.wide);
    root.dataset.readingFocus = String(focusView);
    toolbar.dataset.readingCollapsed = String(!preferences.toolbar);
    button("toolbar")?.setAttribute("aria-expanded", String(preferences.toolbar));
    button("navigation")?.setAttribute("aria-expanded", String(preferences.navigation && !focusView));
    button("contents")?.setAttribute("aria-expanded", String(preferences.contents && !focusView));
    button("wide")?.setAttribute("aria-pressed", String(preferences.wide));
    fullscreenButton?.setAttribute("aria-pressed", String(focusView));
    label("navigation", preferences.navigation ? "Hide navigation" : "Show navigation");
    label("contents", preferences.contents ? "Hide contents" : "Show contents");
    label("wide", preferences.wide ? "Standard view" : "Wide view");
    label("fullscreen", focusView ? "Exit full screen" : "Full screen");
    label("toolbar", preferences.toolbar ? "Hide panel" : "Reading view");
    measure();
    // Material recomputes its sticky sidebars when the available space changes.
    window.dispatchEvent(new Event("resize"));
  };

  const save = () => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(preferences));
    } catch {
      // Preferences still work for the current lecture without persistence.
    }
  };

  const toggleFullscreen = async () => {
    if (status) status.textContent = "";
    try {
      if (document.fullscreenElement === root) {
        await document.exitFullscreen();
      } else {
        // Avoid preserving an open drawer/search behind the reading view.
        for (const id of ["__drawer", "__search"]) {
          const toggle = document.getElementById(id);
          if (toggle?.checked) toggle.click();
        }
        await root.requestFullscreen();
      }
    } catch {
      if (status) {
        status.textContent = "Full screen is unavailable here." +
          (window.matchMedia("(min-width: 60em)").matches ? " Use Wide view to expand the lesson." : "");
      }
    }
  };

  toolbar.addEventListener("click", event => {
    const control = event.target.closest("[data-reading-action]");
    if (!control || !toolbar.contains(control)) return;
    const action = control.dataset.readingAction;
    if (action === "fullscreen") {
      void toggleFullscreen();
    } else if (Object.hasOwn(preferences, action)) {
      preferences[action] = !preferences[action];
      if (status) status.textContent = "";
      save();
      render();
    }
  });

  document.addEventListener("fullscreenchange", () => {
    const wasFocused = focusView;
    focusView = document.fullscreenElement === root;
    render();
    if (wasFocused && !focusView) {
      if (pendingSearch) {
        pendingSearch = false;
        const toggle = document.getElementById("__search");
        if (toggle && !toggle.checked) toggle.click();
        const query = document.querySelector(".md-search__input");
        query?.focus();
        query?.select();
      } else {
        // Native Escape and the exit button return to a visible control.
        fullscreenButton?.focus({ preventScroll: true });
      }
    }
  });

  const revealSearch = async () => {
    if (pendingSearch) return;
    pendingSearch = true;
    try {
      await document.exitFullscreen();
    } catch {
      pendingSearch = false;
    }
  };

  // Material cannot focus its hidden search input in full-screen reading.
  // Preserve its shortcuts without intercepting browser shortcuts or typing.
  document.addEventListener("keydown", event => {
    if (!focusView || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey ||
        event.target.closest("input, textarea, select, [contenteditable]") ||
        !["f", "s", "/"].includes(event.key)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void revealSearch();
  }, true);

  document.getElementById("__search")?.addEventListener("change", event => {
    if (focusView && event.target.checked) void revealSearch();
  });

  if (fullscreenButton) {
    fullscreenButton.hidden = !(document.fullscreenEnabled && root.requestFullscreen);
    toolbar.dataset.readingFullscreenAvailable = String(!fullscreenButton.hidden);
  }
  toolbar.hidden = false;
  root.dataset.readingReady = "true";
  render();

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(measure);
    if (header) observer.observe(header);
    observer.observe(toolbar);
  }
  window.addEventListener("resize", measure, { passive: true });
})();
