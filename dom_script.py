dom_script = r"""
(() => {
  const SHOULD_LOG_INPUT = window.__FP_LOG_EACH_KEYSTROKE__ === true;

  function nowMs() { return Date.now(); }
  function safeText(t) {
    if (!t) return "";
    return String(t).replace(/\s+/g, " ").trim();
  }

  function hash32(str) {
    // FNV-1a 32-bit
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    return ("00000000" + h.toString(16)).slice(-8);
  }

  function resolveIdRefs(idList) {
    const ids = String(idList || "").trim().split(/\s+/).filter(Boolean);
    if (!ids.length) return "";
    const parts = [];
    for (const id of ids) {
      const n = document.getElementById(id);
      if (!n) continue;
      const t = safeText(n.textContent || n.innerText || "");
      if (t) parts.push(t);
    }
    return safeText(parts.join(" "));
  }

  function labelTextForInput(el) {
    if (!el) return "";
    const aria = el.getAttribute && el.getAttribute("aria-label");
    if (aria) return safeText(aria);

    const by = el.getAttribute && el.getAttribute("aria-labelledby");
    if (by) {
      const t = resolveIdRefs(by);
      if (t) return t;
    }

    const id = el.getAttribute && el.getAttribute("id");
    if (id) {
      const lbl = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (lbl) return safeText(lbl.innerText || lbl.textContent || "");
    }

    const wrap = el.closest && el.closest("label");
    if (wrap) return safeText(wrap.innerText || wrap.textContent || "");

    return "";
  }

  function elementDescriptor(el) {
    if (!el) return {};
    const tag = (el.tagName || "").toLowerCase();
    const type = el.getAttribute && (el.getAttribute("type") || "");
    const role = el.getAttribute && (el.getAttribute("role") || "");
    const id = el.getAttribute && (el.getAttribute("id") || "");
    const nameAttr = el.getAttribute && (el.getAttribute("name") || "");
    const ariaLabel = el.getAttribute && (el.getAttribute("aria-label") || "");
    const classes = safeText((el.className || "").toString()).slice(0, 300);

    const a11y_name =
      safeText(ariaLabel) ||
      safeText(labelTextForInput(el)) ||
      safeText(nameAttr) ||
      safeText(id);

    return {
      tag,
      type,
      role,
      id: safeText(id),
      name_attr: safeText(nameAttr),
      classes,
      a11y_name: a11y_name.slice(0, 300),
    };
  }

  function cssPath(el) {
    try {
      if (!el || !el.nodeType || el.nodeType !== 1) return "";
      const parts = [];
      let cur = el;
      let depth = 0;
      while (cur && depth < 12) {
        let part = cur.tagName.toLowerCase();
        if (cur.id) {
          part += `#${CSS.escape(cur.id)}`;
          parts.unshift(part);
          break;
        } else {
          let i = 1;
          let sib = cur;
          while ((sib = sib.previousElementSibling)) {
            if (sib.tagName === cur.tagName) i++;
          }
          part += `:nth-of-type(${i})`;
        }
        parts.unshift(part);
        cur = cur.parentElement;
        depth++;
      }
      return parts.join(" > ").slice(0, 600);
    } catch (e) {
      return "";
    }
  }

  function xPath(el) {
    try {
      if (!el || el.nodeType !== 1) return "";
      const parts = [];
      let cur = el;
      let depth = 0;
      while (cur && depth < 12) {
        let idx = 1;
        let sib = cur;
        while ((sib = sib.previousElementSibling)) {
          if (sib.tagName === cur.tagName) idx++;
        }
        parts.unshift(`${cur.tagName.toLowerCase()}[${idx}]`);
        cur = cur.parentElement;
        depth++;
      }
      return "/" + parts.join("/").slice(0, 600);
    } catch (e) {
      return "";
    }
  }

  function viewportMetrics() {
    try {
      const vv = window.visualViewport;
      return {
        device_pixel_ratio: (window.devicePixelRatio || 1),
        inner: { width: (window.innerWidth || 0), height: (window.innerHeight || 0) },
        outer: { width: (window.outerWidth || 0), height: (window.outerHeight || 0) },
        visual_viewport: vv ? {
          width: vv.width,
          height: vv.height,
          offsetLeft: vv.offsetLeft,
          offsetTop: vv.offsetTop,
          pageLeft: vv.pageLeft,
          pageTop: vv.pageTop,
          scale: vv.scale
        } : null,
      };
    } catch (e) {
      return { device_pixel_ratio: 1 };
    }
  }

  function bboxForElement(el) {
    try {
      if (!el || !el.getBoundingClientRect) return {};
      const r = el.getBoundingClientRect();
      const vw = window.innerWidth || 1;
      const vh = window.innerHeight || 1;

      const x1 = r.left, y1 = r.top, x2 = r.right, y2 = r.bottom;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;

      return {
        viewport: { width: vw, height: vh },
        scroll: { x: window.scrollX || 0, y: window.scrollY || 0 },
        bbox_px: [x1, y1, x2, y2],
        click_px: [cx, cy],
        bbox_norm: [x1 / vw, y1 / vh, x2 / vw, y2 / vh],
        click_norm: [cx / vw, cy / vh],
      };
    } catch (e) {
      return {};
    }
  }

  function valueSnapshot(el) {
    try {
      if (!el) return {};
      const tag = (el.tagName || "").toLowerCase();

      if (el.isContentEditable) {
        const t = safeText(el.innerText || el.textContent || "");
        return { field_kind: "contenteditable", value: t.slice(0, 2000) };
      }

      if (tag === "input" || tag === "textarea") {
        const type = (el.getAttribute("type") || "").toLowerCase();
        if (type === "checkbox" || type === "radio") {
          return { field_kind: type, checked: !!el.checked, value: (el.value ?? "") };
        }
        return { field_kind: tag, value: (el.value ?? "") };
      }

      if (tag === "select") {
        const idx = el.selectedIndex;
        const opt = idx >= 0 ? el.options[idx] : null;
        return {
          field_kind: "select",
          value: (el.value ?? ""),
          selected_text: safeText(opt ? opt.textContent : "")
        };
      }

      const role = (el.getAttribute("role") || "").toLowerCase();
      if (role === "checkbox" || role === "radio") {
        return { field_kind: role, checked: (el.getAttribute("aria-checked") === "true") };
      }
      if (role === "option") {
        return { field_kind: "option", selected: (el.getAttribute("aria-selected") === "true") };
      }

      return { field_kind: tag || role || "unknown" };
    } catch (e) {
      return {};
    }
  }

  function textSnippet(el) {
    try {
      const txt = safeText((el.innerText || el.textContent || ""));
      return txt.slice(0, 300);
    } catch (e) {
      return "";
    }
  }

  function originOf(url) {
    try { return new URL(url).origin; }
    catch (e) { return (location && location.origin) ? location.origin : ""; }
  }

  function normalizeName(s) {
    return safeText(s).slice(0, 160);
  }

  function computeElementId(d, css_path, xpath, frame_url) {
    const origin = originOf(frame_url || location.href);

    if (d.id) {
      const seedA = `A|o=${origin}|tag=${d.tag}|id=${d.id}`;
      return { element_id: "eid:" + hash32(seedA), element_id_seed: seedA, element_id_tier: "A", origin };
    }

    if (d.name_attr) {
      const seedB = `B|o=${origin}|tag=${d.tag}|name=${d.name_attr}|type=${d.type}|role=${d.role}`;
      return { element_id: "eid:" + hash32(seedB), element_id_seed: seedB, element_id_tier: "B", origin };
    }

    if (d.role || d.a11y_name) {
      const seedC = `C|o=${origin}|tag=${d.tag}|role=${d.role}|type=${d.type}|a11y=${normalizeName(d.a11y_name)}`;
      return { element_id: "eid:" + hash32(seedC), element_id_seed: seedC, element_id_tier: "C", origin };
    }

    if (css_path) {
      const seedD = `D|o=${origin}|css=${css_path}`;
      return { element_id: "eid:" + hash32(seedD), element_id_seed: seedD, element_id_tier: "D", origin };
    }

    const seedE = `E|o=${origin}|xpath=${xpath || ""}`;
    return { element_id: "eid:" + hash32(seedE), element_id_seed: seedE, element_id_tier: "E", origin };
  }

  function escapeCssAttr(v) {
    return String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  }

  function buildSelectorCandidates(d, css_path, xpath) {
    const out = [];

    if (d.role && d.a11y_name) {
      out.push({ kind: "role_name", role: d.role, name: d.a11y_name });
    }

    if (d.id) {
      out.push({ kind: "css", value: `#${d.id}` });
      out.push({ kind: "css", value: `${d.tag}#${d.id}` });
    }

    if (d.name_attr) {
      out.push({ kind: "css", value: `${d.tag}[name="${escapeCssAttr(d.name_attr)}"]` });
      out.push({ kind: "css", value: `[name="${escapeCssAttr(d.name_attr)}"]` });
    }

    const aria = safeText(d.a11y_name || "");
    if (aria && aria.length <= 160) {
      out.push({ kind: "text", value: aria });
    }

    if (css_path) out.push({ kind: "css_path", value: css_path });
    if (xpath) out.push({ kind: "xpath", value: xpath });

    return out;
  }

  // ---- UPDATED DEDUPE ----
  const lastByKey = new Map();
  const DEDUPE_WINDOW_MS = 600;

  function stableValueKey(kind, payload) {
    const v = payload || {};
    const isCheck =
      v.field_kind === "checkbox" || v.field_kind === "radio" ||
      v.role === "checkbox" || v.role === "radio";

    const checked = isCheck ? (v.checked ? "1" : "0") : "";

    const val =
      (typeof v.value === "string" || typeof v.value === "number")
        ? String(v.value).slice(0, 200)
        : "";

    const selected = (v.field_kind === "option") ? (v.selected ? "1" : "0") : "";

    return [
      kind,
      v.element_id || "",
      v.element_fingerprint || "",
      v.tag || "",
      v.role || "",
      v.type || "",
      checked,
      selected,
      val
    ].join("|").slice(0, 900);
  }

  function emit(kind, el, extra = {}) {
    try {
      if (!window.logUiEvent || !el) return;

      const d = elementDescriptor(el);
      const css_path = cssPath(el);
      const xpath = xPath(el);
      const v = valueSnapshot(el);
      const b = bboxForElement(el);
      const vm = viewportMetrics();

      const frame_url = window.location.href;

      const fpSeed = `${d.tag}|${d.role}|${d.id}|${d.name_attr}|${d.a11y_name}|${css_path}|${xpath}|${location.origin}`;
      const element_fingerprint = hash32(fpSeed);

      const eid = computeElementId(d, css_path, xpath, frame_url);
      const selector_candidates = buildSelectorCandidates(d, css_path, xpath);
      const target_ref = `fp:${element_fingerprint}`;

      const payload = {
        kind,
        browser_time_ms: nowMs(),
        frame_url,
        is_iframe: (window.self !== window.top),

        element_fingerprint,
        target_ref,

        element_id: eid.element_id,
        element_id_tier: eid.element_id_tier,
        origin: eid.origin,
        selector_candidates,

        ...d,
        ...v,
        text_snippet: textSnippet(el),
        css_path,
        xpath,

        ...b,
        viewport_metrics: vm,

        ...extra
      };

      // Stable dedupe with time window
      const now = payload.browser_time_ms || Date.now();
      const sKey = stableValueKey(kind, payload);
      const prev = lastByKey.get(sKey);
      if (prev && (now - prev) < DEDUPE_WINDOW_MS) return;
      lastByKey.set(sKey, now);

      window.logUiEvent(payload);
    } catch (e) {}
  }

  function pickTarget(e) {
    const t = e.target;
    if (!t) return null;
    return (
      t.closest?.("button,a,input,textarea,select,[role='button'],[role='link'],[role='checkbox'],[role='radio'],[role='option'],[contenteditable='true']") ||
      t
    );
  }

  document.addEventListener("click", (e) => {
    const el = pickTarget(e);
    if (!el) return;
    setTimeout(() => emit("click", el), 0);
  }, true);

  document.addEventListener("change", (e) => {
    const el = pickTarget(e);
    if (!el) return;

    // Optional: reduce noise for checkbox/radio (Google Forms is super chatty)
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.getAttribute && (el.getAttribute("type") || "").toLowerCase()) || "";
    const role = (el.getAttribute && (el.getAttribute("role") || "").toLowerCase()) || "";
    if (tag === "input" && (type === "checkbox" || type === "radio")) return;
    if (role === "checkbox" || role === "radio") return;

    emit("change", el);
  }, true);

  document.addEventListener("input", (e) => {
    if (!SHOULD_LOG_INPUT) return;
    const el = pickTarget(e);
    if (!el) return;
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "input") {
      const type = (el.getAttribute("type") || "").toLowerCase();
      if (type === "checkbox" || type === "radio") return;
    }
    emit("input", el);
  }, true);

  document.addEventListener("submit", (e) => {
    const el = e.target;
    if (!el) return;
    emit("submit", el);
  }, true);

  document.addEventListener("focusin", (e) => {
    const el = pickTarget(e);
    if (!el) return;
    emit("focusin", el);
  }, true);

  document.addEventListener("focusout", (e) => {
    const el = pickTarget(e);
    if (!el) return;
    emit("focusout", el);
  }, true);

  const SPECIAL = new Set([
    "Enter","Escape","Tab",
    "Backspace","Delete",
    "ArrowUp","ArrowDown","ArrowLeft","ArrowRight",
    "Home","End","PageUp","PageDown"
  ]);

  document.addEventListener("keydown", (e) => {
    const k = e.key || "";
    if (!SPECIAL.has(k)) return;
    const el = pickTarget(e);
    if (!el) return;
    emit("keydown", el, {
      key: k,
      ctrl: !!e.ctrlKey,
      meta: !!e.metaKey,
      shift: !!e.shiftKey,
      alt: !!e.altKey
    });
  }, true);

  window.__FP_DOM_LOGGER_READY__ = true;
})();
"""