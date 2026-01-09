# Agent Architecture Overview

This repository implements **workflow agents** that can **record**, **compile**, and **replay** user workflows across different environments in a robust and scalable way.

The core idea is **not** to reason directly over UI elements while recording, but instead to:
1. Capture **environment traces**
2. Convert them into a **canonical action graph**
3. Replay them using a **finite set of primitives**

This makes the system stable, debuggable, and extensible across browsers and desktop environments.

---

## Supported Agents

- **Browser Agent (Playwright-based)**
  - Deterministic execution
  - DOM + navigation driven
- **Desktop Agent (macOS)**
  - UI-driven execution
  - OCR / vision-based grounding at runtime

---

## High-Level Pipeline
User Interaction
↓
Raw Event Traces
↓
Compiler (Rules + Anchors)
↓
Canonical Workflow Steps
↓
Agent Execution (Ground → Act → Verify)

---
## 1. Recording: Capturing Raw Traces (activity_recorder.py)

The recorder captures **high-recall environment signals**.

### Browser Signals
- Navigation events & frame URLs
- Focus changes
- Click / type / select / submit actions
- Final URL after navigation bursts

### Desktop Signals
- Active application & bundle ID
- Window titles
- Mouse clicks & keypresses
- Screenshots (pre-click + post-click buffers)

> The recorder’s job is **not correctness**, but **completeness**. Noise is expected and handled later.

---

## 2. Compilation: Logs → Canonical Steps (derive_workflow.py)

Raw logs are noisy and environment-specific. The compiler normalizes them into **stable workflow steps**.

### Compilation Stages

1. **Filtering & Ordering**
   - Normalize timestamps
   - Remove irrelevant system noise

2. **Deduplication & Burst Merging**
   - Merge navigation bursts
   - Collapse repeated focus events

3. **Anchor Selection**
   Each step is anchored to the event that best represents user intent.

   Typical anchor priority (browser):
    submit > click > enter > select > type

4. **Step Construction**
    Each canonical step includes:
    - `step_id`
    - `t_anchor_ms`, `t_start_ms`, `t_end_ms`
    - `anchor_type` + payload
    - `final_url` and `final_url_canon`
    - Evidence event IDs (for debugging)

    The output is a normalized workflow file such as: workflow_steps.v1.json


---

## 3. Canonical Action Space (Primitives) (orchestrator.py)

All execution happens through a **finite set of primitives**.  
Agents do not invent new actions at runtime.

### Browser Primitives
- `browser_launch`
- `navigate`
- `browser_click`
- `browser_type`
- `browser_select`
- `browser_submit`

**Key property:** deterministic replay using Playwright.

---

### Desktop Primitives
- `desktop_launch`
- `desktop_click`
- `desktop_type`
- `desktop_hotkey`
- `desktop_observe` (OCR / vision grounding)

**Key property:** perception-driven replay.  
Recorded coordinates are *not trusted*.

---

## 4. Execution Model: Ground → Act → Verify (desktop_agent.py / browser_agent.py)

Each agent runs a loop similar to:
for step in workflow:
ground(step)
act(step)
verify(step)
retry_if_needed()


### Grounding
- **Browser:** URLs, selectors, page state
- **Desktop:** OCR / vision locate target using text hints or labels

### Action
- Click, type, navigate, submit, etc.

### Verification
- Browser: URL canonicalization, load completion
- Desktop: optional observe-after-step or visual checks

### Retry & Backoff
- Bounded retries per step
- Structured failures with evidence artifacts

---

## Validation Strategy

### Browser Validation
- Canonicalized URLs (tracking params removed)
- Navigation completion checks
- Optional DOM assertions

### Desktop Validation
- Screenshot-based observe
- OCR/vision returns candidate bounding boxes
- Agent clicks **center of matched box**
- Optional observe-between-steps for stability

---

## Why This Architecture Works

- **Finite primitive set** → predictable execution
- **Compilation isolates noise** → agents stay simple
- **Runtime grounding** → robust to UI changes
- **Evidence-first debugging** → every failure is explainable

This architecture scales across workflows because **rules stay constant** while **feature signals adapt per environment**.

---

## Workflow JSON: What It Represents

A compiled workflow contains ordered steps such as:

- **Browser**
  - launch
  - navigate (with `final_url_canon`)
  - click / type / submit

- **Desktop**
  - launch app (fullscreen)
  - click / type via OCR grounding

Each step links back to:
- Evidence event IDs
- Screenshots
- Validation metadata

This makes workflows **replayable, inspectable, and debuggable**.

---

## Summary

This system treats workflows as **compiled programs**, not UI scripts.

- Record everything
- Compile into canonical steps
- Execute using grounded primitives
- Validate aggressively
- Debug with evidence

This design allows workflows to scale across browsers, desktop apps, and future environments without rewriting the planner or agents.





