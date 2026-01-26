 # Browser + Desktop Agent (Record → Extract → Execute)

 Design Choice: https://drive.google.com/file/d/1NHTc66iUBkGqOenlF0Nk9OnjzLn82kbS/view?usp=sharing
 
 This module supports a simple workflow lifecycle:
 
 1) **Record** a real user session (browser + desktop) into raw event logs and artifacts.
 2) **Extract** a reusable **primitive** workflow from those events.
 3) **Execute** the extracted workflow using:
    - **Browser Agent** grounded by **Playwright + DOM**
    - **Desktop Agent** grounded by **OCR** (screen text + bounding boxes)
 
 The core design choice is to keep the action space *primitive-first* so that **any workflow can be recorded and mimicked**, rather than relying on app-specific macros.
 
 
 ## Why primitives?
 
 This repo intentionally models workflows as small, generic actions (click, type, select, navigate, hotkey, etc.).
 
 Benefits:
 
 - **Record anything**: you don’t need a custom “integration” per app or website.
 - **Replay robustly**: each primitive can be retried, validated, and grounded (DOM/OCR) at runtime.
 - **Composable**: extracted steps can be reordered, merged, or used as building blocks for higher-level automation.
 
 
 ## Grounding sources
 
 - **Browser grounding**: Playwright provides DOM access for reliable targeting (selectors, element identity, page state).
 - **Desktop grounding**: OCR provides “what’s on screen” so the agent can click/validate by visible text and UI elements.
 
 
 # 1) Record a user workflow
 
 The recorder captures:
 
 - **`events.jsonl`**: structured timeline of OS + browser events
 - **`screen_recording.avi`**: screen video (debugging / audit)
 - **`browser_trace.zip`**: Playwright trace (browser debugging)
 - **`web_screenshots/`**: event-driven browser screenshots
 - **`desktop_screenshots/`**: event-driven desktop screenshots (pre/post click evidence)
 - **`voice_notes/`** (optional): voice note audio + transcripts (used later by some validators)
 
 Run:
 
 ```bash
 python browser_desktop_agent/record_user_workflow.py
 ```
 
 Recording is controlled via explicit markers typed into the terminal:
 
 - Start: `###WORKFLOW_RECORDING_START###`
 - Stop: `###WORKFLOW_RECORDING_STOP###`
 
 These markers are logged into `events.jsonl` as `workflow_control` events and are intended to be filtered out during extraction.
 
 Output is written under:
 
 - `recordings/<session_id>/...`
 
 
 # 2) Extract a reusable workflow
 
 Extraction converts the raw `events.jsonl` timeline into a **normalized step list** that can be executed by primitive agents.
 
 Run:
 
 ```bash
 python browser_desktop_agent/extract_user_workflow.py \
   --session-dir recordings/<session_id> \
   --events events.jsonl
 ```
 
 Optional (recommended for desktop robustness): provide an OCR server so desktop clicks can be grounded to visible text.
 
 ```bash
 python browser_desktop_agent/extract_user_workflow.py \
   --session-dir recordings/<session_id> \
   --desktop-ocr-url "https://<your-ocr-host>" \
   --desktop-ocr-min-conf 0.55 \
   --desktop-ocr-near-radius 60
 ```
 
 The extractor writes:
 
 - `recordings/<session_id>/workflow_steps.v1.json`
 
 This file contains an ordered list of steps with `agent` set to either `browser` or `desktop`.
 
 What extraction does at a high level:
 
 - **Browser**
   - Derives UI steps (click/type/select/submit) from DOM events.
   - Clusters noisy navigation chains into stable “nav steps”.
 - **Desktop**
   - Merges low-level OS events into higher-level primitives (click, type text, hotkey, scroll).
   - Optionally enriches click steps with OCR-derived targets (text + bbox) using evidence screenshots.
 
 
 # 3) Execute the workflow
 
 You can execute the same extracted `workflow_steps.v1.json` using either agent (or both, depending on which steps you want to run).
 
 ## 3A) Execute browser steps (Playwright / DOM)
 
 Run:
 
 ```bash
 python browser_desktop_agent/browser_agent.py \
   --workflow recordings/<session_id>/workflow_steps.v1.json
 ```
 
 Common options:
 
 - `--headful`: run with a visible browser window
 - `--channel chrome`: use system Chrome (often more stable on macOS)
 - `--observe-between`: print lightweight state between steps
 
 The browser agent is **DOM-grounded** and resolves targets using recorded selector candidates with safe fallbacks.
 
 
 ## 3B) Execute desktop steps (OCR)
 
 Run:
 
 ```bash
 python browser_desktop_agent/desktop_agent.py \
   --workflow recordings/<session_id>/workflow_steps.v1.json \
   --ocr-url "https://<your-ocr-host>"
 ```
 
 Notes:
 
 - The desktop agent is **OCR-grounded**. It uses OCR endpoints like `/observe` and `/click_target` to find UI elements by visible text and validate outcomes.
 - On macOS you must grant Accessibility / Screen Recording permissions for control + screenshots.
 - Coordinate spaces can differ on Retina displays; the agent clicks in `pyautogui` space and scales from recorded evidence when needed.
 
 
 ## 3C) Optional: orchestrated execution (validated high-level workflow)
 
 In addition to replaying primitive steps directly, there is also an orchestrated runner that validates/normalizes higher-level workflows before execution:
 
 ```bash
 python browser_desktop_agent/execute_agent_workflow.py --workflow <workflow.json>
 ```
 
 This path is useful when you have an LLM-produced workflow that needs normalization (for example: label cleanup for Google Forms) before running.
