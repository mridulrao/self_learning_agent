# computer-use

Deterministic macOS computer-use workflow with screenshot grounding, GUI-only mouse actions, and a local LocateAnything-based region locator.

## What This Project Does

This project automates a small end-to-end desktop task on macOS:

1. Locate and click Safari in the macOS Dock.
2. Take a screenshot of the desktop.
3. Use a LocateAnything model to find the Safari search/address bar.
4. Move the mouse and click that GUI target.
5. Search for a query.
6. Open the first relevant result.
7. Extract stock-related text from the loaded page with a vision model.
8. Locate and click Notes in the macOS Dock.
9. Use the GUI to locate and click the New Note button.
10. Paste the extracted content into the note.

The workflow is deterministic in structure, but uses visual grounding for the clickable UI targets.

## Project Structure

`workflow_examples/computer_use_workflow.py`
Main macOS workflow runner. Handles screenshots, coordinate translation, clicking, verification, artifact generation, and the Safari -> Notes demo flow.

`workflow_examples/macos_input.swift`
Quartz-based macOS helper for screen-geometry lookup and GUI mouse movement/clicking.

`workflow_examples/call_locateanything_rpc.py`
Small client for calling the LocateAnything JSON-RPC endpoint.

`model/locateanything/server.py`
FastAPI server exposing the LocateAnything model over HTTP/JSON-RPC.

`model/locateanything/service.py`
Model loading and inference wrapper for LocateAnything.

`model/locateanything/bootstrap.py`
Downloads model assets ahead of runtime.

`model/locateanything/Dockerfile`
Container image for the LocateAnything model server.

## Requirements

- macOS
- Python 3.11+
- Accessibility permissions for Terminal / your Python runner
- Screen Recording permissions for screenshot capture
- A running LocateAnything endpoint
- A vision model for page verification/content extraction

## Python Dependencies

Dependencies are declared in [pyproject.toml](/Users/mridulrao/Downloads/psuedo_desktop/computer_use/pyproject.toml).

Current direct dependencies:

- `openai`
- `requests`
- `Pillow`

Install with your preferred tool, for example:

```bash
pip install -e .
```

or:

```bash
uv sync
```

## Running the LocateAnything Model Server

Build the container:

```bash
docker build --platform linux/amd64 -f model/locateanything/Dockerfile -t mridulrao/locateanything:v0.0.2 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  mridulrao/locateanything:v0.0.2
```

The workflow expects a LocateAnything JSON-RPC endpoint. By default the helper client uses the endpoint configured in `workflow_examples/call_locateanything_rpc.py`.

## Vision Model Configuration

The workflow needs a second vision model for:

- screenshot verification
- webpage content extraction

You can configure either:

### OpenAI-compatible Responses API

Set:

- `OPENAI_API_KEY`
- `CONTENT_VISION_MODEL`
- `VERIFY_VISION_MODEL`
- optional `OPENAI_BASE_URL`

### Custom HTTP endpoints

Set:

- `VISION_CONTENT_URL`
- `VISION_VERIFY_URL`
- optional `VISION_API_KEY`

## Running the Demo Workflow

Run the full Safari -> Notes flow:

```bash
python workflow_examples/computer_use_workflow.py --verify
```

Useful flags:

- `--query`
  Text to type into Safari.
- `--safari-app-prompt`
  Prompt used to visually locate the Safari app icon.
- `--search-box-prompt`
  Prompt used to visually locate the Safari address/search bar.
- `--first-link-prompt`
  Prompt used to visually locate the first result link.
- `--notes-app-prompt`
  Prompt used to visually locate the Notes app icon.
- `--new-note-prompt`
  Prompt used to visually locate the Notes New Note button.
- `--safari-fullscreen-prompt`
  Prompt used to visually locate the Safari fullscreen button.
- `--output-dir`
  Directory for screenshots and JSON artifacts.
- `--fullscreen`
  Click the Safari fullscreen button after launch.
- `--verify`
  Run vision-based verification after major steps.

Example:

```bash
python workflow_examples/computer_use_workflow.py \
  --query "Top stocks in the US" \
  --verify
```

## Demo Flow Covered Today

The implemented demo flow is:

1. Screenshot the current screen.
2. Locate the Safari Dock icon and click it.
3. Optionally locate the Safari fullscreen button and click it.
4. Locate the Safari search/address bar from a screenshot.
5. Translate screenshot coordinates into macOS screen coordinates.
6. Move the real cursor and click the GUI target.
7. Search for the configured query.
8. Wait for Google results.
9. Screenshot the results page.
10. Locate the first relevant result link.
11. Move and click that GUI target.
12. Screenshot the loaded page.
13. Extract visible stock-related content.
14. Locate the Notes Dock icon and click it.
15. Screenshot the Notes window.
16. Locate the New Note toolbar button.
17. Move the cursor and click that GUI target.
18. Paste the extracted content into a note.
19. Save step artifacts for debugging.

## Coordinate Translation and Clicking

The workflow uses:

- screenshot pixel dimensions from the captured PNG
- live macOS display geometry from `workflow_examples/macos_input.swift`

It translates screenshot-space points into screen-space points before clicking.

The click helper now works by:

1. Warping the real cursor to the translated target point.
2. Emitting a short move event.
3. Sending left mouse down/up events with small delays.

This turned out to be important for Notes toolbar interactions, where synthetic clicks without real cursor movement were not always accepted reliably.

## Debug Artifacts

Artifacts are written under [workflow_examples/artifacts](/Users/mridulrao/Downloads/psuedo_desktop/computer_use/workflow_examples/artifacts) by default.

For each step, the workflow writes a JSON artifact. For click-target steps it also writes a debug screenshot:

- `step2_search_box_debug.png`
- `step4_first_link_debug.png`
- `step7_new_note_debug.png`

These debug images show:

- a red rectangle for the located bounding box
- a blue crosshair/dot for the actual click point

This makes it easy to distinguish:

- bad visual localization
- bad coordinate translation
- bad click delivery

## Common Troubleshooting

### Docker build fails in `bootstrap.py`

The image build should run the bootstrap script from the packaged model path and include `env_config.py`. If you see path-related `IndexError` failures during build, rebuild with the latest Dockerfile and bootstrap changes.

### `switch must be exhaustive` in `macos_input.swift`

This means the Swift helper has an enum case that was not handled in the action switch. Re-run with the updated script in this repo.

### Cursor lands correctly but the button does not activate

This is usually not a coordinate-mapping problem. It is often caused by how the click event is delivered to macOS. Use the latest `workflow_examples/macos_input.swift`, which warps the real cursor before clicking.

### No screenshots captured

Grant Screen Recording permission to the terminal or app that runs the workflow.

### Mouse/keyboard actions do nothing

Grant Accessibility permission to the terminal or app that runs the workflow.

## Output

At the end of a successful run, the workflow prints a JSON summary containing:

- `step_name`
- `screenshot_path`
- `debug_image_path`
- `verification`
- `content`
- `artifact_path`

## Notes

- This workflow is macOS-specific.
- The current demo flow is intentionally narrow and deterministic.
- The LocateAnything step is used only for grounded GUI targeting, not general planning.
