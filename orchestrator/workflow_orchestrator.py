
import json
import logging
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from domain import StepAction, StepArtifact, WorkflowDefinition, WorkflowError
from services import (
    DescribeRequest,
    LocateRequest,
    LocatorService,
    ScreenshotRequest,
    UIAutomationService,
    VerificationRequest,
    VerificationService,
)

LOGGER = logging.getLogger(__name__)


class WorkflowOrchestrator:
    def __init__(
        self,
        ui_service: UIAutomationService,
        locator_service: LocatorService,
        verification_service: VerificationService,
    ) -> None:
        self.ui_service = ui_service
        self.locator_service = locator_service
        self.verification_service = verification_service

    def run(self, workflow: WorkflowDefinition) -> list[StepArtifact]:
        LOGGER.info("Starting workflow: %s", workflow.name)
        LOGGER.info("Workflow screenshots directory: %s", workflow.output_dir)
        artifacts: list[StepArtifact] = []
        context = dict(workflow.context)
        artifacts_by_name: dict[str, StepArtifact] = {}

        for step_definition in workflow.steps:
            LOGGER.info("Running workflow step: %s", step_definition.step_name)
            step = StepArtifact(step_name=step_definition.step_name)

            for action in step_definition.before_actions:
                self._execute_action(action, step, context)

            if step_definition.capture_screenshot:
                step.screenshot_path = self.ui_service.take_screenshot(
                    ScreenshotRequest(step_name=step.step_name, output_dir=workflow.output_dir)
                )

            if step_definition.locate:
                screenshot_path = step.screenshot_path
                if step_definition.locate.screenshot_ref_step_name:
                    referenced_step = artifacts_by_name[step_definition.locate.screenshot_ref_step_name]
                    screenshot_path = referenced_step.screenshot_path
                if screenshot_path is None:
                    raise WorkflowError(f"No screenshot available for locate step {step.step_name}")

                prompt = self._render_template(step_definition.locate.prompt_template, context)
                target = self.locator_service.locate(
                    LocateRequest(screenshot_path=screenshot_path, prompt=prompt)
                )
                step.screenshot_path = screenshot_path
                step.coordinates = target.click_point
                step.bounding_box = target.bounding_box
                step.screen_coordinates = self.ui_service.translate_coordinates_for_screen(
                    screenshot_path, target.click_point
                )

            if step_definition.action:
                self._execute_action(step_definition.action, step, context)

            for action in step_definition.after_actions:
                self._execute_action(action, step, context)

            if step_definition.content_template:
                step.content = self._render_template(step_definition.content_template, context)

            if step_definition.description:
                if step.screenshot_path is None:
                    raise WorkflowError(f"No screenshot available for description step {step.step_name}")
                description = self.verification_service.describe(
                    DescribeRequest(
                        screenshot_path=step.screenshot_path,
                        prompt=self._render_template(step_definition.description.prompt_template, context),
                    )
                )
                step.content = description
                if step_definition.description.output_key:
                    context[step_definition.description.output_key] = description

            if step_definition.verification:
                self._verify_step(workflow, step, step_definition.verification.prompt_template, context)

            self._add_step_artifact(artifacts, step, workflow.output_dir)
            artifacts_by_name[step.step_name] = step

        LOGGER.info("Workflow completed successfully.")
        return artifacts

    def _verify_step(self, workflow: WorkflowDefinition, step: StepArtifact, prompt_template: str, context: dict[str, str]) -> None:
        if not workflow.verify:
            return
        LOGGER.info("Running verification for %s", step.step_name)
        result = self.verification_service.verify(
            VerificationRequest(
                screenshot_path=step.screenshot_path,
                prompt=self._render_template(prompt_template, context),
            )
        )
        step.verification = result.message
        if not result.passed:
            raise WorkflowError(f"Verification failed for {step.step_name}: {result.message}")
        LOGGER.info("Verification passed for %s: %s", step.step_name, result.message)

    def _execute_action(self, action: StepAction, step: StepArtifact, context: dict[str, str]) -> None:
        if action.kind == "press_return":
            self.ui_service.press_return()
            return

        if action.kind == "paste_text":
            if action.text_template is None:
                raise WorkflowError("paste_text action requires text_template")
            self.ui_service.paste_text(self._render_template(action.text_template, context))
            return

        if action.kind == "wait":
            if action.seconds is None:
                raise WorkflowError("wait action requires seconds")
            self.ui_service.wait_for(action.seconds)
            return

        if action.kind == "click_located_target":
            if step.screen_coordinates is None:
                raise WorkflowError(f"No located target available to click for step {step.step_name}")
            self.ui_service.move_and_click(step.screen_coordinates)
            return

        raise WorkflowError(f"Unsupported workflow action kind: {action.kind}")

    def _render_template(self, template: str, context: dict[str, str]) -> str:
        try:
            return template.format(**context)
        except KeyError as error:
            raise WorkflowError(f"Missing workflow context key for template rendering: {error}") from error

    def _annotate_click_target(self, step: StepArtifact, output_dir: Path) -> Path | None:
        if not step.screenshot_path or not step.bounding_box or not step.coordinates:
            return None

        debug_image_path = output_dir / f"{step.step_name}_debug.png"
        with Image.open(step.screenshot_path) as image:
            annotated = image.convert("RGBA")
            overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            box = step.bounding_box
            click = step.coordinates
            crosshair_radius = max(10, round(min(annotated.size) * 0.012))
            marker_radius = max(6, round(min(annotated.size) * 0.008))

            draw.rectangle(
                [(box.x1, box.y1), (box.x2, box.y2)],
                outline=(255, 64, 64, 255),
                width=5,
                fill=(255, 64, 64, 35),
            )
            draw.ellipse(
                [
                    (click.x - marker_radius, click.y - marker_radius),
                    (click.x + marker_radius, click.y + marker_radius),
                ],
                fill=(0, 200, 255, 255),
                outline=(255, 255, 255, 255),
                width=2,
            )
            draw.line(
                [(click.x - crosshair_radius, click.y), (click.x + crosshair_radius, click.y)],
                fill=(0, 200, 255, 255),
                width=3,
            )
            draw.line(
                [(click.x, click.y - crosshair_radius), (click.x, click.y + crosshair_radius)],
                fill=(0, 200, 255, 255),
                width=3,
            )

            annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
            annotated.save(debug_image_path)

        step.debug_image_path = debug_image_path
        LOGGER.info("Wrote debug screenshot with click target: %s", debug_image_path)
        return debug_image_path

    def _persist_step_artifact(self, step: StepArtifact, output_dir: Path) -> Path:
        artifact_path = output_dir / f"{step.step_name}.json"
        self._annotate_click_target(step, output_dir)
        payload = {
            "step_name": step.step_name,
            "screenshot_path": str(step.screenshot_path) if step.screenshot_path else None,
            "debug_image_path": str(step.debug_image_path) if step.debug_image_path else None,
            "verification": step.verification,
            "content": step.content,
            "coordinates": asdict(step.coordinates) if step.coordinates else None,
            "bounding_box": asdict(step.bounding_box) if step.bounding_box else None,
            "screen_coordinates": asdict(step.screen_coordinates) if step.screen_coordinates else None,
        }
        artifact_path.write_text(json.dumps(payload, indent=2))
        step.artifact_path = artifact_path
        LOGGER.info("Wrote step artifact: %s", artifact_path)
        return artifact_path

    def _add_step_artifact(self, artifacts: list[StepArtifact], step: StepArtifact, output_dir: Path) -> None:
        self._persist_step_artifact(step, output_dir)
        artifacts.append(step)
