"""
Domain models for the computer use workflow. These models are used across the services defined in the services directory
to represent the core data structures and business logic.
"""

from dataclasses import dataclass, field
from pathlib import Path


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def center(self) -> Point:
        return Point(
            x=round((self.x1 + self.x2) / 2),
            y=round((self.y1 + self.y2) / 2),
        )


@dataclass(frozen=True)
class LocateTarget:
    bounding_box: BoundingBox
    click_point: Point


@dataclass
class StepArtifact:
    step_name: str
    screenshot_path: Path | None = None
    coordinates: Point | None = None
    bounding_box: BoundingBox | None = None
    screen_coordinates: Point | None = None
    verification: str | None = None
    content: str | None = None
    debug_image_path: Path | None = None
    artifact_path: Path | None = None


@dataclass(frozen=True)
class StepAction:
    kind: str
    app_name: str | None = None
    text_template: str | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class LocateSpec:
    prompt_template: str
    screenshot_ref_step_name: str | None = None


@dataclass(frozen=True)
class VerificationSpec:
    prompt_template: str
    required_tokens: tuple[str, ...] = ("yes",)


@dataclass(frozen=True)
class DescriptionSpec:
    prompt_template: str
    output_key: str | None = None


@dataclass(frozen=True)
class WorkflowStepDefinition:
    step_name: str
    before_actions: list[StepAction] = field(default_factory=list)
    capture_screenshot: bool = False
    locate: LocateSpec | None = None
    action: StepAction | None = None
    after_actions: list[StepAction] = field(default_factory=list)
    verification: VerificationSpec | None = None
    description: DescriptionSpec | None = None
    content_template: str | None = None


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    output_dir: Path
    verify: bool
    context: dict[str, str]
    steps: list[WorkflowStepDefinition]
