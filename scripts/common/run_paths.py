"""Central path resolution for unified run directories.

Every script that reads or writes run artifacts should import from here.
No script should construct result paths on its own.

Run directory layout:
    runs/{env_name}/{run_name}/
        config.yaml
        model.zip
        metadata.json
        tensorboard/
        checkpoints/
        trajectories/
        slurm/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


RUNS_ROOT = Path("runs")


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for all artifacts of a single run."""

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def model(self) -> Path:
        return self.root / "model.zip"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def tensorboard_dir(self) -> Path:
        return self.root / "tensorboard"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def trajectories_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def slurm_dir(self) -> Path:
        return self.root / "slurm"

    @property
    def env_name(self) -> str:
        return self.root.parent.name

    @property
    def run_name(self) -> str:
        return self.root.name

    @property
    def run_id(self) -> str:
        """Canonical '{env_name}/{run_name}' identifier."""
        return f"{self.env_name}/{self.run_name}"

    @classmethod
    def from_run_id(cls, env_name: str, run_name: str) -> RunPaths:
        return cls(root=RUNS_ROOT / env_name / run_name)

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> RunPaths:
        return cls(root=run_dir)

    def trajectory_subdir(self, name: str) -> Path:
        return self.trajectories_dir / name

    def latest_checkpoint(self) -> Path | None:
        if not self.checkpoints_dir.exists():
            return None

        def _extract_steps(p: Path) -> int:
            # checkpoint_50000_steps.zip -> 50000
            for part in p.stem.split("_"):
                if part.isdigit():
                    return int(part)
            return 0

        checkpoints = sorted(
            self.checkpoints_dir.glob("*.zip"),
            key=_extract_steps,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def create_dirs(self) -> None:
        for d in (
            self.root,
            self.tensorboard_dir,
            self.checkpoints_dir,
            self.trajectories_dir,
            self.slurm_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.root.exists()


def resolve_run(run_ref: str) -> RunPaths:
    """Resolve a flexible run reference to a RunPaths object.

    Accepted formats:
        - Full/relative path:  runs/PopulationWrapper-v0/RealMap_base_2026-...
        - Canonical ID:        PopulationWrapper-v0/RealMap_base_2026-...
        - Bare run name:       RealMap_base_2026-...  (searches all env dirs)
    """
    ref = Path(run_ref).expanduser()

    # Strip known suffixes (.yaml, .zip)
    if ref.suffix in (".yaml", ".zip"):
        ref = ref.with_suffix("")

    # If it's an absolute path or starts with the runs root, use directly
    if ref.is_absolute():
        return RunPaths.from_run_dir(ref)

    # Try as path relative to RUNS_ROOT
    candidate = RUNS_ROOT / ref
    if candidate.exists():
        return RunPaths.from_run_dir(candidate)

    # Try as-is (might be runs/env/name already)
    if ref.exists():
        return RunPaths.from_run_dir(ref.resolve())

    # If it looks like env_name/run_name (has exactly one slash)
    parts = ref.parts
    if len(parts) == 2:
        return RunPaths.from_run_id(parts[0], parts[1])

    # Bare run name — search all env directories
    if len(parts) == 1:
        bare_name = parts[0]
        if RUNS_ROOT.exists():
            for env_dir in sorted(RUNS_ROOT.iterdir()):
                if env_dir.is_dir():
                    candidate = env_dir / bare_name
                    if candidate.exists():
                        return RunPaths.from_run_dir(candidate)
        # Not found yet — cannot determine env_name, raise
        raise FileNotFoundError(
            f"Could not find run '{bare_name}' in any env directory under {RUNS_ROOT}"
        )

    raise ValueError(f"Cannot resolve run reference: {run_ref}")


# ── Metadata helpers ─────────────────────────────────────────


def write_metadata(run_paths: RunPaths, **kwargs) -> None:
    run_paths.metadata.write_text(json.dumps(kwargs, indent=2, default=str))


def read_metadata(run_paths: RunPaths) -> dict:
    if not run_paths.metadata.exists():
        return {}
    return json.loads(run_paths.metadata.read_text())


def update_metadata(run_paths: RunPaths, **kwargs) -> None:
    data = read_metadata(run_paths)
    data.update(kwargs)
    write_metadata(run_paths, **data)
