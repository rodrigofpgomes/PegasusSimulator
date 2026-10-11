"""
| File: training_export.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause.
| Description: Export TensorBoard scalar histories and effective training metadata.
|
| The trainer writes event files during training. This module flushes the
| writer, exports every scalar event to a separate CSV, and records a manifest
| that maps TensorBoard tags to filenames. Repeated steps from resumed runs
| remain in event order, together with their original wall times.
"""

import csv
import hashlib
import json
import logging
import re
from pathlib import Path


def export_scalars(log_dir, output_folder=None, status="manual", accumulator_factory=None):
    """Export all scalar events from one TensorBoard run.

    Args:
        log_dir: Exact experiment directory, not the parent containing multiple runs.
        output_folder: Optional output directory; defaults to run/download_csvs.
        status: completed, interrupted, failed or manual.
        accumulator_factory: Optional EventAccumulator factory for offline tests.

    Returns:
        The manifest containing run status, CSV columns and exported tags.
    """
    # Read the exact run directory, including events written by resumed training
    root = Path(log_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if accumulator_factory is None:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accumulator_factory = EventAccumulator
    accumulator = accumulator_factory(str(root), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    accumulator.Reload()

    # Keep the CSVs and manifest together with the run unless a destination is supplied
    out = Path(output_folder).resolve() if output_folder else root / "download_csvs"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": status, "run_dir": str(root), "columns": ["wall_time", "step", "value"], "scalars": []}
    tags = sorted(accumulator.Tags().get("scalars", []))
    for tag in tags:
        # Keep a readable tag and add a digest because distinct tags can sanitize identically
        stem = re.sub(r'[^\w .-]', '_', tag).strip(' .')[:120] or "scalar"
        name = stem + "--" + hashlib.sha256(tag.encode()).hexdigest()[:12] + ".csv"
        events = accumulator.Scalars(tag)
        target = out / name
        temp = target.with_suffix(".tmp")

        # Replace each finished CSV atomically so an interrupted export has no partial file
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(manifest["columns"])
            writer.writerows((e.wall_time, e.step, e.value) for e in events)
        temp.replace(target)
        manifest["scalars"].append({"tag": tag, "file": name, "events": len(events)})
    if not tags:
        manifest["warning"] = "No scalar events found; no metrics fabricated"
        logging.warning(manifest["warning"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_and_export(trainer, agent, run_dir, metadata=None):
    """Train an agent and export the events produced before completion or interruption.

    Training exceptions retain their original traceback. Export errors are
    logged separately and written to export_error.json when possible.

    Args:
        trainer: skrl trainer exposing train().
        agent: Agent whose TensorBoard writer may need flushing.
        run_dir: Directory for metadata, event files and exported CSVs.
        metadata: Effective configuration and run identifiers to preserve.
    """
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Save configuration before training, including runs that terminate early
    (root / "run_metadata.json").write_text(json.dumps(metadata or {}, indent=2, default=str))
    status = "failed"
    try:
        trainer.train()
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        try:
            # EventAccumulator must read the final events already flushed by skrl
            writer = getattr(agent, "writer", None)
            if writer is not None:
                writer.flush()
            export_scalars(root, status=status)
        except Exception as exc:
            logging.exception("Training CSV export failed in %s", root)
            try:
                (root / "export_error.json").write_text(json.dumps({"status": status, "error": str(exc)}, indent=2))
            except OSError:
                logging.exception("Could not persist export failure metadata")


def snapshot_config(config):
    """Convert an effective configuration into JSON-compatible metadata.

    Reading attributes also captures plain class defaults, such as surface
    limits, which dataclass field enumeration would omit.
    """
    def encode(value):
        """Encode nested mappings, sequences, tensors and configuration objects."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if hasattr(value, 'detach'):
            return value.detach().cpu().tolist()
        if hasattr(value, '__dict__'):
            return snapshot_config(value)
        return str(value)
    return {name: encode(getattr(config, name)) for name in dir(config)
            if not name.startswith('_') and not callable(getattr(config, name))}
