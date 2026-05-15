"""Backfill explicit model_metadata.json files for old PvZRL run folders."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pvzrl_model_metadata import (
    MODEL_METADATA_FILENAME,
    infer_fixed_metadata_from_legacy_config,
)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    candidates = set()
    for config_name in ("resolved_config.json", "config.json"):
        for path in root.rglob(config_name):
            candidates.add(path.parent)
    return sorted(candidates)


def backfill_one(run_dir: Path, *, dry_run: bool, force: bool) -> Dict[str, Any]:
    metadata_path = run_dir / MODEL_METADATA_FILENAME
    if metadata_path.exists() and not force:
        return {"run_dir": str(run_dir), "status": "skipped_exists", "metadata_path": str(metadata_path)}
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        config_path = run_dir / "config.json"
    config = load_json(config_path)
    if config is None:
        return {"run_dir": str(run_dir), "status": "blocked", "blocked_reason": "config_missing_or_invalid"}
    metadata, blocked_reason = infer_fixed_metadata_from_legacy_config(config)
    if metadata is None:
        return {"run_dir": str(run_dir), "status": "blocked", "blocked_reason": blocked_reason}
    metadata["metadata_source"] = f"backfilled_from:{config_path.name}"
    metadata["created_at"] = metadata.get("created_at") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    metadata["config_path"] = str(config_path)
    if dry_run:
        return {"run_dir": str(run_dir), "status": "would_write", "metadata_path": str(metadata_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "status": "written", "metadata_path": str(metadata_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill PvZRL model_metadata.json files for legacy fixed runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = [
        backfill_one(run_dir, dry_run=bool(args.dry_run), force=bool(args.force))
        for run_dir in run_dirs(args.runs_dir)
    ]
    print(json.dumps({"results": results}, indent=2))
    blocked = sum(1 for result in results if result.get("status") == "blocked")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
