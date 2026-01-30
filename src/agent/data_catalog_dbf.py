from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Dynamically determine the project root and set paths relative to it
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_FOLDER = PROJECT_ROOT / "data/tasks/task_4/boundaries_community"
OUT_PATH = PROJECT_ROOT / "data/tasks/task_4/boundaries_community/layers.json"


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


pyshp = _try_import("shapefile")  # pip install pyshp


def _read_any_prj_wkt(folder: Path) -> str:
    prjs = sorted(folder.glob("*.prj"))
    if not prjs:
        return ""
    try:
        return prjs[0].read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _pick_dbf(folder: Path) -> Optional[Path]:
    dbfs = sorted(folder.glob("*.dbf"))
    if not dbfs:
        return None
    return dbfs[0]


def read_fields_from_dbf(dbf_path: Path) -> List[Dict[str, Any]]:
    if pyshp is None:
        raise RuntimeError("pyshp is not installed. Run: pip install pyshp")

    # Read DBF only
    r = pyshp.Reader(dbf=str(dbf_path))
    fields = []
    # r.fields: first entry is DeletionFlag
    for f in r.fields[1:]:
        # (name, fieldType, size, decimal)
        fields.append({
            "name": str(f[0]),
            "type": str(f[1]),
            "size": int(f[2]),
            "decimal": int(f[3]),
        })
    return fields


def build_layers_json() -> Dict[str, Any]:
    if not LAYER_FOLDER.exists() or not LAYER_FOLDER.is_dir():
        raise FileNotFoundError(f"Folder not found: {LAYER_FOLDER}")

    dbf = _pick_dbf(LAYER_FOLDER)
    if dbf is None:
        raise FileNotFoundError(f"No .dbf found in: {LAYER_FOLDER}")

    fields = read_fields_from_dbf(dbf)
    crs = _read_any_prj_wkt(LAYER_FOLDER)

    catalog = {
        "task_dir": str(LAYER_FOLDER.parent),
        "layers": [
            {
                "name": LAYER_FOLDER.name,
                "path": str(dbf),
                "geometry_type": "",
                "fields": fields,
                "crs": crs
            }
        ]
    }
    return catalog


def write_layers_json(obj: Dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    obj = build_layers_json()
    write_layers_json(obj)
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
