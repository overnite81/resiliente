#!/usr/bin/env python3
"""Download MIT-BIH records into a flat directory for ``--data-dir``.

The WFDB loader used by the experiments expects ``<data-dir>/<record>.hea``,
``.dat`` and ``.atr``. This helper downloads the records once and verifies the
flat layout, so later runs avoid the slow PhysioNet fetch (the pilot reported
~550 s of data loading on every run).

    python -m experiments.download_mitdb \
        --records 100 101 103 105 --data-dir data/mitdb
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import wfdb

DEFAULT_RECORDS = ("100", "101", "103", "105")
EXTENSIONS = (".hea", ".dat", ".atr")


def flatten(data_dir: Path) -> None:
    """Move any files found in nested subdirectories up into ``data_dir``."""
    for path in list(data_dir.rglob("*")):
        if path.is_file() and path.parent != data_dir:
            target = data_dir / path.name
            if not target.exists():
                shutil.move(str(path), str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", default=list(DEFAULT_RECORDS))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--db", default="mitdb", help="PhysioNet database name.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        record for record in args.records
        if args.overwrite or not (args.data_dir / f"{record}.hea").exists()
    ]
    if pending:
        print(f"Baixando {len(pending)} registros para {args.data_dir} ...", flush=True)
        wfdb.dl_database(
            args.db,
            dl_dir=str(args.data_dir),
            records=list(pending),
            keep_subdirs=False,
            overwrite=args.overwrite,
        )
    else:
        print(f"Todos os registros ja estao em {args.data_dir}; nada a baixar.")

    flatten(args.data_dir)

    missing = [
        record for record in args.records
        if not all((args.data_dir / f"{record}{ext}").exists() for ext in EXTENSIONS)
    ]
    if missing:
        raise SystemExit(
            "Layout incompleto para: " + ", ".join(missing)
            + f"\nConfira {args.data_dir} manualmente."
        )

    print(f"OK: {len(args.records)} registros prontos em {args.data_dir.resolve()}")
    for record in args.records:
        size = (args.data_dir / f"{record}.dat").stat().st_size
        print(f"  {record}: hea/dat/atr presentes (dat={size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())