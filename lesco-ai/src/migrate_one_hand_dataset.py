"""Migrate selected one-hand LESCO samples to fixed two-hand format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OLD_SAMPLE_SHAPE = (21, 3)
NEW_SAMPLE_SHAPE = (2, 21, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migra clases de una mano a formato (frames, 2, 21, 3).")
    parser.add_argument("--source", required=True, type=Path, help="Dataset origen con samples (frames, 21, 3).")
    parser.add_argument("--target", required=True, type=Path, help="Dataset destino para samples (frames, 2, 21, 3).")
    parser.add_argument("--labels", required=True, nargs="+", help="Labels de una mano que se migrarán.")
    return parser.parse_args()


def migrate_sample(source_file: Path, target_file: Path) -> None:
    sequence = np.load(source_file)
    if sequence.ndim != 3 or sequence.shape[1:] != OLD_SAMPLE_SHAPE:
        raise ValueError(f"shape inválido: {sequence.shape}")

    migrated = np.zeros((sequence.shape[0], *NEW_SAMPLE_SHAPE), dtype=np.float32)
    migrated[:, 0, :, :] = sequence.astype(np.float32)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_file, migrated)


def main() -> None:
    args = parse_args()
    source = args.source
    target = args.target
    labels = [label.strip().lower() for label in args.labels]

    if not source.exists():
        raise FileNotFoundError(f"No existe dataset origen: {source}")
    target.mkdir(parents=True, exist_ok=True)

    migrated_classes: list[str] = []
    omitted_classes: list[str] = []
    shape_errors: list[str] = []
    migrated_samples = 0

    available_labels = sorted(path.name for path in source.iterdir() if path.is_dir())
    for label in available_labels:
        source_label_dir = source / label
        if label not in labels:
            omitted_classes.append(label)
            continue

        sample_files = sorted(source_label_dir.glob("sample_*.npy"))
        if not sample_files:
            omitted_classes.append(label)
            continue

        migrated_classes.append(label)
        for source_file in sample_files:
            target_file = target / label / source_file.name
            try:
                migrate_sample(source_file, target_file)
                migrated_samples += 1
            except ValueError as exc:
                shape_errors.append(f"{label}/{source_file.name}: {exc}")

    print("Resumen migración dos manos")
    print(f"Clases migradas: {', '.join(migrated_classes) if migrated_classes else '-'}")
    print(f"Muestras migradas: {migrated_samples}")
    print(f"Clases omitidas: {', '.join(omitted_classes) if omitted_classes else '-'}")
    if shape_errors:
        print("Errores por shape:")
        for error in shape_errors:
            print(f"  {error}")
    else:
        print("Errores por shape: -")


if __name__ == "__main__":
    main()
