"""Small local configuration editor for live recognition."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from runtime_config import LiveRecognitionConfig, load_runtime_config, save_runtime_config


def _parse_value(raw: str, current: object) -> object:
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "si", "s"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError("usa true/false")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw.strip()


def open_config_editor(config_path: Path) -> None:
    """Open a lightweight terminal configuration screen."""
    config = load_runtime_config(config_path)
    field_names = [field.name for field in fields(LiveRecognitionConfig)]

    while True:
        print("\nConfiguracion LESCO live")
        print("------------------------")
        for index, name in enumerate(field_names, start=1):
            print(f"{index:2d}. {name}: {getattr(config, name)}")
        print("\ns. guardar | r. restaurar defaults | q. salir")

        choice = input("Seleccion: ").strip().lower()
        if choice == "q":
            return
        if choice == "r":
            config = LiveRecognitionConfig()
            print("Defaults restaurados en memoria.")
            continue
        if choice == "s":
            save_runtime_config(config, config_path)
            print(f"Configuracion guardada en {config_path}")
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(field_names):
            print("Opcion invalida.")
            continue

        name = field_names[int(choice) - 1]
        current = getattr(config, name)
        raw_value = input(f"Nuevo valor para {name} [{current}]: ").strip()
        if not raw_value:
            continue
        try:
            updated = LiveRecognitionConfig.from_dict({**config.to_dict(), name: _parse_value(raw_value, current)})
        except ValueError as exc:
            print(f"Valor invalido: {exc}")
            continue
        config = updated
