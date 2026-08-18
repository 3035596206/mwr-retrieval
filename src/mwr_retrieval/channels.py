"""Single source of truth for instrument channel selections."""

from __future__ import annotations

from .settings import load_instruments


def channel_indices(instrument_id: str, channel_set: str) -> list[int]:
    instrument = load_instruments()[instrument_id]
    try:
        return list(instrument["channel_sets"][channel_set])
    except KeyError as error:
        choices = ", ".join(sorted(instrument["channel_sets"]))
        raise ValueError(f"Unknown channel set {channel_set!r}; choose one of: {choices}") from error


def frequencies_ghz(instrument_id: str) -> list[float]:
    return [channel["frequency_ghz"] for channel in load_instruments()[instrument_id]["channels"]]
