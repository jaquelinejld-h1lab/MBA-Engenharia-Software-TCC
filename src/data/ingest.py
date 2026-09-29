"""Raw dataset ingestion: integrity check, Excel read and Parquet materialization.

Reading the 12 MB workbook takes about twenty seconds; the pipeline therefore
converts it once to Parquet under ``data/interim`` and every later stage reads
the Parquet copy. The SHA-256 of the workbook is checked against the value in
``configs/config.yaml`` before anything is read, so a silently replaced file
fails fast with a ``DataContractError``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DataConfig, PathsConfig
from src.exceptions import DataContractError
from src.logging_setup import get_logger

logger = get_logger(__name__)

_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class IngestReport:
    """What the ingestion produced."""

    source: Path
    sha256: str
    rows: int
    columns: int
    interim_path: Path


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    """Compare the file digest with ``expected``.

    Returns
    -------
    str
        The computed digest.

    Raises
    ------
    DataContractError
        If the file is missing or the digest differs.
    """
    if not path.is_file():
        raise DataContractError(f"raw dataset not found: '{path}'")
    actual = sha256_of_file(path)
    if actual != expected:
        raise DataContractError(
            f"raw dataset '{path.name}' has SHA-256 {actual}, expected {expected}. "
            "The file is not the frozen PNS 2013 extract; refusing to ingest."
        )
    return actual


def read_raw_excel(path: Path, sheet: str) -> pd.DataFrame:
    """Read the workbook sheet and upper-case column names (as the notebook did).

    Parameters
    ----------
    path : Path
        Workbook location.
    sheet : str
        Sheet name declared in the config.

    Returns
    -------
    pandas.DataFrame
        Raw frame with upper-cased column names.
    """
    frame = pd.read_excel(path, sheet_name=sheet)
    frame.columns = frame.columns.str.upper()
    return frame


def check_dimensions(frame: pd.DataFrame, data_config: DataConfig) -> None:
    """Fail if the raw frame does not have the declared shape."""
    rows, columns = frame.shape
    if (rows, columns) != (data_config.expected_raw_rows, data_config.expected_raw_columns):
        raise DataContractError(
            f"raw dataset has shape {(rows, columns)}, expected "
            f"{(data_config.expected_raw_rows, data_config.expected_raw_columns)}"
        )


def write_interim(frame: pd.DataFrame, interim_dir: Path, stem: str) -> Path:
    """Write ``frame`` as Parquet under ``interim_dir`` and return the file path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    target = interim_dir / f"{stem}.parquet"
    frame.to_parquet(target, index=False)
    return target


def ingest(paths: PathsConfig, data_config: DataConfig) -> IngestReport:
    """Run the full ingestion: hash check, read, shape check, Parquet write.

    Parameters
    ----------
    paths : PathsConfig
        Locations from the config.
    data_config : DataConfig
        Expected hash, sheet and shape.

    Returns
    -------
    IngestReport
    """
    source = paths.absolute("raw_dataset")
    digest = verify_sha256(source, data_config.raw_sha256)
    logger.info("raw dataset hash verified", extra={"file": source.name, "sha256": digest})

    frame = read_raw_excel(source, data_config.raw_sheet)
    check_dimensions(frame, data_config)

    interim_path = write_interim(frame, paths.absolute("interim_dir"), f"raw_{digest[:12]}")
    logger.info(
        "raw dataset materialized",
        extra={
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "sha256": digest,
            "interim_path": str(interim_path.relative_to(interim_path.parents[2])),
        },
    )
    return IngestReport(
        source=source,
        sha256=digest,
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        interim_path=interim_path,
    )
