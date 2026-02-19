#!/usr/bin/env python3
"""Transfer downloaded TIFFs from `Config.DOWNLOAD_DIR` to the HPC.

Behavior (default):
- Query the local DuckDB for tasks with status `STATUS_DOWNLOADED` and collect
  their `local_filepath` entries (uses `DownloadDatabase.get_tasks_by_status`).
- Bundle files (and their accompanying .json metadata) into uncompressed tar
  archives (batch size = Config.COMPRESSION_BATCH_SIZE) and SCP them to
  ``HPC_DATA_PATH/downloads`` on the remote host.
- Extract archives on the remote host and (optionally) remove local files.

This script integrates with the existing `download.py` + `database.py` logic
so only known/downloaded TIFFs are transferred by default.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List
from .config import Config
from .database import DownloadDatabase

logger = logging.getLogger(__name__)


def find_files_from_db(db: DownloadDatabase, limit: int | None = None) -> List[Path]:
    """Return existing local file Paths for tasks marked `downloaded` in DB.

    - Queries the provided `DownloadDatabase` for tasks with status
      `Config.STATUS_DOWNLOADED` and returns a list of `Path` objects for
      files that actually exist on disk.
    - Rows without a `local_filepath` or entries pointing to missing files
      are skipped (missing files are logged as warnings).
    - `limit` (if provided) is passed through to the DB query.
    """
    cfg = Config()
    tasks = db.get_tasks_by_status(cfg.STATUS_DOWNLOADED, limit=limit, include_filenames=True)
    files: List[Path] = []
    for t in tasks:
        fp = t.get('local_filepath')
        if not fp:
            # DB row didn't include a local filepath — skip it.
            continue
        p = Path(fp)
        if p.exists():
            files.append(p)
        else:
            # File referenced in DB but not present locally; log for investigation.
            logger.warning(f"File listed in DB not found on disk: {p}")
    return files


def find_files_by_scan(download_dir: Path, pattern: str = 'LANDSAT_*.tif', limit: int | None = None) -> List[Path]:
    """Scan `download_dir` for files matching `pattern` and return sorted Paths.

    - Useful for ad-hoc transfers when the DB is not being consulted.
    - `limit` truncates the sorted result (keeps earliest entries after sort).
    """
    files = sorted(download_dir.glob(pattern))
    if limit:
        files = files[:limit]
    return files


def chunk_list(xs: List[Path], n: int) -> List[List[Path]]:
    """Yield successive n-sized chunks (sublists) from `xs`.

    This helper is used to split files into batches for archiving.
    """
    for i in range(0, len(xs), n):
        # slice returns a new list containing up to `n` elements
        yield xs[i:i + n]


def remote_mkdir(remote_user: str, remote_host: str, ssh_key: Path, remote_dir: str) -> None:
    """Ensure `remote_dir` exists on the remote host (creates parent directories).

    Uses SSH with the provided `ssh_key`. `StrictHostKeyChecking=no` is used
    to avoid interactive host key confirmation during automated runs. Raises
    subprocess.CalledProcessError on failure.
    """
    cmd = [
        'ssh',
        '-i', str(ssh_key),
        '-o', 'StrictHostKeyChecking=no',
        f"{remote_user}@{remote_host}",
        f"mkdir -p {remote_dir} && ls -ld {remote_dir}"
    ]
    # capture_output=True prevents printing SSH output to the caller's stdout
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def scp_transfer(ssh_key: Path, local_path: Path, remote_user: str, remote_host: str, remote_dir: str) -> None:
    """Transfer `local_path` to `remote_dir` on `remote_host` using `scp`.

    The trailing slash in `remote_dir/` ensures the archive is copied into
    that directory (not renamed).
    """
    remote_target = f"{remote_user}@{remote_host}:{remote_dir}/"
    scp_cmd = [
        'scp',
        '-i', str(ssh_key),
        '-o', 'StrictHostKeyChecking=no',
        str(local_path),
        remote_target
    ]
    subprocess.run(scp_cmd, check=True)


def remote_extract_and_cleanup(remote_user: str, remote_host: str, ssh_key: Path, remote_dir: str, archive_name: str) -> None:
    """SSH to remote host, extract `archive_name` inside `remote_dir`, then remove it.

    Extraction uses `tar -xf` (uncompressed). Any failure raises CalledProcessError.
    """
    cmd = [
        'ssh',
        '-i', str(ssh_key),
        '-o', 'StrictHostKeyChecking=no',
        f"{remote_user}@{remote_host}",
        f"cd {remote_dir} && tar -xf {archive_name} && rm {archive_name}"
    ]
    subprocess.run(cmd, check=True)


def make_archive(archive_path: Path, files: List[Path]) -> None:
    """Create an uncompressed tar archive at `archive_path` containing `files`.

    - Files are added with `arcname=p.name` to avoid storing full local paths.
    - For each TIFF, any adjacent `.json` metadata file is also added (if present).
    """
    # create uncompressed tar (fast for many small files)
    with tarfile.open(archive_path, 'w') as tar:
        for p in files:
            arcname = p.name
            # add file without its directory structure
            tar.add(p, arcname=arcname)
            # include accompanying .json metadata if present
            meta = p.with_suffix('.json')
            if meta.exists():
                tar.add(meta, arcname=meta.name)


def transfer_batches(files: List[Path], cfg: Config, batch_size: int | None = None, dry_run: bool = False) -> int:
    """Bundle `files` into tar archives and transfer them to the HPC host.

    - files: list of local TIFF Paths to transfer
    - cfg: Config instance (provides HPC connection info and directories)
    - batch_size: override for number of files per archive (defaults to cfg.COMPRESSION_BATCH_SIZE)
    - dry_run: if True, don't perform network operations (helpful for testing)
    Returns the number of files successfully transferred.
    """
    if not files:
        logger.info("No TIFF files found to transfer")
        return 0

    batch_size = batch_size or cfg.COMPRESSION_BATCH_SIZE
    remote_dir = f"{cfg.HPC_DATA_PATH}/downloads"
    transferred = 0

    # Use a temporary directory inside cfg.DATA_DIR to store the archives
    with tempfile.TemporaryDirectory(dir=str(cfg.DATA_DIR)) as td:
        temp_dir = Path(td)
        # chunk_list yields batches of at most `batch_size` files
        for idx, batch in enumerate(chunk_list(files, batch_size), start=1):
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            archive_name = f"downloads_{timestamp}_{idx}.tar"
            archive_path = temp_dir / archive_name

            logger.info(f"Creating archive ({len(batch)} files): {archive_name}")
            make_archive(archive_path, batch)

            if dry_run:
                # don't perform network operations during dry-run
                logger.info(f"DRY-RUN: would transfer {archive_path} -> {cfg.HPC_HOST}:{remote_dir}")
                continue

            try:
                # ensure remote directory exists before attempting copy
                logger.info(f"Ensuring remote directory: {remote_dir}")
                remote_mkdir(cfg.HPC_USER, cfg.HPC_HOST, cfg.SSH_KEY, remote_dir)

                logger.info(f"Transferring archive to {cfg.HPC_HOST}:{remote_dir}")
                scp_transfer(cfg.SSH_KEY, archive_path, cfg.HPC_USER, cfg.HPC_HOST, remote_dir)

                logger.info("Extracting archive on remote host")
                remote_extract_and_cleanup(cfg.HPC_USER, cfg.HPC_HOST, cfg.SSH_KEY, remote_dir, archive_name)

                # update counters only after successful remote extraction
                transferred += len(batch)

                # local cleanup removed — transferred files are retained locally

            except subprocess.CalledProcessError as e:
                # bubble up the SSH/SCP error after logging
                logger.error(f"Transfer failed for archive {archive_name}: {e}")
                raise
            finally:
                # remove temporary archive (if it still exists)
                if archive_path.exists():
                    archive_path.unlink()

    logger.info(f"Transferred {transferred} files to {cfg.HPC_HOST}:{remote_dir}")
    return transferred


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the transfer script.

    Supports dry-run mode and limiting number of files.
    """
    p = argparse.ArgumentParser(description='Transfer downloaded TIFFs to HPC')
    p.add_argument('--limit', type=int, default=None, help='Maximum number of files to transfer')
    p.add_argument('--batch-size', type=int, default=None, help='Files per archive (overrides config)')
    p.add_argument('--scan-dir', action='store_true', help='Scan DOWNLOAD_DIR instead of DB for files')
    p.add_argument('--dry-run', action='store_true', help='Show files/archives that would be transferred without performing the transfer')
    return p.parse_args()


def main() -> int:
    """CLI entrypoint for the transfer script.

    - Parses arguments, locates files (DB or directory scan), and calls `transfer_batches`.
    - Returns 0 on normal completion.
    """
    args = parse_args()

    cfg = Config()
    # create any missing local directories used by the script
    cfg.ensure_dirs()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    files: List[Path] = []
    if args.scan_dir:
        logger.info(f"Scanning download directory: {cfg.DOWNLOAD_DIR}")
        files = find_files_by_scan(cfg.DOWNLOAD_DIR, limit=args.limit)
    else:
        db = DownloadDatabase(str(cfg.DB_PATH))
        logger.info("Querying database for tasks with status 'downloaded'")
        files = find_files_from_db(db, limit=args.limit)

    if not files:
        logger.info("No files to transfer (nothing to do)")
        return 0

    logger.info(f"Preparing to transfer {len(files)} files (batch size={args.batch_size or cfg.COMPRESSION_BATCH_SIZE})")

    transferred = transfer_batches(files, cfg, batch_size=args.batch_size, dry_run=args.dry_run)

    logger.info(f"Done — transferred {transferred} files")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
