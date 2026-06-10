"""
upload_smap_h5_to_s3.py
-----------------------
Batch-upload SMAP SPL3SMP_E HDF5 files from a local directory to AWS S3
using boto3 multipart transfer with a progress bar.

Source directory : /scratch/thq3hn/smap_h5/
S3 destination   : s3://central-virginia-tree-canopy-project/SMAP/

Usage
-----
    # Upload all *.h5 files (default)
    python upload_smap_h5_to_s3.py

    # Dry-run — list files that would be uploaded, no actual transfer
    python upload_smap_h5_to_s3.py --dry-run

    # Skip files already present in S3 (resume interrupted upload)
    python upload_smap_h5_to_s3.py --skip-existing

    # Override source directory or bucket
    python upload_smap_h5_to_s3.py --src-dir /scratch/thq3hn/smap_h5 \
                                    --bucket central-virginia-tree-canopy-project \
                                    --s3-prefix SMAP/
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_SRC_DIR   = "/scratch/thq3hn/smap_h5"
DEFAULT_BUCKET    = "central-virginia-tree-canopy-project"
DEFAULT_S3_PREFIX = "SMAP/"          # trailing slash keeps folder structure
PART_SIZE_MB      = 64               # multipart part size in MB
MAX_CONCURRENCY   = 10               # parallel upload threads per file
FILE_GLOB         = "*.h5"


# ── Progress bar ──────────────────────────────────────────────────────────
class ProgressBar:
    """Thread-safe per-file upload progress bar."""

    def __init__(self, filename: str, total_bytes: int):
        self._filename    = filename
        self._total       = total_bytes
        self._uploaded    = 0
        self._start       = time.time()
        self._lock        = threading.Lock()
        self._bar_width   = 30

    def __call__(self, bytes_transferred: int):
        with self._lock:
            self._uploaded += bytes_transferred
            pct     = self._uploaded / self._total if self._total else 1.0
            filled  = int(self._bar_width * pct)
            bar     = "█" * filled + "░" * (self._bar_width - filled)
            elapsed = time.time() - self._start
            speed   = (self._uploaded / elapsed / 1_048_576) if elapsed > 0 else 0
            done_mb = self._uploaded / 1_048_576
            tot_mb  = self._total    / 1_048_576
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1%}  {done_mb:6.1f}/{tot_mb:.1f} MB"
                f"  {speed:6.1f} MB/s"
            )
            sys.stdout.flush()
            if self._uploaded >= self._total:
                elapsed_s = time.time() - self._start
                avg_speed = self._total / elapsed_s / 1_048_576 if elapsed_s else 0
                print(f"  ✓  {elapsed_s:.1f}s  ({avg_speed:.1f} MB/s avg)")


# ── Helpers ───────────────────────────────────────────────────────────────
def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def s3_key_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def validate_credentials(s3_client) -> str:
    """Return the AWS identity string or raise on failure."""
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    return identity.get("Arn", "unknown")


# ── Main upload logic ─────────────────────────────────────────────────────
def upload_files(
    src_dir: str,
    bucket: str,
    s3_prefix: str,
    dry_run: bool,
    skip_existing: bool,
) -> None:

    src_path = Path(src_dir)
    if not src_path.is_dir():
        print(f"[ERROR] Source directory not found: {src_dir}")
        sys.exit(1)

    h5_files = sorted(src_path.glob(FILE_GLOB))
    if not h5_files:
        print(f"[WARN]  No {FILE_GLOB} files found in {src_dir}")
        sys.exit(0)

    total_bytes = sum(f.stat().st_size for f in h5_files)
    print("=" * 64)
    print("  SMAP HDF5 → S3 Batch Upload")
    print("=" * 64)
    print(f"  Source dir  : {src_dir}")
    print(f"  S3 target   : s3://{bucket}/{s3_prefix}")
    print(f"  Files found : {len(h5_files):,}")
    print(f"  Total size  : {human_size(total_bytes)}")
    print(f"  Part size   : {PART_SIZE_MB} MB   Threads : {MAX_CONCURRENCY}")
    if dry_run:
        print("\n  [DRY-RUN MODE — no files will be transferred]\n")
    print("=" * 64)

    # ── Validate AWS credentials ──────────────────────────────────────────
    if not dry_run:
        try:
            s3 = boto3.client("s3")
            arn = validate_credentials(s3)
            print(f"\n  AWS identity : {arn}")
        except NoCredentialsError:
            print("\n[ERROR] No AWS credentials found.")
            print("        Run: aws configure   or set AWS_ACCESS_KEY_ID / "
                  "AWS_SECRET_ACCESS_KEY environment variables.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Credential validation failed: {e}")
            sys.exit(1)

        transfer_cfg = TransferConfig(
            multipart_threshold = PART_SIZE_MB * 1_048_576,
            multipart_chunksize = PART_SIZE_MB * 1_048_576,
            max_concurrency     = MAX_CONCURRENCY,
            use_threads         = True,
        )

    # ── Upload loop ───────────────────────────────────────────────────────
    uploaded   = 0
    skipped    = 0
    failed     = 0
    start_all  = time.time()

    for i, fpath in enumerate(h5_files, 1):
        s3_key    = s3_prefix + fpath.name
        file_size = fpath.stat().st_size

        print(f"\n  [{i:>4}/{len(h5_files)}]  {fpath.name}")
        print(f"           Size : {human_size(file_size)}")
        print(f"           Key  : s3://{bucket}/{s3_key}")

        if dry_run:
            print("           [DRY-RUN — skipped]")
            continue

        # Skip if already in S3
        if skip_existing and s3_key_exists(s3, bucket, s3_key):
            print("           [SKIP — already exists in S3]")
            skipped += 1
            continue

        # Upload with progress bar
        progress = ProgressBar(fpath.name, file_size)
        try:
            s3.upload_file(
                str(fpath),
                bucket,
                s3_key,
                Config    = transfer_cfg,
                Callback  = progress,
                ExtraArgs = {"ContentType": "application/x-hdf5"},
            )
            uploaded += 1
        except ClientError as e:
            print(f"\n  [ERROR] Upload failed: {e}")
            failed += 1
            continue

        # Verify remote file size
        try:
            head = s3.head_object(Bucket=bucket, Key=s3_key)
            remote_size = head["ContentLength"]
            if remote_size == file_size:
                print(f"           Verified : {human_size(remote_size)} ✓")
            else:
                print(f"           [WARN] Size mismatch — "
                      f"local {file_size} vs remote {remote_size}")
        except ClientError as e:
            print(f"           [WARN] Verification failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_all
    print("\n" + "=" * 64)
    print("  UPLOAD SUMMARY")
    print("=" * 64)
    if dry_run:
        print(f"  [DRY-RUN]  {len(h5_files)} files would be uploaded")
        print(f"             Total size : {human_size(total_bytes)}")
    else:
        print(f"  Uploaded : {uploaded:>4} files")
        print(f"  Skipped  : {skipped:>4} files (already in S3)")
        print(f"  Failed   : {failed:>4} files")
        print(f"  Duration : {elapsed_total:.1f}s")
        if uploaded > 0 and elapsed_total > 0:
            avg_speed = total_bytes / elapsed_total / 1_048_576
            print(f"  Avg speed: {avg_speed:.1f} MB/s")
        print(f"\n  S3 prefix : s3://{bucket}/{s3_prefix}")
    print("=" * 64)


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-upload SMAP SPL3SMP_E HDF5 files to AWS S3"
    )
    p.add_argument(
        "--src-dir", default=DEFAULT_SRC_DIR,
        help=f"Local directory containing *.h5 files (default: {DEFAULT_SRC_DIR})"
    )
    p.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    p.add_argument(
        "--s3-prefix", default=DEFAULT_S3_PREFIX,
        help=f"S3 key prefix / folder (default: {DEFAULT_S3_PREFIX})"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be uploaded without transferring"
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip files already present in S3 (resume interrupted upload)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Ensure prefix ends with /
    prefix = args.s3_prefix.rstrip("/") + "/"
    upload_files(
        src_dir      = args.src_dir,
        bucket       = args.bucket,
        s3_prefix    = prefix,
        dry_run      = args.dry_run,
        skip_existing= args.skip_existing,
    )
