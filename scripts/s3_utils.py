"""
s3_utils.py — AWS S3 Integration for Central Virginia Tree Canopy Project
==========================================================================
Provides robust, non-fatal S3 upload functions for pipeline scripts.
Upload failures log an error but do not crash the calling script.
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "central-virginia-tree-canopy-project"

def get_s3_client():
    """Return a boto3 S3 client if available and configured, else None."""
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 not installed. S3 upload disabled. Run: pip install boto3")
        return None
    
    try:
        # Boto3 automatically uses ~/.aws/credentials or environment variables
        client = boto3.client('s3')
        # Test credentials by checking caller identity (STS)
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return client
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"AWS credentials not configured or invalid. S3 upload disabled. {e}")
        return None


def upload_file(client, local_path: Path, bucket: str, s3_key: str) -> bool:
    """
    Upload a single file to S3. Non-fatal: returns False on error.
    """
    if not client or not local_path.exists():
        return False
        
    try:
        # Determine content type based on extension
        content_type = "binary/octet-stream"
        if local_path.suffix == ".tif":
            content_type = "image/tiff"
        elif local_path.suffix == ".csv":
            content_type = "text/csv"
        elif local_path.suffix in (".txt", ".log"):
            content_type = "text/plain"
            
        extra_args = {'ContentType': content_type}
        
        client.upload_file(
            str(local_path), 
            bucket, 
            s3_key,
            ExtraArgs=extra_args
        )
        logger.info(f"S3 Upload OK: s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 Upload FAILED for {local_path.name}: {e}")
        return False


def build_s3_key(category: str, county: str, year_tag: str, filename: str) -> str:
    """
    Construct a standardized S3 object key.
    Category examples: 'chm', 'centroids', 'change', 'logs'
    """
    county_clean = county.lower().replace(" ", "_")
    
    # Special handling for change detection which has two years (e.g. 2015_2020)
    if category == "change":
        return f"{category}/{county_clean}/{year_tag}/{filename}"
        
    # Standard single-year output
    return f"{category}/{county_clean}/{year_tag}/{filename}"
