"""Publish one immutable Android host kit, then its Hub install receipt."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

from publish_hub_objects import PUBLIC_ENDPOINT, Publisher

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
from android_support import ANDROID_SUPPORT_VERSION, archive_name


def publish(archive: Path, publisher: Publisher) -> None:
    if archive.name != archive_name():
        raise ValueError(f"The Android kit for this host must be named {archive_name()}")
    key = f"android-support/{ANDROID_SUPPORT_VERSION}/{archive.name}"
    if publisher.exists(key):
        # Different engine releases may share one kit. Attach the original bytes
        # to the next GitHub release even if Gradle produced different metadata.
        with urllib.request.urlopen(f"{PUBLIC_ENDPOINT}/{key}", timeout=600) as source:
            with archive.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    else:
        publisher.upload(archive, key)
    receipt = archive.with_suffix(archive.suffix + ".json")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt.write_text(json.dumps({"sha256": digest, "size": archive.stat().st_size}) + "\n",
                       encoding="utf-8", newline="\n")
    publisher.upload(receipt, key + ".json")


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Publisher(os.environ["R2_UPLOAD_TOKEN"]))
