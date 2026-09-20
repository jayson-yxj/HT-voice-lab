"""Download only the pinned official checkpoints; never print credentials."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def token():
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return path.read_text().strip() if path.is_file() else None


def fetch_one(name, spec):
    target = ROOT / "models" / name
    target.mkdir(parents=True, exist_ok=True)
    lock = target / "download.json"
    old = json.loads(lock.read_text()) if lock.exists() else {}
    records = {}
    opener = urllib.request.build_opener(SafeRedirect())
    for filename in spec["files"]:
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = old.get("files", {}).get(filename, {})
        if path.is_file() and previous.get("size") == path.stat().st_size and old.get("revision") == spec["revision"]:
            records[filename] = previous
            continue
        url = "https://huggingface.co/{}/resolve/{}/{}".format(spec["repo"], spec["revision"], filename)
        headers = {"User-Agent": "HT-voice-lab/1.0"}
        access_token = token()
        if access_token:
            headers["Authorization"] = "Bearer " + access_token
        # An interrupted large weight download can resume, but never across revisions.
        part = path.with_name(path.name + "." + spec["revision"] + ".part")
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                offset = part.stat().st_size if part.exists() else 0
                request_headers = dict(headers)
                if offset:
                    request_headers["Range"] = "bytes={}-".format(offset)
                print("Downloading {}/{}".format(name, filename), flush=True)
                request_url = url + ("?resume={}".format(offset) if offset else "")
                with opener.open(urllib.request.Request(request_url, headers=request_headers), timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith("bytes {}-".format(offset)):
                        raise IOError("Unexpected download resume offset")
                    if append:
                        with part.open("rb") as existing:
                            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                                digest.update(chunk)
                    size = offset if append else 0
                    expected = response.headers.get("Content-Length")
                    expected_size = size + int(expected) if expected else None
                    with part.open("ab" if append else "wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if expected_size is not None and size != expected_size:
                        raise IOError("Incomplete download")
                part.replace(path)
                records[filename] = {"size": size, "sha256": digest.hexdigest()}
                lock.write_text(json.dumps(dict(spec, files=records), indent=2) + "\n")
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    return {"model": name, "ready": False, "reason": "HF access required (HTTP {})".format(exc.code)}
                if attempt == 2:
                    return {"model": name, "ready": False, "reason": "HTTP {}".format(exc.code)}
                time.sleep(1)
            except (OSError, TimeoutError) as exc:
                if attempt == 2:
                    return {"model": name, "ready": False, "reason": type(exc).__name__}
                time.sleep(1)
    lock.write_text(json.dumps(dict(spec, files=records), indent=2) + "\n")
    return {"model": name, "ready": True}


def main():
    specs = json.loads((ROOT / "models.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(specs), default=list(specs))
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: fetch_one(name, specs[name]), args.only))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(r["ready"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
