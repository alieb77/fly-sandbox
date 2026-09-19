"""Download FlyWire connectome data (public, no login).
Sources:
  - Neuron annotations: flyconnectome/flywire_annotations (GitHub)
  - Edge list / connections: Zenodo 10676866
Usage:
  python download.py annotations   # small TSV
  python download.py connections   # 852 MB feather
"""
import sys, os, time, urllib.request

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)

FILES = {
    "annotations": [
        # try main then master
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "https://media.githubusercontent.com/media/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    ],
    "connections": [
        "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
    ],
}
OUT = {
    "annotations": os.path.join(DATA, "neuron_annotations.tsv"),
    "connections": os.path.join(DATA, "proofread_connections_783.feather"),
}


def human(n):
    for u in "B KB MB GB".split():
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url, out):
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        print(f"  status={r.status}  size={human(total) if total else '?'}")
        tmp = out + ".part"
        got = 0
        t0 = time.time()
        last = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 2:
                    last = time.time()
                    sp = got / max(time.time() - t0, 1e-9)
                    pct = f"{100*got/total:.0f}%" if total else ""
                    print(f"  {human(got)} {pct}  {human(sp)}/s", flush=True)
        os.replace(tmp, out)
    print(f"  -> saved {out} ({human(os.path.getsize(out))})")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "annotations"
    out = OUT[key]
    if os.path.exists(out):
        print(f"already have {out} ({human(os.path.getsize(out))}) - skip")
        return
    last_err = None
    for url in FILES[key]:
        try:
            download(url, out)
            return
        except Exception as e:
            print(f"  failed: {e}")
            last_err = e
    raise SystemExit(f"all mirrors failed: {last_err}")


if __name__ == "__main__":
    main()
