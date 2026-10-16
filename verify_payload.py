"""Check that the published records carry the same numbers as the records as written, and
that every number quoted in README.md is the number in its record.

Before publication, the absolute data-root prefix of the paths quoted in the column
source_file of two records (results/statistical_tests/gnn_best_epoch_all_metrics.csv and
cnn_best_epoch_all_metrics.csv) was replaced by ${DEMGNN_DATA_ROOT}. Every other record is
byte-identical to the record as written.

For each file listed in PAYLOAD_SHA256 this script computes the sha256 of its "payload":
the number of rows and of cells per row, and, for each cell, the list of the numbers it
contains, in order, after the prefix above is turned into a fixed token. PAYLOAD_SHA256
lists the digest of each record as written, computed before publication.

The script then checks each entry of README_CLAIMS: the quoted text must appear in
README.md, and the cell of the given record (row selected by its biome, given column),
rounded or formatted as quoted, must be the quoted value.

Usage (from the repository root):  python3 verify_payload.py
"""
import csv
import hashlib
import io
import json
import pathlib
import re
import sys

ROOTS = {"${DEMGNN_DATA_ROOT}": "<root>"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
BEST = "results/statistical_tests/gnn_best_epoch_all_metrics.csv"
T2 = "results/statistical_tests/t2_statistical_framework.csv"
NAMES = {"amazonia": "Amazonia", "cerrado": "Cerrado", "mata_atlantica": "Mata Atlantica", "pantanal": "Pantanal"}

# (quoted text in README.md, record, biome, column, quoted value, format)
README_CLAIMS = []
for b, mae, rmse in (("amazonia", "2.72", "3.49"), ("cerrado", "2.21", "2.77"),
                     ("mata_atlantica", "2.10", "3.03"), ("pantanal", "2.81", "9.84")):
    row = f"| {NAMES[b]} | {mae} | {rmse} |"
    README_CLAIMS += [(row, BEST, b, "elevation_mae", mae, ".2f"), (row, BEST, b, "elevation_rmse", rmse, ".2f")]
for b, n, g, c, p, w in (("amazonia", "120", "2.52", "9.62", "2.0e-21", "GNN"),
                         ("cerrado", "120", "13.41", "2.86", "2.2e-21", "CNN"),
                         ("mata_atlantica", "118", "3.24", "16.17", "9.8e-21", "GNN"),
                         ("pantanal", "120", "0.63", "3.19", "5.3e-19", "GNN")):
    row = f"| {NAMES[b]} | {n} | {g} | {c} | {p} | {w} |"
    README_CLAIMS += [(row, T2, b, "n_pairs", n, "d"), (row, T2, b, "gnn_mean", g, ".2f"),
                      (row, T2, b, "cnn_mean", c, ".2f"), (row, T2, b, "wilcoxon_p", p, ".1e"),
                      (row, T2, b, "winner", w, "s")]


def root(s):
    for prefix, token in ROOTS.items():
        s = s.replace(prefix, token)
    return s


def csv_payload(text, root_fn=root):
    delim = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    canon = json.dumps([[NUMBER.findall(root_fn(c)) for c in r] for r in rows], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def payload_sha256(path, rel):
    return csv_payload(pathlib.Path(path).read_text(encoding="utf-8-sig"))


def cell(base, rel, biome, column):
    rows = list(csv.DictReader(io.StringIO((base / rel).read_text(encoding="utf-8-sig"))))
    hits = [r for r in rows if r["biome"] == biome]
    return hits[0][column] if len(hits) == 1 else None


def fmt(v, f):
    if v is None:
        return None
    if f == "s":
        return v
    if f == "d":
        return str(int(float(v)))
    return format(float(v), f)


def main():
    base = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (base / "PAYLOAD_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        ok = payload_sha256(base / rel, rel) == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} records: payload identical to the record as written")
    readme = (base / "README.md").read_text(encoding="utf-8")
    bad_claims = 0
    for quoted, rel, biome, column, value, f in README_CLAIMS:
        got = fmt(cell(base, rel, biome, column), f)
        ok = quoted in readme and got == value
        bad_claims += not ok
        if not ok:
            print(f"CLAIM FAILS: {quoted!r} {column} = {got!r}, quoted {value!r}")
    print(f"{len(README_CLAIMS) - bad_claims}/{len(README_CLAIMS)} README numbers found in their records")
    sys.exit(1 if bad or bad_claims else 0)


if __name__ == "__main__":
    main()
