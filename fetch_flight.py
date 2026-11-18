"""Bird photos whose observers said whether the bird was flying (iNaturalist observation fields).

  ./run ../bioclip-bird-heads/fetch_flight.py   # from a bioclip-bird checkout -> data/flight-raw/{fly,perch}/OBSERVATION_PHOTO.jpg + manifest.csv

fly: "birds in flight" (any value), "Behavior: flying" = yes, "Flying?" = yes. perch: "Flying?" = no.
One licensed photo per observation (its first), 500 px, never one of the golden set's. About 5,000 photos, 0.5 GB,
some 35 API calls. The label is the observer's word about the observation, not about this photo: expect some noise.
"""
from __future__ import annotations

import csv
from pathlib import Path

import distill
from distill.fetch_inat import BIRDS, FIELDS, LICENSES, PAGE, Api, Budget, download, golden_ids, pick

OUT = Path(distill.__file__).resolve().parents[1] / "data" / "flight-raw"
SOURCES = [("fly", "birds in flight", "", 2000), ("fly", "Behavior: flying", "yes", 2000), ("fly", "Flying?", "yes", 1200), ("perch", "Flying?", "no", 2000)]


def main() -> None:
    api, skip, seen, rows = Api(), golden_ids(), set(), []
    for label, field, value, cap in SOURCES:
        got, page = 0, 1
        while got < cap:
            results = api("observations", taxon_id=BIRDS, photos="true", photo_license=",".join(LICENSES), per_page=PAGE, page=page,
                          **{f"field:{field}": value})["results"]
            for obs in results:
                for r in pick([obs], {"scientific": label, "taxon_id": (obs.get("taxon") or {}).get("id", "")}, skip, seen, min(1, cap - got)):
                    rows.append({**r, "scientific": (obs.get("taxon") or {}).get("name", "")})
                    got += 1
            if len(results) < PAGE:
                break
            page += 1
        print(f"{label}: {field}={value or '*'}: {got}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    download(OUT, rows, Budget(4.0, 1.0))
    print(f"done: {sum(1 for r in rows if (OUT / r['path']).exists())}/{len(rows)} photos on disk")


if __name__ == "__main__":
    main()
