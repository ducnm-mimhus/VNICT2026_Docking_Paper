#!/usr/bin/env python3
"""Print dataset / channel statistics before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_types(path: Path) -> dict:
    rows, labels, affinities, gninatypes_refs = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            toks = stripped.split()
            rows.append(toks)
            if toks:
                try:
                    labels.append(int(float(toks[0])))
                except ValueError:
                    pass
            if len(toks) > 1:
                try:
                    affinities.append(float(toks[1]))
                except ValueError:
                    pass
            gninatypes_refs.extend(t for t in toks if t.endswith(".gninatypes"))
    return {
        "n": len(rows),
        "label_counts": Counter(labels),
        "pos_aff": sum(1 for x in affinities if x > 0),
        "neg_aff": sum(1 for x in affinities if x < 0),
        "zero_aff": sum(1 for x in affinities if x == 0),
        "affinities": affinities,
        "gninatypes_refs": gninatypes_refs,
    }


def infer_channel_names(data_root: Path, project_dir: Path, parsed: dict, max_files: int = 200) -> list[str]:
    channels, seen = [], set()
    for ref in parsed["gninatypes_refs"][:max_files]:
        for candidate in (
            data_root / "PDBbind2016" / ref,
            data_root / ref,
            project_dir / ref,
            Path(ref),
        ):
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    toks = line.strip().split()
                    if toks and toks[0] not in seen:
                        seen.add(toks[0])
                        channels.append(toks[0])
            break
    return channels


def molgrid_probe(data_root: Path, types_path: Path, batch_size: int, dimension: float, resolution: float) -> dict:
    out: dict = {}
    try:
        import molgrid

        provider = molgrid.ExampleProvider(
            data_root=str(data_root),
            balanced=False,
            shuffle=False,
            default_batch_size=batch_size,
            iteration_scheme=molgrid.IterationScheme.SmallEpoch,
            cache_structs=False,
        )
        provider.populate(str(types_path))
        out["provider_size"] = provider.size()
        out["num_labels"] = provider.num_labels()
        out["num_types"] = provider.num_types()
        grid_maker = molgrid.GridMaker(resolution=resolution, dimension=dimension)
        out["grid_dims"] = tuple(int(x) for x in grid_maker.grid_dimensions(provider.num_types()))
        for attr in ("type_names", "get_type_names", "get_typenames"):
            if hasattr(provider, attr):
                value = getattr(provider, attr)
                names = value() if callable(value) else value
                out["molgrid_type_names"] = [str(x) for x in names]
                break
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    train = parse_types(args.train_file)
    test = parse_types(args.test_file)
    probe = molgrid_probe(args.data_root, args.train_file, args.batch_size, 23.5, 0.5)
    channels = infer_channel_names(args.data_root, args.project_dir, train)
    channel_names = probe.get("molgrid_type_names") or channels
    if not channel_names and probe.get("num_types"):
        channel_names = [f"channel_{i:02d}" for i in range(int(probe["num_types"]))]

    print("==========================================")
    print("DATASET REPORT")
    print("==========================================")
    print("Dataset: CrossDocked2020 / PDBbind2016 (Francoeur2020 split)")
    print(f"Data root:   {args.data_root}")
    print(f"Train types: {args.train_file}")
    print(f"Test types:  {args.test_file}")
    print()
    for name, parsed in (("train", train), ("test", test)):
        aff = parsed["affinities"]
        print(f"  {name}: n={parsed['n']}, pose 0/1={parsed['label_counts'].get(0, 0)}/{parsed['label_counts'].get(1, 0)}, "
              f"aff +/-/0={parsed['pos_aff']}/{parsed['neg_aff']}/{parsed['zero_aff']}")
        if aff:
            print(f"           affinity min/mean/max={min(aff):.4f}/{sum(aff)/len(aff):.4f}/{max(aff):.4f}")
    print()
    if "provider_size" in probe:
        print(f"molgrid: size={probe['provider_size']}, labels={probe['num_labels']}, channels={probe['num_types']}, grid={probe['grid_dims']}")
    elif "error" in probe:
        print(f"molgrid probe unavailable: {probe['error']}")
    print()
    for idx, name in enumerate(channel_names):
        print(f"  channel[{idx:02d}] {name}")
    print("==========================================")


if __name__ == "__main__":
    main()
