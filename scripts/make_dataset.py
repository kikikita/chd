import argparse
from pathlib import Path

import pandas as pd

from data_utils import process_all, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from raw metrics")
    parser.add_argument("--metrics-glob", required=True, help="Glob pattern for raw metrics files")
    parser.add_argument("--incidents", type=Path, required=True, help="Path to incidents csv")
    parser.add_argument("--info", type=Path, required=True, help="Path to info csv")
    parser.add_argument("--out", type=Path, required=True, help="Output csv path")
    parser.add_argument("--feature-dir", type=Path, required=True, help="Directory to store feature mapping")
    parser.add_argument("--time-lag", type=int, default=24, help="Hours for initial target")
    parser.add_argument("--is-test", action="store_true", help="Process test data without incidents")
    args = parser.parse_args()

    raw_df = process_all(
        metrics_glob=args.metrics_glob,
        incidents_path=args.incidents if not args.is_test else None,
        info_path=args.info,
        out_dir=args.feature_dir,
        time_lag_hours=args.time_lag,
        is_test=args.is_test,
    )

    feature_list = args.feature_dir / "feature_list.json"
    rename_map = args.feature_dir / "rename_map.json"

    df = prepare_dataset(
        raw_df,
        feature_list_path=None if args.is_test else feature_list,
        rename_map_path=rename_map,
        is_test=args.is_test,
    )

    df.to_csv(args.out, index=False)
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()
