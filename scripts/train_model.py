import argparse
from pathlib import Path

import pandas as pd

from data_utils import (
    load_incidents,
    calc_hours_to_fail,
    update_target_by_htf,
    split_dataset,
    ModelTrainer,
    DEVICE_COL,
    TIME_COL,
    TARGET_COL,
)


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset CSV")
    parser.add_argument("--incidents", type=Path, required=True, help="Incidents CSV")
    parser.add_argument("--lag-hours", type=int, default=3, help="Prediction horizon in hours")
    parser.add_argument("--date-split", required=True, help="Date YYYY-MM-DD for train/test split")
    parser.add_argument("--device-tag", default="model", help="Name for saved model")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    inc_df = load_incidents(args.incidents)
    df["hours_to_fail"] = calc_hours_to_fail(df, inc_df, max_lag_hours=args.lag_hours)
    df = update_target_by_htf(df, args.lag_hours)

    train_df, val_df = split_dataset(df, by_date=args.date_split)
    features = [c for c in train_df.columns if c not in (DEVICE_COL, TIME_COL, "hours_to_fail", TARGET_COL)]

    trainer = ModelTrainer("lgbm", n_trials=3, cv_splits=3)
    trainer.fit(train_df, features, df_test=val_df, device=args.device_tag)
    trainer.evaluate(val_df, features, tag="validation")


if __name__ == "__main__":
    main()
