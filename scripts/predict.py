import argparse
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from data_utils import DEVICE_COL, TIME_COL


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--data", type=Path, required=True, help="Test dataset CSV")
    parser.add_argument("--model", type=Path, required=True, help="Path to model file")
    parser.add_argument("--threshold", type=Path, required=True, help="Path to threshold txt")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV with predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    booster = lgb.Booster(model_file=str(args.model))
    thr = float(Path(args.threshold).read_text())
    proba = booster.predict(df.drop(columns=[DEVICE_COL, TIME_COL], errors="ignore"))
    preds = (proba >= thr).astype(int)
    df["y_pred"] = preds
    df[[DEVICE_COL, TIME_COL, "y_pred"]].to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
