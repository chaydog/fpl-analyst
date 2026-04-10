"""Per-position XGBoost + Random Forest ensemble for FPL points prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .features import FeatureBuilder

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


class PointsPredictor:
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[int, dict] = {}
        self.feature_cols: list[str] = []

    def train(self, features_df: pd.DataFrame, feature_cols: list[str]) -> dict:
        """Train per-position models. Returns evaluation metrics."""
        self.feature_cols = feature_cols
        metrics = {}

        for pos_id, pos_name in POSITIONS.items():
            pos_df = features_df[features_df["element_type"] == pos_id].copy()
            if len(pos_df) < 50:
                print(f"  Skipping {pos_name}: only {len(pos_df)} rows")
                continue

            # Temporal split: last 5 GWs for validation
            max_gw = pos_df["round"].max()
            val_gws = list(range(max_gw - 4, max_gw + 1))

            train_mask = ~pos_df["round"].isin(val_gws)
            val_mask = pos_df["round"].isin(val_gws)

            available_cols = [c for c in feature_cols if c in pos_df.columns]
            X_train = pos_df.loc[train_mask, available_cols].fillna(0)
            y_train = pos_df.loc[train_mask, "target"]
            X_val = pos_df.loc[val_mask, available_cols].fillna(0)
            y_val = pos_df.loc[val_mask, "target"]

            if len(X_train) < 30 or len(X_val) < 10:
                print(f"  Skipping {pos_name}: insufficient train/val split")
                continue

            # XGBoost
            xgb = XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42,
            )
            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            # Ensemble predictions
            xgb_pred = xgb.predict(X_val)
            rf_pred = rf.predict(X_val)
            ensemble_pred = 0.6 * xgb_pred + 0.4 * rf_pred

            mae = mean_absolute_error(y_val, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            corr = np.corrcoef(y_val, ensemble_pred)[0, 1] if len(y_val) > 1 else 0

            metrics[pos_name] = {"mae": round(mae, 3), "rmse": round(rmse, 3), "corr": round(corr, 3),
                                 "train_rows": len(X_train), "val_rows": len(X_val)}
            self.models[pos_id] = {"xgb": xgb, "rf": rf}

            print(f"  {pos_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, r={corr:.3f} "
                  f"(train={len(X_train)}, val={len(X_val)})")

        self.save()
        return metrics

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict expected points for each player."""
        predictions = pd.Series(0.0, index=features_df.index)
        available_cols = [c for c in self.feature_cols if c in features_df.columns]

        for pos_id, model_dict in self.models.items():
            mask = features_df["element_type"] == pos_id
            if not mask.any():
                continue

            X = features_df.loc[mask, available_cols].fillna(0)
            xgb_pred = model_dict["xgb"].predict(X)
            rf_pred = model_dict["rf"].predict(X)
            predictions.loc[mask] = 0.6 * xgb_pred + 0.4 * rf_pred

        # Multiply by availability chance
        if "chance_of_playing" in features_df.columns:
            predictions *= features_df["chance_of_playing"]
        if "has_fixture" in features_df.columns:
            predictions *= features_df["has_fixture"]

        return predictions.clip(lower=0)

    def feature_importance(self, position: int) -> pd.DataFrame:
        if position not in self.models:
            return pd.DataFrame()

        xgb = self.models[position]["xgb"]
        importance = xgb.feature_importances_
        available_cols = [c for c in self.feature_cols]
        return (
            pd.DataFrame({"feature": available_cols[:len(importance)], "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self):
        with open(self.model_dir / "models.pkl", "wb") as f:
            pickle.dump({"models": self.models, "feature_cols": self.feature_cols}, f)

    def load(self):
        path = self.model_dir / "models.pkl"
        if not path.exists():
            raise FileNotFoundError("No trained models found. Run 'python main.py train' first.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.feature_cols = data["feature_cols"]
