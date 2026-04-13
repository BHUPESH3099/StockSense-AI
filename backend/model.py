"""
model.py
--------
XGBoost-based stock price prediction models.
Trains separate regressors for 1-day, 1-week, and 1-month horizons.
Includes evaluation metrics, feature importance, and model persistence.
"""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from backend.feature_engineering import prepare_dataset, get_feature_columns

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

HORIZONS = {
    "1d": {"target": "target_1d", "label": "1 Day"},
    "1w": {"target": "target_5d", "label": "1 Week"},
    "1m": {"target": "target_21d", "label": "1 Month"},
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
}


class StockPredictor:
    """
    Multi-horizon XGBoost stock price predictor.
    Trains 3 models: 1d / 1w / 1m ahead.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.models: dict = {}
        self.scalers: dict = {}
        self.feature_cols: list = []
        self.metrics: dict = {}
        self.feature_importance: dict = {}
        self.trained = False

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train(self, raw_df: pd.DataFrame, test_size: float = 0.15) -> dict:
        """
        Train all three horizon models on the provided raw OHLCV data.

        Args:
            raw_df: Raw OHLCV DataFrame from data_fetcher
            test_size: Fraction of data to hold out for evaluation

        Returns:
            dict of evaluation metrics per horizon
        """
        df = prepare_dataset(raw_df)
        self.feature_cols = get_feature_columns(df)

        n = len(df)
        split_idx = int(n * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        for key, cfg in HORIZONS.items():
            target_col = cfg["target"]

            X_train = train_df[self.feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[self.feature_cols]
            y_test = test_df[target_col]

            # Scale features
            scaler = RobustScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Split for XGB early stopping eval set
            val_size = max(1, int(len(X_train_sc) * 0.1))
            X_tr = X_train_sc[:-val_size]
            y_tr = y_train.values[:-val_size]
            X_val = X_train_sc[-val_size:]
            y_val = y_train.values[-val_size:]

            model = XGBRegressor(**XGB_PARAMS)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate on test set
            preds = model.predict(X_test_sc)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            mape = np.mean(np.abs((y_test.values - preds) / (y_test.values + 1e-10))) * 100

            self.models[key] = model
            self.scalers[key] = scaler
            self.metrics[key] = {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
                "mape": round(mape, 2),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "label": cfg["label"],
            }
            self.feature_importance[key] = dict(
                zip(self.feature_cols, model.feature_importances_)
            )

        self.trained = True
        self._save()
        return self.metrics

    # ─────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────

    def predict(self, raw_df: pd.DataFrame) -> dict:
        """
        Predict future prices for all horizons.
        Uses the LAST row of the engineered feature set.

        Returns:
            dict: { '1d': price, '1w': price, '1m': price }
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        from backend.feature_engineering import engineer_features
        df = engineer_features(raw_df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        last_row = df[self.feature_cols].iloc[[-1]]
        predictions = {}

        for key in HORIZONS:
            scaler = self.scalers[key]
            model = self.models[key]
            X = scaler.transform(last_row)
            predictions[key] = float(model.predict(X)[0])

        return predictions

    def predict_historical(self, raw_df: pd.DataFrame, horizon: str = "1d") -> pd.DataFrame:
        """
        Generate in-sample + out-of-sample predictions for charting.
        Returns a DataFrame with Date, Actual, Predicted columns.
        """
        from backend.feature_engineering import engineer_features, get_feature_columns

        df = engineer_features(raw_df)
        target_col = HORIZONS[horizon]["target"]
        df[target_col] = raw_df["Close"].shift(-{"1d": 1, "1w": 5, "1m": 21}[horizon])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=self.feature_cols + [target_col], inplace=True)

        scaler = self.scalers[horizon]
        model = self.models[horizon]

        X = scaler.transform(df[self.feature_cols])
        preds = model.predict(X)

        result = pd.DataFrame({
            "Date": df.index,
            "Actual": df[target_col].values,
            "Predicted": preds,
        }).set_index("Date")
        return result

    def get_top_features(self, horizon: str = "1d", n: int = 15) -> pd.DataFrame:
        """Return top N most important features for a given horizon."""
        fi = self.feature_importance.get(horizon, {})
        s = pd.Series(fi).sort_values(ascending=False).head(n)
        return s.reset_index().rename(columns={"index": "Feature", 0: "Importance"})

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def _model_path(self) -> Path:
        return MODEL_DIR / f"{self.ticker}_predictor.joblib"

    def _save(self):
        payload = {
            "ticker": self.ticker,
            "models": self.models,
            "scalers": self.scalers,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }
        joblib.dump(payload, self._model_path())

    def load(self) -> bool:
        """Try to load a saved model. Returns True if successful."""
        path = self._model_path()
        if not path.exists():
            return False
        try:
            payload = joblib.load(path)
            self.ticker = payload["ticker"]
            self.models = payload["models"]
            self.scalers = payload["scalers"]
            self.feature_cols = payload["feature_cols"]
            self.metrics = payload["metrics"]
            self.feature_importance = payload["feature_importance"]
            self.trained = True
            return True
        except Exception:
            return False

    @classmethod
    def load_or_train(cls, ticker: str, raw_df: pd.DataFrame) -> "StockPredictor":
        """
        Load cached model if available, otherwise train fresh.
        Always retrain if cached model is more than 24h old.
        """
        predictor = cls(ticker)
        path = predictor._model_path()

        stale = True
        if path.exists():
            import time
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            stale = age_hours > 24

        if not stale and predictor.load():
            return predictor

        predictor.train(raw_df)
        return predictor
