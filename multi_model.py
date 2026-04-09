"""
nested_loocv.py  —  Nested Leave-One-Out Cross-Validation Regression
=====================================================================
Supported models (pass via --model):
  ols        Ordinary Least Squares         (no hyperparameter tuning)
  lasso      Lasso  L1 regularisation       (tunes: alpha)
  ridge      Ridge  L2 regularisation       (tunes: alpha)
  rf         Random Forest                  (tunes: n_estimators, max_depth)
  xgb        XGBoost                        (tunes: n_estimators, max_depth, learning_rate)

Usage examples
--------------
  python nested_loocv.py --model lasso
  python nested_loocv.py --model ridge  --data my_data.csv --target price
  python nested_loocv.py --model rf     --outer_cv 10
  python nested_loocv.py --model xgb    --outer_cv loocv  --n_jobs -1
  python nested_loocv.py --model ols    --data my_data.csv --target y --no_scale
"""

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    LeaveOneOut, KFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")   # suppress convergence noise during CV


# ── 1. Model Registry ──────────────────────────────────────────────────────────
#
# Each entry contains:
#   "estimator"    : a callable that returns a fresh unfitted estimator
#   "tune"         : True  → wrap in GridSearchCV for inner CV
#                    False → use estimator directly (OLS, or models with
#                            built-in CV like LassoCV / RidgeCV)
#   "param_grid"   : hyperparameter grid passed to GridSearchCV (if tune=True)
#   "needs_scale"  : whether StandardScaler should be applied by default
#   "inner_cv"     : "loocv" | int (k-fold) | None  (None = uses outer setting)
#
# To add a new model, add one entry here — nothing else needs to change.

def get_model_registry(n_jobs=1):
    alphas = np.logspace(-4, 2, 60)

    registry = {

        # ── Ordinary Least Squares ─────────────────────────────────────────
        "ols": {
            "estimator"   : lambda: LinearRegression(),
            "tune"        : False,
            "param_grid"  : None,
            "needs_scale" : True,
            "inner_cv"    : None,
            "label"       : "Ordinary Least Squares",
        },

        # ── Lasso (L1) ─────────────────────────────────────────────────────
        # LassoCV has a built-in warm-path alpha search — faster than GridSearchCV
        "lasso": {
            "estimator"   : lambda: LassoCV(
                                alphas=alphas,
                                cv=LeaveOneOut(),
                                max_iter=10_000,
                                n_jobs=n_jobs,
                            ),
            "tune"        : False,   # LassoCV tunes itself internally
            "param_grid"  : None,
            "needs_scale" : True,
            "inner_cv"    : None,
            "label"       : "Lasso (L1) with inner LOOCV",
        },

        # ── Ridge (L2) ─────────────────────────────────────────────────────
        # RidgeCV uses a closed-form LOOCV — extremely fast
        "ridge": {
            "estimator"   : lambda: RidgeCV(
                                alphas=alphas,
                                cv=LeaveOneOut(),
                            ),
            "tune"        : False,   # RidgeCV tunes itself internally
            "param_grid"  : None,
            "needs_scale" : True,
            "inner_cv"    : None,
            "label"       : "Ridge (L2) with inner LOOCV",
        },

        # ── Random Forest ──────────────────────────────────────────────────
        "rf": {
            "estimator"   : lambda: RandomForestRegressor(random_state=42),
            "tune"        : True,
            "param_grid"  : {
                "n_estimators" : [100, 300],
                "max_depth"    : [None, 5, 10],
                "max_features" : ["sqrt", 0.5],
            },
            "needs_scale" : False,   # tree models don't need scaling
            "inner_cv"    : 5,       # 5-fold inner CV (LOOCV would be very slow)
            "label"       : "Random Forest with inner 5-fold CV",
        },

        # ── XGBoost ────────────────────────────────────────────────────────
        "xgb": {
            "estimator"   : None,    # set at runtime after import check
            "tune"        : True,
            "param_grid"  : {
                "n_estimators"  : [100, 300],
                "max_depth"     : [3, 5],
                "learning_rate" : [0.05, 0.1, 0.2],
                "subsample"     : [0.8, 1.0],
            },
            "needs_scale" : False,
            "inner_cv"    : 5,
            "label"       : "XGBoost with inner 5-fold CV",
        },
    }
    return registry


# ── 2. Data loader ─────────────────────────────────────────────────────────────

def load_data(args):
    """Load X, y either from a CSV file or generate synthetic data."""
    if args.data:
        df = pd.read_csv(args.data)
        if args.target not in df.columns:
            sys.exit(f"[ERROR] Column '{args.target}' not found. "
                     f"Available: {list(df.columns)}")
        y = df[args.target].values
        X = df.drop(columns=[args.target]).values
        feature_names = [c for c in df.columns if c != args.target]
        print(f"[data]  Loaded '{args.data}'  →  {X.shape[0]} samples, "
              f"{X.shape[1]} features")
    else:
        print("[data]  No CSV provided — using synthetic data "
              "(50 samples, 10 features)")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        true_coef = rng.standard_normal(10)
        true_coef[5:] = 0          # last 5 features are noise
        y = X @ true_coef + rng.standard_normal(50) * 0.5
        feature_names = [f"x{i}" for i in range(10)]

    return X, y, feature_names


# ── 3. Outer CV factory ────────────────────────────────────────────────────────

def make_outer_cv(setting, n_samples):
    if setting == "loocv" or setting is None:
        print(f"[cv]    Outer CV: LOOCV  ({n_samples} folds)")
        return LeaveOneOut()
    k = int(setting)
    print(f"[cv]    Outer CV: {k}-fold KFold")
    return KFold(n_splits=k, shuffle=True, random_state=42)


# ── 4. Inner CV factory ────────────────────────────────────────────────────────

def make_inner_cv(model_cfg):
    setting = model_cfg["inner_cv"]
    if setting is None or setting == "loocv":
        return LeaveOneOut()
    return KFold(n_splits=int(setting), shuffle=True, random_state=0)


# ── 5. Main nested CV loop ─────────────────────────────────────────────────────

def run_nested_loocv(X, y, model_cfg, outer_cv, scale, n_jobs):
    y_true_list, y_pred_list = [], []
    tuned_params_list = []

    n = X.shape[0]
    splits = list(outer_cv.split(X))
    n_outer = len(splits)

    print(f"[run]   Starting {n_outer} outer folds …")
    t0 = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── Scaling (fit on train only) ──────────────────────────────────
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # ── Inner loop: hyperparameter tuning or self-tuning model ───────
        estimator = model_cfg["estimator"]()

        if model_cfg["tune"]:
            inner_cv  = make_inner_cv(model_cfg)
            estimator = GridSearchCV(
                estimator=estimator,
                param_grid=model_cfg["param_grid"],
                cv=inner_cv,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                refit=True,          # refit best model on full train fold
            )
            estimator.fit(X_train, y_train)
            tuned_params_list.append(estimator.best_params_)
            # GridSearchCV.predict() uses the refitted best estimator
        else:
            # OLS, LassoCV, RidgeCV — they tune (or don't need to) internally
            estimator.fit(X_train, y_train)
            # Extract tuned alpha if available
            alpha = getattr(estimator, "alpha_", None)
            if alpha is not None:
                tuned_params_list.append({"alpha": alpha})

        y_pred = estimator.predict(X_test)
        y_true_list.extend(y_test.tolist())
        y_pred_list.extend(y_pred.tolist())

        # Progress indicator for long runs
        if fold_i % max(1, n_outer // 10) == 0 or fold_i == n_outer:
            elapsed = time.time() - t0
            print(f"  fold {fold_i:4d}/{n_outer}  |  elapsed {elapsed:6.1f}s")

    return np.array(y_true_list), np.array(y_pred_list), tuned_params_list


# ── 6. Reporting ───────────────────────────────────────────────────────────────

def report(y_true, y_pred, tuned_params, model_label, feature_names):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"  Model  : {model_label}")
    print("=" * 60)
    print(f"  RMSE   : {rmse:.4f}")
    print(f"  MAE    : {mae:.4f}")
    print(f"  R²     : {r2:.4f}")

    if tuned_params:
        # Summarise tuned hyperparameters across folds
        param_df = pd.DataFrame(tuned_params)
        print("\n  Tuned hyperparameters (across outer folds):")
        for col in param_df.columns:
            vals = param_df[col]
            if vals.dtype == object or vals.nunique() <= 5:
                counts = vals.value_counts().to_dict()
                print(f"    {col:20s}: {counts}")
            else:
                print(f"    {col:20s}: "
                      f"mean={vals.mean():.4g}  "
                      f"std={vals.std():.4g}  "
                      f"[{vals.min():.4g}, {vals.max():.4g}]")

    # Per-sample results
    results = pd.DataFrame({
        "y_true"   : y_true,
        "y_pred"   : y_pred.round(4),
        "residual" : (y_true - y_pred).round(4),
    })
    print("\n  Per-sample predictions (first 10 rows):")
    print(results.head(10).to_string(index=True))
    print("=" * 60)

    return results


# ── 7. CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Nested LOOCV regression with selectable model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True,
        choices=["ols", "lasso", "ridge", "rf", "xgb"],
        help="Model type to train.",
    )
    p.add_argument(
        "--data", default=None,
        help="Path to a CSV file. If omitted, synthetic data is used.",
    )
    p.add_argument(
        "--target", default="y",
        help="Name of the target column in the CSV (default: 'y').",
    )
    p.add_argument(
        "--outer_cv", default="loocv",
        help="Outer CV strategy: 'loocv' or an integer k for k-fold (default: loocv).",
    )
    p.add_argument(
        "--no_scale", action="store_true",
        help="Disable StandardScaler even for models that request it.",
    )
    p.add_argument(
        "--n_jobs", type=int, default=1,
        help="Parallel jobs for GridSearchCV / LassoCV (default: 1; -1 = all cores).",
    )
    p.add_argument(
        "--output", default=None,
        help="Optional path to save per-sample results CSV.",
    )
    return p.parse_args()


# ── 8. Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── XGBoost optional import ──────────────────────────────────────────
    registry = get_model_registry(n_jobs=args.n_jobs)
    if args.model == "xgb":
        try:
            from xgboost import XGBRegressor
            registry["xgb"]["estimator"] = lambda: XGBRegressor(
                random_state=42,
                verbosity=0,
                n_jobs=args.n_jobs,
            )
        except ImportError:
            sys.exit("[ERROR] XGBoost not installed. Run: pip install xgboost")

    model_cfg = registry[args.model]
    print(f"\n[model] {model_cfg['label']}")

    # ── Load data ────────────────────────────────────────────────────────
    X, y, feature_names = load_data(args)

    # ── Scaling decision ─────────────────────────────────────────────────
    scale = model_cfg["needs_scale"] and not args.no_scale
    print(f"[scale] StandardScaler: {'ON' if scale else 'OFF'}")

    # ── Outer CV ─────────────────────────────────────────────────────────
    outer_cv = make_outer_cv(args.outer_cv, n_samples=X.shape[0])

    # ── Run ──────────────────────────────────────────────────────────────
    y_true, y_pred, tuned_params = run_nested_loocv(
        X, y, model_cfg, outer_cv, scale, n_jobs=args.n_jobs
    )

    # ── Report ───────────────────────────────────────────────────────────
    results = report(y_true, y_pred, tuned_params, model_cfg["label"], feature_names)

    # ── Optional save ────────────────────────────────────────────────────
    if args.output:
        results.to_csv(args.output, index=True)
        print(f"\n[save]  Results written to '{args.output}'")


if __name__ == "__main__":
    main()
