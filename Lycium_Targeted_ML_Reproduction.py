# -*- coding: utf-8 -*-
"""Classify Lycium leaf origin and cultivar from targeted LC-MS/MS data.

Technical replicate pairs are averaged before repeated nested cross-validation
(10 fixed outer partitions, five outer folds and four inner folds). Imputation
and scaling are fitted on the training folds.

Consensus predictions average each sample's out-of-fold probabilities across
partitions. Stratified bootstrap intervals describe sample-level uncertainty;
SD and range across partitions describe prediction stability. Neither replaces
external validation. The optional full-data SHAP analysis is descriptive and
does not assess cross-validation ranking stability.

Dependencies: numpy, pandas, matplotlib, scikit-learn, openpyxl;
xgboost for XGBoost models and shap for the optional SHAP analysis.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings("ignore")


# Input files and analysis settings

# Input paths can also be supplied with --origin and --cultivar.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
ORIGIN_FILE = ROOT / "Lycium_Origin_Data.csv"
CULTIVAR_FILE = ROOT / "Lycium_Cultivar_Data.csv"
OUTPUT_DIR = ROOT / "reproduction_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults overridden by command-line arguments.
RUN_FIGURES = True
RUN_SHAP = False
SHAP_MAX_SAMPLES = 20
# Fixed seeds for the outer CV partitions.
OUTER_SEEDS = [
    71477,
    28804,
    73432,
    43747,
    53980,
    4900,
    96187,
    67754,
    77483,
    34335,
]

N_SPLITS = 5
INNER_SPLITS = 4
INNER_SCORING = "roc_auc_ovr"

# Keep tree estimators single-threaded when parallelizing the grid search.
N_JOBS_INNER = -1
RF_N_JOBS = 1
XGB_N_JOBS = 1

# Bootstrap biological samples from the consensus out-of-fold predictions.
CI_LEVEL = 0.95
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261030
SAVE_BOOTSTRAP_DRAWS = True

# Figure export
PNG_DPI = 600
CM = 1 / 2.54


# Figure style

ARIAL_PATHS = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\Arial.ttf"),
]

for fp in ARIAL_PATHS:
    if fp.exists():
        try:
            font_manager.fontManager.addfont(str(fp))
        except Exception:
            pass
        break

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.1,
    "savefig.dpi": PNG_DPI,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_AUC = "#4E79A7"
COL_ACC = "#B8B8B8"
COL_SEN = "#E15759"
COL_SPE = "#59A14F"
COL_RF_BLUE = "#2F6C99"
COL_CULTIVAR = "#A36A43"
CMAP = "Blues"

MODEL_COLORS = {
    "RF": "#2F6C99",
    "XGBoost": "#4E79A7",
    "SVM": "#59A14F",
    "MLP": "#F28E2B",
    "Logistic Regression": "#9C755F",
    "GaussianNB": "#BAB0AC",
    "KNN": "#E15759",
    "AdaBoost": "#B07AA1",
}

OVR_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#9C755F",
    "#B07AA1",
]


# Data preparation

def prepare_dataset(path: Path, label_col: str):
    """Average technical replicate pairs; leave imputation to model pipelines."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df = df.dropna(how="all").copy()
    df = df[df["NAME"].notna()].copy()

    rename_map = {
        "Lyciumins A": "Lyciumin A",
        "Lyciumins B": "Lyciumin B",
        "N-trans-Feruloyltyramine": "N-trans-feruloyltyramine",
        "N-trans-Feruloyl-3-methoxytyramine": "N-trans-feruloyl-3-methoxytyramine",
    }
    df = df.rename(columns=rename_map)

    names = df["NAME"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["_BIO_ID"] = names.str[:-1]

    # Select the quantified compounds; the CSVs also contain UV fingerprints.
    feature_cols = [
        "Choline", "Betaine", "Trigonelline", "p-Hydroxyphenethylamine",
        "Tryptamine", "Caffeoylputrescine", "Feruloylputrescine",
        "p-Coumaroyltyramine", "N-trans-feruloyltyramine",
        "N-trans-feruloyl-3-methoxytyramine", "Lyciumin A", "Lyciumin B",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing targeted compounds in {path.name}: {missing}")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("_BIO_ID", sort=False)

    X = (
        grouped[feature_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )

    y = (
        grouped[label_col]
        .first()
        .reset_index(drop=True)
    )

    bio_ids = pd.Series(
        list(grouped.groups.keys()),
        name="BiologicalSampleID",
        dtype="object",
    )

    if len(X) != 125:
        raise ValueError(f"Expected 125 biological samples, found {len(X)} in {path.name}")
    if y.isna().any():
        raise ValueError(f"Missing labels in {path.name}")
    return X, y, feature_cols, bio_ids


# Classifiers and parameter grids

MODEL_ORDER = [
    "RF",
    "XGBoost",
    "SVM",
    "MLP",
    "Logistic Regression",
    "GaussianNB",
    "KNN",
    "AdaBoost",
]


def build_model(name: str, seed: int, n_classes: int):
    """Build a classifier with fold-specific imputation and scaling as needed."""

    if name == "RF":
        clf = RandomForestClassifier(
            random_state=seed,
            n_jobs=RF_N_JOBS,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    if name == "XGBoost":
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=XGB_N_JOBS,
            verbosity=0,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    if name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ])

    if name == "MLP":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=2500, random_state=seed)),
        ])

    if name == "Logistic Regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ])

    if name == "GaussianNB":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ])

    if name == "KNN":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ])

    if name == "AdaBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", AdaBoostClassifier(random_state=seed)),
        ])

    raise ValueError(f"Unknown model: {name}")


def get_param_grid(name: str):
    """Hyperparameter grids for inner cross-validation."""
    grids = {
        "RF": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 8],
            "clf__min_samples_leaf": [1, 2],
        },
        "XGBoost": {
            "clf__n_estimators": [120, 240],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05, 0.10],
            "clf__subsample": [0.90],
            "clf__colsample_bytree": [0.90],
        },
        "SVM": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(32,), (64,)],
            "clf__alpha": [1e-4, 1e-3],
        },
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0],
        },
        "GaussianNB": {
            "clf__var_smoothing": [1e-10, 1e-9, 1e-8],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        },
        "AdaBoost": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.1, 0.5],
        },
    }
    return grids[name]


# Performance metrics

def safe_div(num, den):
    return float(num / den) if den > 0 else np.nan


def class_specific_metrics(y_true_idx, y_pred_idx, y_proba, classes):
    """One-vs-rest metrics for every class."""
    rows = []

    for class_i, class_name in enumerate(classes):
        y_true_bin = (y_true_idx == class_i).astype(int)
        y_pred_bin = (y_pred_idx == class_i).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            labels=[0, 1],
        ).ravel()

        try:
            auc_i = roc_auc_score(y_true_bin, y_proba[:, class_i])
        except ValueError:
            auc_i = np.nan

        sensitivity = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        precision = safe_div(tp, tp + fp)
        npv = safe_div(tn, tn + fn)
        f1_i = safe_div(2 * tp, 2 * tp + fp + fn)
        balanced_accuracy = np.nanmean([sensitivity, specificity])

        rows.append({
            "Class": str(class_name),
            "AUC": auc_i,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "NPV": npv,
            "F1": f1_i,
            "BalancedAccuracy": balanced_accuracy,
            "Support": int(np.sum(y_true_bin)),
        })

    return pd.DataFrame(rows)


def macro_sensitivity_specificity(y_true_idx, y_pred_idx, classes_idx):
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=classes_idx)
    sens = []
    spec = []

    for i in range(len(classes_idx)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(safe_div(tp, tp + fn))
        spec.append(safe_div(tn, tn + fp))

    return float(np.nanmean(sens)), float(np.nanmean(spec)), cm


def calculate_overall_metrics(y_true_idx, y_proba):
    """Overall multiclass metrics from probabilities using argmax classification."""
    y_pred_idx = np.argmax(y_proba, axis=1)
    class_idx = np.arange(y_proba.shape[1])

    macro_auc = roc_auc_score(
        y_true_idx,
        y_proba,
        multi_class="ovr",
        average="macro",
    )
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    macro_precision = precision_score(
        y_true_idx,
        y_pred_idx,
        average="macro",
        zero_division=0,
    )
    sensitivity, specificity, _ = macro_sensitivity_specificity(
        y_true_idx,
        y_pred_idx,
        class_idx,
    )

    return {
        "Macro-AUC": float(macro_auc),
        "Accuracy": float(accuracy),
        "Macro-F1": float(macro_f1),
        "Macro-Precision": float(macro_precision),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
    }


def percentile_ci(values, confidence=CI_LEVEL):
    """Return a two-sided percentile interval from bootstrap draws."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    lo, hi = np.quantile(x, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(max(0.0, lo)), float(min(1.0, hi))


def split_stability(values):
    """Summarize variability across CV partitions as mean, SD and range."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    return mean, sd, float(np.min(x)), float(np.max(x)), int(len(x))


def make_stratified_bootstrap_indices(y_true_idx, n_boot, seed):
    """Resample biological samples within classes, preserving class counts."""
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y_true_idx == c)[0] for c in np.unique(y_true_idx)]

    out = np.empty((n_boot, len(y_true_idx)), dtype=np.int32)
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True)
            for ix in class_indices
        ])
        # Mix class blocks while retaining the stratified sample counts.
        rng.shuffle(idx)
        out[b] = idx
    return out

# Repeated nested cross-validation

def evaluate_task(X, y, bio_ids, task_name):
    le = LabelEncoder()
    y_idx = le.fit_transform(y.astype(str))
    classes = le.classes_
    class_idx = np.arange(len(classes))
    Xv = X.to_numpy(dtype=float)
    bio_ids = np.asarray(bio_ids.astype(str))

    metric_rows = []
    class_metric_rows = []
    probability_rows = []
    best_param_rows = []
    fold_assignment_rows = []

    for model_name in MODEL_ORDER:
        print(f"[{task_name}] nested CV: {model_name}")

        for split_no, outer_seed in enumerate(OUTER_SEEDS, start=1):
            print(f"  Split {split_no:02d}/{len(OUTER_SEEDS)} | seed={outer_seed}")

            outer_cv = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=outer_seed,
            )

            repeat_proba = np.full(
                (len(y_idx), len(classes)),
                np.nan,
                dtype=float,
            )
            repeat_pred = np.full(len(y_idx), -1, dtype=int)
            repeat_fold = np.full(len(y_idx), -1, dtype=int)

            for outer_fold, (train_idx, test_idx) in enumerate(
                outer_cv.split(Xv, y_idx),
                start=1,
            ):
                inner_seed = int(outer_seed * 100 + outer_fold)

                inner_cv = StratifiedKFold(
                    n_splits=INNER_SPLITS,
                    shuffle=True,
                    random_state=inner_seed,
                )

                pipeline = build_model(
                    model_name,
                    seed=inner_seed,
                    n_classes=len(classes),
                )

                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=get_param_grid(model_name),
                    scoring=INNER_SCORING,
                    cv=inner_cv,
                    refit=True,
                    n_jobs=N_JOBS_INNER,
                    return_train_score=False,
                    error_score="raise",
                )

                search.fit(Xv[train_idx], y_idx[train_idx])

                fold_proba = search.best_estimator_.predict_proba(Xv[test_idx])
                fold_pred = np.argmax(fold_proba, axis=1)

                repeat_proba[test_idx, :] = fold_proba
                repeat_pred[test_idx] = fold_pred
                repeat_fold[test_idx] = outer_fold

                best_param_rows.append({
                    "Task": task_name,
                    "Model": model_name,
                    "SplitNo": split_no,
                    "OuterSeed": outer_seed,
                    "OuterFold": outer_fold,
                    "OuterTrainN": len(train_idx),
                    "OuterTestN": len(test_idx),
                    "InnerBestMacroAUC": float(search.best_score_),
                    "BestParams": json.dumps(search.best_params_, ensure_ascii=False),
                })

                for local_i, sample_idx in enumerate(test_idx):
                    row = {
                        "Task": task_name,
                        "Model": model_name,
                        "SplitNo": split_no,
                        "OuterSeed": outer_seed,
                        "OuterFold": outer_fold,
                        "SampleIndex": int(sample_idx),
                        "BiologicalSampleID": bio_ids[sample_idx],
                        "TrueClass": str(classes[y_idx[sample_idx]]),
                        "PredictedClass": str(classes[fold_pred[local_i]]),
                    }
                    for class_i, class_name in enumerate(classes):
                        row[f"Probability_{class_name}"] = float(
                            fold_proba[local_i, class_i]
                        )
                    probability_rows.append(row)

            if np.isnan(repeat_proba).any():
                raise RuntimeError(
                    f"Missing outer-test probabilities for {task_name}, "
                    f"{model_name}, split {split_no}."
                )

            if (repeat_pred < 0).any():
                raise RuntimeError(
                    f"Missing outer-test predictions for {task_name}, "
                    f"{model_name}, split {split_no}."
                )

            # Save fold assignment once per task/split; model-independent.
            if model_name == MODEL_ORDER[0]:
                for sample_idx in range(len(y_idx)):
                    fold_assignment_rows.append({
                        "Task": task_name,
                        "SplitNo": split_no,
                        "OuterSeed": outer_seed,
                        "SampleIndex": sample_idx,
                        "BiologicalSampleID": bio_ids[sample_idx],
                        "Class": str(classes[y_idx[sample_idx]]),
                        "OuterFold": int(repeat_fold[sample_idx]),
                    })

            macro_auc = roc_auc_score(
                y_idx,
                repeat_proba,
                multi_class="ovr",
                average="macro",
            )
            acc = accuracy_score(y_idx, repeat_pred)
            macro_f1 = f1_score(y_idx, repeat_pred, average="macro")
            macro_precision = precision_score(
                y_idx,
                repeat_pred,
                average="macro",
                zero_division=0,
            )
            sensitivity, specificity, _ = macro_sensitivity_specificity(
                y_idx,
                repeat_pred,
                class_idx,
            )

            metric_rows.append({
                "Task": task_name,
                "Model": model_name,
                "SplitNo": split_no,
                "OuterSeed": outer_seed,
                "Macro-AUC": float(macro_auc),
                "Accuracy": float(acc),
                "Macro-F1": float(macro_f1),
                "Macro-Precision": float(macro_precision),
                "Sensitivity": float(sensitivity),
                "Specificity": float(specificity),
            })

            cls_df = class_specific_metrics(
                y_idx,
                repeat_pred,
                repeat_proba,
                classes,
            )
            cls_df.insert(0, "OuterSeed", outer_seed)
            cls_df.insert(0, "SplitNo", split_no)
            cls_df.insert(0, "Model", model_name)
            cls_df.insert(0, "Task", task_name)
            class_metric_rows.extend(cls_df.to_dict("records"))

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(class_metric_rows),
        pd.DataFrame(probability_rows),
        pd.DataFrame(best_param_rows),
        pd.DataFrame(fold_assignment_rows),
        classes,
    )


# Consensus out-of-fold predictions

def build_consensus_predictions(probability_df, task_name, model_name):
    sub = probability_df[
        (probability_df["Task"] == task_name)
        & (probability_df["Model"] == model_name)
    ].copy()

    prob_cols = [
        c for c in sub.columns
        if c.startswith("Probability_") and sub[c].notna().any()
    ]
    class_names = [c.replace("Probability_", "", 1) for c in prob_cols]

    grouped = (
        sub.groupby(
            ["SampleIndex", "BiologicalSampleID", "TrueClass"],
            as_index=False,
        )[prob_cols]
        .mean()
        .sort_values("SampleIndex")
        .reset_index(drop=True)
    )

    p = grouped[prob_cols].to_numpy(float)
    pred_idx = np.argmax(p, axis=1)
    grouped["ConsensusPredictedClass"] = [class_names[i] for i in pred_idx]

    return grouped, class_names


# Bootstrap intervals and prediction stability

def build_sample_prediction_stability(probability_df):
    """Summarize per-sample probability variability and class agreement."""
    output_rows = []

    for (task_name, model_name), sub in probability_df.groupby(["Task", "Model"], sort=False):
        prob_cols = [
            c for c in sub.columns
            if c.startswith("Probability_") and sub[c].notna().any()
        ]
        class_names = [c.replace("Probability_", "", 1) for c in prob_cols]

        for keys, g in sub.groupby(
            ["SampleIndex", "BiologicalSampleID", "TrueClass"],
            sort=False,
        ):
            sample_idx, bio_id, true_class = keys
            means = g[prob_cols].mean()
            consensus_idx = int(np.argmax(means.to_numpy(float)))
            consensus_class = class_names[consensus_idx]
            agreement = float(np.mean(g["PredictedClass"].astype(str) == consensus_class))

            row = {
                "Task": task_name,
                "Model": model_name,
                "SampleIndex": int(sample_idx),
                "BiologicalSampleID": str(bio_id),
                "TrueClass": str(true_class),
                "ConsensusPredictedClass": consensus_class,
                "SplitPredictionAgreement": agreement,
                "NumberOfSplits": int(len(g)),
            }

            for col, class_name in zip(prob_cols, class_names):
                vals = g[col].to_numpy(float)
                row[f"Probability_{class_name}_Mean"] = float(np.mean(vals))
                row[f"Probability_{class_name}_SD"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                row[f"Probability_{class_name}_Min"] = float(np.min(vals))
                row[f"Probability_{class_name}_Max"] = float(np.max(vals))

            true_col = f"Probability_{true_class}"
            if true_col in g.columns:
                true_vals = g[true_col].to_numpy(float)
                row["TrueClassProbability_Mean"] = float(np.mean(true_vals))
                row["TrueClassProbability_SD"] = float(np.std(true_vals, ddof=1)) if len(true_vals) > 1 else np.nan
                row["TrueClassProbability_Min"] = float(np.min(true_vals))
                row["TrueClassProbability_Max"] = float(np.max(true_vals))

            output_rows.append(row)

    return pd.DataFrame(output_rows)


def bootstrap_uncertainty_for_model(
    consensus,
    class_names,
    bootstrap_indices,
    task_name,
    model_name,
):
    """Bootstrap overall and class-specific metrics from consensus OOF predictions."""
    prob_cols = [f"Probability_{c}" for c in class_names]
    p = consensus[prob_cols].to_numpy(float)
    y_true_idx = pd.Categorical(
        consensus["TrueClass"],
        categories=class_names,
    ).codes

    point_overall = calculate_overall_metrics(y_true_idx, p)
    point_pred = np.argmax(p, axis=1)
    point_class = class_specific_metrics(
        y_true_idx,
        point_pred,
        p,
        class_names,
    )

    overall_draws = []
    class_draws = []

    for b, idx in enumerate(bootstrap_indices, start=1):
        yb = y_true_idx[idx]
        pb = p[idx]
        pred_b = np.argmax(pb, axis=1)

        om = calculate_overall_metrics(yb, pb)
        om.update({
            "Task": task_name,
            "Model": model_name,
            "BootstrapReplicate": b,
        })
        overall_draws.append(om)

        cm = class_specific_metrics(yb, pred_b, pb, class_names)
        cm.insert(0, "BootstrapReplicate", b)
        cm.insert(0, "Model", model_name)
        cm.insert(0, "Task", task_name)
        class_draws.extend(cm.to_dict("records"))

    return (
        point_overall,
        point_class,
        pd.DataFrame(overall_draws),
        pd.DataFrame(class_draws),
    )


def build_bootstrap_summaries(
    probability_df,
    split_overall_df,
    split_class_df,
):
    """Combine consensus bootstrap intervals with CV partition stability."""
    overall_rows = []
    class_rows = []
    all_overall_draws = []
    all_class_draws = []

    for task_no, task_name in enumerate(["Origin", "Cultivar"]):
        # Share the same bootstrap resamples across models within each task.
        reference_consensus, reference_classes = build_consensus_predictions(
            probability_df,
            task_name,
            MODEL_ORDER[0],
        )
        y_ref = pd.Categorical(
            reference_consensus["TrueClass"],
            categories=reference_classes,
        ).codes
        boot_idx = make_stratified_bootstrap_indices(
            y_ref,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED + task_no,
        )

        reference_ids = reference_consensus["BiologicalSampleID"].astype(str).tolist()

        for model_name in MODEL_ORDER:
            consensus, class_names = build_consensus_predictions(
                probability_df,
                task_name,
                model_name,
            )

            ids = consensus["BiologicalSampleID"].astype(str).tolist()
            if ids != reference_ids:
                raise RuntimeError(
                    f"Consensus sample ordering differs across models for {task_name}."
                )
            if list(class_names) != list(reference_classes):
                raise RuntimeError(
                    f"Class ordering differs across models for {task_name}."
                )

            point_overall, point_class, overall_draws, class_draws = bootstrap_uncertainty_for_model(
                consensus,
                class_names,
                boot_idx,
                task_name,
                model_name,
            )

            split_sub = split_overall_df[
                (split_overall_df["Task"] == task_name)
                & (split_overall_df["Model"] == model_name)
            ]

            row = {
                "Task": task_name,
                "Model": model_name,
                "CI Method": (
                    f"Stratified biological-sample percentile bootstrap; "
                    f"B={BOOTSTRAP_REPS}; consensus of 10 outer-OOF predictions"
                ),
            }

            for metric in [
                "Macro-AUC",
                "Accuracy",
                "Macro-F1",
                "Macro-Precision",
                "Sensitivity",
                "Specificity",
            ]:
                estimate = float(point_overall[metric])
                lo, hi = percentile_ci(overall_draws[metric].to_numpy(float))
                sm, ssd, smin, smax, sn = split_stability(split_sub[metric].to_numpy(float))

                row[f"{metric} Estimate"] = estimate
                row[f"{metric} 95%CI Low"] = lo
                row[f"{metric} 95%CI High"] = hi
                row[f"{metric} Split Mean"] = sm
                row[f"{metric} Split SD"] = ssd
                row[f"{metric} Split Min"] = smin
                row[f"{metric} Split Max"] = smax
                row[f"{metric} Split N"] = sn

            overall_rows.append(row)

            split_class_sub = split_class_df[
                (split_class_df["Task"] == task_name)
                & (split_class_df["Model"] == model_name)
            ]

            for _, point_row in point_class.iterrows():
                class_name = str(point_row["Class"])
                boot_cls = class_draws[class_draws["Class"].astype(str) == class_name]
                split_cls = split_class_sub[split_class_sub["Class"].astype(str) == class_name]

                c_row = {
                    "Task": task_name,
                    "Model": model_name,
                    "Class": class_name,
                    "Support": int(point_row["Support"]),
                    "CI Method": (
                        f"Stratified biological-sample percentile bootstrap; "
                        f"B={BOOTSTRAP_REPS}; consensus of 10 outer-OOF predictions"
                    ),
                }

                for metric in [
                    "AUC",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "NPV",
                    "F1",
                    "BalancedAccuracy",
                ]:
                    estimate = float(point_row[metric])
                    lo, hi = percentile_ci(boot_cls[metric].to_numpy(float))
                    sm, ssd, smin, smax, sn = split_stability(split_cls[metric].to_numpy(float))

                    c_row[f"{metric} Estimate"] = estimate
                    c_row[f"{metric} 95%CI Low"] = lo
                    c_row[f"{metric} 95%CI High"] = hi
                    c_row[f"{metric} Split Mean"] = sm
                    c_row[f"{metric} Split SD"] = ssd
                    c_row[f"{metric} Split Min"] = smin
                    c_row[f"{metric} Split Max"] = smax
                    c_row[f"{metric} Split N"] = sn

                class_rows.append(c_row)

            if SAVE_BOOTSTRAP_DRAWS:
                all_overall_draws.append(overall_draws)
                all_class_draws.append(class_draws)

    overall_summary = pd.DataFrame(overall_rows)
    class_summary = pd.DataFrame(class_rows)
    overall_draw_df = (
        pd.concat(all_overall_draws, ignore_index=True)
        if all_overall_draws else pd.DataFrame()
    )
    class_draw_df = (
        pd.concat(all_class_draws, ignore_index=True)
        if all_class_draws else pd.DataFrame()
    )

    return overall_summary, class_summary, overall_draw_df, class_draw_df

# Plotting

def save_figure(fig, stem):
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def style_cartesian_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def plot_metrics_bar(summary_df, task_name):
    """Plot consensus metrics with biological-sample bootstrap intervals."""
    sub = summary_df[summary_df["Task"] == task_name].copy()
    sub["Model"] = pd.Categorical(
        sub["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    sub = sub.sort_values("Model")

    y = np.arange(len(sub))
    h = 0.16
    fig, ax = plt.subplots(figsize=(11.7 * CM, 6.0 * CM))

    specs = [
        ("Macro-AUC", COL_AUC, "AUC", -1.5),
        ("Accuracy", COL_ACC, "Accuracy", -0.5),
        ("Sensitivity", COL_SEN, "Sensitivity", 0.5),
        ("Specificity", COL_SPE, "Specificity", 1.5),
    ]

    for metric, color, label, offset in specs:
        mean_col = f"{metric} Estimate"
        lo_col = f"{metric} 95%CI Low"
        hi_col = f"{metric} 95%CI High"
        means = sub[mean_col].to_numpy(float)
        lo = sub[lo_col].to_numpy(float)
        hi = sub[hi_col].to_numpy(float)
        xerr = np.vstack([means - lo, hi - means])

        ax.barh(
            y + offset * h,
            means,
            xerr=xerr,
            height=h,
            capsize=2,
            color=color,
            edgecolor="none",
            label=label,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sub["Model"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Metric value")
    style_cartesian_axis(ax)
    ax.legend(
        frameon=False,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    save_figure(fig, f"{task_name.lower()}_metrics_bootstrap95CI")


def plot_auc_ci_forest(summary_df, task_name):
    """Plot each model's macro-AUC and bootstrap interval."""
    sub = summary_df[summary_df["Task"] == task_name].copy()
    sub["Model"] = pd.Categorical(
        sub["Model"], categories=MODEL_ORDER, ordered=True
    )
    sub = sub.sort_values("Model")

    y = np.arange(len(sub))
    means = sub["Macro-AUC Estimate"].to_numpy(float)
    lo = sub["Macro-AUC 95%CI Low"].to_numpy(float)
    hi = sub["Macro-AUC 95%CI High"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.1 * CM, 6.5 * CM))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lo, hi - means]),
        fmt="o",
        markersize=4.5,
        capsize=2.5,
        linewidth=1.0,
        color=COL_AUC,
        ecolor=COL_AUC,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["Model"])
    ax.invert_yaxis()

    xmin = max(0.5, float(np.nanmin(lo)) - 0.02)
    ax.set_xlim(xmin, 1.005)
    ax.set_xlabel("Macro-AUC (consensus OOF estimate and 95% bootstrap CI)")
    style_cartesian_axis(ax)
    save_figure(fig, f"{task_name.lower()}_macroAUC_bootstrap95CI_forest")


def plot_class_auc_ci(class_summary_df, task_name, model_name):
    sub = class_summary_df[
        (class_summary_df["Task"] == task_name)
        & (class_summary_df["Model"] == model_name)
    ].copy()

    y = np.arange(len(sub))
    means = sub["AUC Estimate"].to_numpy(float)
    lo = sub["AUC 95%CI Low"].to_numpy(float)
    hi = sub["AUC 95%CI High"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.1 * CM, max(5.5, 0.75 * len(sub) + 2.2) * CM))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lo, hi - means]),
        fmt="o",
        markersize=4.2,
        capsize=2.5,
        linewidth=1.0,
        color=MODEL_COLORS[model_name],
        ecolor=MODEL_COLORS[model_name],
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["Class"])
    ax.invert_yaxis()
    xmin = max(0.5, float(np.nanmin(lo)) - 0.02)
    ax.set_xlim(xmin, 1.005)
    ax.set_xlabel("Class-specific AUC (consensus OOF estimate and 95% bootstrap CI)")
    ax.set_title(f"{task_name}: {model_name}", pad=3)
    style_cartesian_axis(ax)
    stem_model = model_name.lower().replace(" ", "_")
    save_figure(
        fig,
        f"{task_name.lower()}_{stem_model}_classAUC_bootstrap95CI",
    )


def plot_consensus_confusion(probability_df, task_name, model_name):
    consensus, class_names = build_consensus_predictions(
        probability_df,
        task_name,
        model_name,
    )

    true_idx = pd.Categorical(
        consensus["TrueClass"],
        categories=class_names,
    ).codes
    pred_idx = pd.Categorical(
        consensus["ConsensusPredictedClass"],
        categories=class_names,
    ).codes

    cm = confusion_matrix(
        true_idx,
        pred_idx,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(6.2 * CM, 6.0 * CM))
    ax.imshow(cm, cmap=CMAP, vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(f"{task_name}: {model_name}", pad=3)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() * 0.55 else "#1F1F1F"
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color=text_color,
                fontsize=8,
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4D4D4D")

    stem_model = model_name.lower().replace(" ", "_")
    save_figure(
        fig,
        f"{task_name.lower()}_consensus_confusion_{stem_model}_10splits",
    )


def macro_roc_curve(y_true_idx, y_proba, n_classes):
    y_bin = label_binarize(y_true_idx, classes=np.arange(n_classes))
    fprs = []
    tprs = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        fprs.append(fpr)
        tprs.append(tpr)

    all_fpr = np.unique(np.concatenate(fprs))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def plot_all_models_consensus_macro_roc(probability_df, summary_df, task_name):
    fig, ax = plt.subplots(figsize=(8.1 * CM, 6.9 * CM))

    for model_name in MODEL_ORDER:
        consensus, class_names = build_consensus_predictions(
            probability_df,
            task_name,
            model_name,
        )
        prob_cols = [f"Probability_{c}" for c in class_names]
        p = consensus[prob_cols].to_numpy(float)
        y_true_idx = pd.Categorical(
            consensus["TrueClass"],
            categories=class_names,
        ).codes

        fpr, tpr, _ = macro_roc_curve(
            y_true_idx,
            p,
            len(class_names),
        )

        row = summary_df[
            (summary_df["Task"] == task_name)
            & (summary_df["Model"] == model_name)
        ].iloc[0]

        mean_auc = row["Macro-AUC Estimate"]
        lo = row["Macro-AUC 95%CI Low"]
        hi = row["Macro-AUC 95%CI High"]

        ax.plot(
            fpr,
            tpr,
            linewidth=1.1,
            color=MODEL_COLORS[model_name],
            label=f"{model_name} {mean_auc:.3f} ({lo:.3f}-{hi:.3f})",
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle=":",
        linewidth=0.9,
        color="#6E6E6E",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    style_cartesian_axis(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    save_figure(
        fig,
        f"{task_name.lower()}_all_models_consensus_macroROC_bootstrapCI",
    )


def plot_model_class_ovr_roc(
    probability_df,
    class_summary_df,
    task_name,
    model_name,
):
    consensus, class_names = build_consensus_predictions(
        probability_df,
        task_name,
        model_name,
    )
    prob_cols = [f"Probability_{c}" for c in class_names]
    p = consensus[prob_cols].to_numpy(float)
    y_true_idx = pd.Categorical(
        consensus["TrueClass"],
        categories=class_names,
    ).codes
    y_bin = label_binarize(y_true_idx, classes=np.arange(len(class_names)))

    fig, ax = plt.subplots(figsize=(8.1 * CM, 6.9 * CM))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], p[:, i])
        row = class_summary_df[
            (class_summary_df["Task"] == task_name)
            & (class_summary_df["Model"] == model_name)
            & (class_summary_df["Class"] == class_name)
        ].iloc[0]

        mean_auc = row["AUC Estimate"]
        lo = row["AUC 95%CI Low"]
        hi = row["AUC 95%CI High"]

        ax.plot(
            fpr,
            tpr,
            linewidth=1.1,
            color=OVR_COLORS[i % len(OVR_COLORS)],
            label=f"{class_name} {mean_auc:.3f} ({lo:.3f}-{hi:.3f})",
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle=":",
        linewidth=0.9,
        color="#6E6E6E",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    style_cartesian_axis(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)

    stem_model = model_name.lower().replace(" ", "_")
    save_figure(
        fig,
        f"{task_name.lower()}_{stem_model}_classOVR_ROC_bootstrapCI",
    )


# Result tables

def format_estimate_ci_stability(row, metric):
    estimate = row[f"{metric} Estimate"]
    lo = row[f"{metric} 95%CI Low"]
    hi = row[f"{metric} 95%CI High"]
    split_mean = row[f"{metric} Split Mean"]
    split_sd = row[f"{metric} Split SD"]
    split_min = row[f"{metric} Split Min"]
    split_max = row[f"{metric} Split Max"]
    return (
        f"{estimate:.3f} ({lo:.3f}-{hi:.3f}); "
        f"split {split_mean:.3f} ± {split_sd:.3f}, "
        f"range {split_min:.3f}-{split_max:.3f}"
    )


def build_formatted_overall_table(summary_df):
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            "Classification task": row["Task"],
            "Model": row["Model"],
            "Macro-AUC, estimate (95% bootstrap CI); split stability": format_estimate_ci_stability(row, "Macro-AUC"),
            "Accuracy, estimate (95% bootstrap CI); split stability": format_estimate_ci_stability(row, "Accuracy"),
            "Macro-F1, estimate (95% bootstrap CI); split stability": format_estimate_ci_stability(row, "Macro-F1"),
            "Sensitivity, estimate (95% bootstrap CI); split stability": format_estimate_ci_stability(row, "Sensitivity"),
            "Specificity, estimate (95% bootstrap CI); split stability": format_estimate_ci_stability(row, "Specificity"),
        })
    return pd.DataFrame(rows)


# Analysis and export

def main():
    print("Reading targeted data...")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Outer CV seeds: {OUTER_SEEDS}")

    X_origin, y_origin, origin_features, origin_bio_ids = prepare_dataset(
        ORIGIN_FILE,
        "group",
    )
    X_cultivar, y_cultivar, cultivar_features, cultivar_bio_ids = prepare_dataset(
        CULTIVAR_FILE,
        "GROUP",
    )

    if origin_features != cultivar_features:
        raise ValueError(
            "The feature columns in the origin and cultivar files do not match."
        )

    print(f"Origin biological samples:   {len(y_origin)}")
    print(f"Cultivar biological samples: {len(y_cultivar)}")
    print(f"Targeted variables:          {len(origin_features)}")

    (
        origin_long,
        origin_class_long,
        origin_probs,
        origin_best,
        origin_folds,
        origin_classes,
    ) = evaluate_task(
        X_origin,
        y_origin,
        origin_bio_ids,
        "Origin",
    )

    (
        cultivar_long,
        cultivar_class_long,
        cultivar_probs,
        cultivar_best,
        cultivar_folds,
        cultivar_classes,
    ) = evaluate_task(
        X_cultivar,
        y_cultivar,
        cultivar_bio_ids,
        "Cultivar",
    )

    all_repeat_metrics = pd.concat(
        [origin_long, cultivar_long],
        ignore_index=True,
    )
    all_class_metrics = pd.concat(
        [origin_class_long, cultivar_class_long],
        ignore_index=True,
    )
    all_probabilities = pd.concat(
        [origin_probs, cultivar_probs],
        ignore_index=True,
        sort=False,
    )
    all_best_params = pd.concat(
        [origin_best, cultivar_best],
        ignore_index=True,
    )
    all_fold_assignments = pd.concat(
        [origin_folds, cultivar_folds],
        ignore_index=True,
    )

    # Summarize bootstrap intervals separately from CV partition variability.
    overall_summary, class_summary, bootstrap_overall_draws, bootstrap_class_draws = (
        build_bootstrap_summaries(
            all_probabilities,
            all_repeat_metrics,
            all_class_metrics,
        )
    )
    sample_prediction_stability = build_sample_prediction_stability(all_probabilities)
    formatted_table = build_formatted_overall_table(overall_summary)

    # Seed manifest
    pd.DataFrame({
        "SplitNo": np.arange(1, len(OUTER_SEEDS) + 1),
        "OuterSeed": OUTER_SEEDS,
        "OuterCV": [f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)" ] * len(OUTER_SEEDS),
        "InnerCV": [f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True)" ] * len(OUTER_SEEDS),
    }).to_csv(
        OUTPUT_DIR / "10_split_seed_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save full outputs
    all_repeat_metrics.to_csv(
        OUTPUT_DIR / "overall_metrics_each_of_10_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_class_metrics.to_csv(
        OUTPUT_DIR / "class_specific_metrics_each_of_10_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "overall_performance_bootstrap95CI_and_split_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_summary.to_csv(
        OUTPUT_DIR / "class_specific_performance_bootstrap95CI_and_split_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    formatted_table.to_csv(
        OUTPUT_DIR / "manuscript_performance_table_bootstrap95CI.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_probabilities.to_csv(
        OUTPUT_DIR / "all_10splits_nestedCV_prediction_probabilities.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_best_params.to_csv(
        OUTPUT_DIR / "nestedCV_best_parameters_all_10splits.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_fold_assignments.to_csv(
        OUTPUT_DIR / "outer_fold_assignments_all_10splits.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_prediction_stability.to_csv(
        OUTPUT_DIR / "sample_level_prediction_split_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if SAVE_BOOTSTRAP_DRAWS:
        bootstrap_overall_draws.to_csv(
            OUTPUT_DIR / "bootstrap_overall_metric_draws.csv",
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_class_draws.to_csv(
            OUTPUT_DIR / "bootstrap_class_specific_metric_draws.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Record the resampling method and seeds with the results.
    pd.DataFrame([{
        "Performance CI": (
            f"{int(CI_LEVEL*100)}% stratified percentile bootstrap at biological-sample level; "
            f"B={BOOTSTRAP_REPS}; based on consensus (mean) of 10 nested-CV outer-OOF probabilities per sample"
        ),
        "Split stability": "Mean, SD, min and max across the 10 outer-CV partition seeds; not treated as a CI",
        "Sample prediction stability": "Per-sample probability mean, SD, min, max and class agreement across 10 outer partitions; not treated as a CI",
        "Bootstrap seed": BOOTSTRAP_SEED,
        "Outer partition seeds": ",".join(map(str, OUTER_SEEDS)),
    }]).to_csv(
        OUTPUT_DIR / "uncertainty_method_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Excel workbook
    with pd.ExcelWriter(
        OUTPUT_DIR / " 秋季-机器学习_机器学习结果.xlsx"
    ) as writer:
        formatted_table.to_excel(writer, sheet_name="Overall formatted", index=False)
        overall_summary.to_excel(writer, sheet_name="Overall bootstrap CI", index=False)
        class_summary.to_excel(writer, sheet_name="Class bootstrap CI", index=False)
        all_repeat_metrics.to_excel(writer, sheet_name="Overall each split", index=False)
        all_class_metrics.to_excel(writer, sheet_name="Class each split", index=False)
        all_best_params.to_excel(writer, sheet_name="Best params", index=False)
        all_fold_assignments.to_excel(writer, sheet_name="Fold assignments", index=False)
        sample_prediction_stability.to_excel(writer, sheet_name="Sample stability", index=False)

    # Export comparison figures.
    for task_name in (["Origin", "Cultivar"] if RUN_FIGURES else []):
        plot_metrics_bar(overall_summary, task_name)
        plot_auc_ci_forest(overall_summary, task_name)
        plot_all_models_consensus_macro_roc(
            all_probabilities,
            overall_summary,
            task_name,
        )

        # Export per-model figures.
        for model_name in MODEL_ORDER:
            plot_consensus_confusion(
                all_probabilities,
                task_name,
                model_name,
            )
            plot_class_auc_ci(
                class_summary,
                task_name,
                model_name,
            )
            plot_model_class_ovr_roc(
                all_probabilities,
                class_summary,
                task_name,
                model_name,
            )

    print("\nAnalysis complete.")
    print(f"Results: {OUTPUT_DIR}")
    print("\nPerformance (bootstrap 95% CI; CV partition stability):")
    print(formatted_table.to_string(index=False))


# Optional descriptive SHAP analysis
def exploratory_fullfit_shap():
    """Describe a full-data SVM fit with SHAP; this is separate from CV stability."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP requested: install shap") from exc

    for task_name, data_path, label_col in [
        ("Origin", ORIGIN_FILE, "group"),
        ("Cultivar", CULTIVAR_FILE, "GROUP"),
    ]:
        X, y, features, _ = prepare_dataset(data_path, label_col)
        encoder = LabelEncoder().fit(y.astype(str))
        y_idx = encoder.transform(y.astype(str))
        pipeline = build_model("SVM", seed=71477, n_classes=len(encoder.classes_))
        pipeline.set_params(clf__C=1, clf__gamma="scale")
        pipeline.fit(X, y_idx)

        standardized = pipeline.named_steps["imputer"].transform(X)
        standardized = pipeline.named_steps["scaler"].transform(standardized)
        rng = np.random.default_rng(71477)
        sample_indices = rng.choice(
            len(standardized),
            size=min(SHAP_MAX_SAMPLES, len(standardized)),
            replace=False,
        )
        background = shap.kmeans(standardized, min(15, len(standardized)))
        explainer = shap.KernelExplainer(
            pipeline.named_steps["clf"].predict_proba,
            background,
        )
        shap_values = explainer.shap_values(standardized[sample_indices], nsamples=150)
        values = np.asarray(shap_values)
        if values.ndim == 3 and values.shape[1] == len(features):
            importance = np.mean(np.abs(values), axis=(0, 2))
        elif values.ndim == 3 and values.shape[2] == len(features):
            importance = np.mean(np.abs(values), axis=(0, 1))
        elif isinstance(shap_values, list):
            importance = np.mean(np.abs(np.asarray(shap_values)), axis=(0, 1))
        else:
            raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")

        pd.DataFrame({
            "Compound": features,
            "MeanAbsoluteSHAP_exploratory_full_fit": importance,
        }).sort_values(
            "MeanAbsoluteSHAP_exploratory_full_fit",
            ascending=False,
        ).to_csv(OUTPUT_DIR / f"{task_name}_exploratory_fullfit_SHAP.csv", index=False)


def cli():
    global ORIGIN_FILE, CULTIVAR_FILE, OUTPUT_DIR, MODEL_ORDER, OUTER_SEEDS
    global BOOTSTRAP_REPS, N_JOBS_INNER, RUN_FIGURES
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify leaf origin and cultivar using targeted LC-MS/MS data."
    )
    parser.add_argument(
        "--origin", type=Path, default=ORIGIN_FILE,
        help="Origin data CSV (label column: group)",
    )
    parser.add_argument(
        "--cultivar", type=Path, default=CULTIVAR_FILE,
        help="Cultivar data CSV (label column: GROUP)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model names, e.g. SVM,RF (default: all)",
    )
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Number of outer CV partitions, from 1 to 10 (default: 10)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=5000,
        help="Number of stratified bootstrap resamples (default: 5000)",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel jobs for the inner grid search (default: 1)",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--exploratory-shap", action="store_true",
        help="Run descriptive full-data SHAP, separately from CV evaluation",
    )
    args = parser.parse_args()

    ORIGIN_FILE = args.origin.resolve()
    CULTIVAR_FILE = args.cultivar.resolve()
    OUTPUT_DIR = args.output.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.models.lower() != "all":
        names = [n.strip() for n in args.models.split(",")]
        invalid = set(names) - set(MODEL_ORDER)
        if invalid:
            parser.error(f"Unknown models: {sorted(invalid)}; valid: {MODEL_ORDER}")
        MODEL_ORDER = names
    if args.repeats < 1 or args.repeats > 10:
        parser.error("--repeats must be from 1 to 10")
    if args.bootstrap < 10:
        parser.error("--bootstrap must be >=10")

    OUTER_SEEDS = OUTER_SEEDS[:args.repeats]
    BOOTSTRAP_REPS = args.bootstrap
    N_JOBS_INNER = args.jobs
    RUN_FIGURES = not args.no_figures
    print(f"Input files: {ORIGIN_FILE.name} and {CULTIVAR_FILE.name}", flush=True)
    print(
        f"Models={MODEL_ORDER}; repeats={args.repeats}; "
        f"bootstrap={args.bootstrap}; output={OUTPUT_DIR}",
        flush=True,
    )
    main()
    if args.exploratory_shap:
        exploratory_fullfit_shap()


if __name__ == '__main__':
    cli()
