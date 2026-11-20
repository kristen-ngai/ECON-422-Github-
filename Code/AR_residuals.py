
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f as f_distribution
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 1. FIND PROJECT FILES AND SET OUTPUT DIRECTORIES
# ---------------------------------------------------------------------

def find_project_root():
    candidates = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    candidates += [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "data" / "processed" / "forecast_features.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find data/processed/forecast_features.csv. "
        "Place this script inside your project or run it from the project root."
    )

BASE_DIR = find_project_root()
DATA_FILE = BASE_DIR / "data" / "processed" / "forecast_features.csv"
OUTPUT_DIR = BASE_DIR / "results" / "residual_model_1_ARX1"
GRAPH_DIR = OUTPUT_DIR / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Break dates are selected from external events, not from the plotted data.
BREAK_DATES = {
    "Financial_crisis_Sep_2008": pd.Timestamp("2008-09-01"),
    "COVID_Mar_2020": pd.Timestamp("2020-03-01"),
}

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
MIN_REGIME_TRAIN = 24  # keep post-March-2020 models simple

# ---------------------------------------------------------------------
# 2. LOAD DATA AND PRESERVE THE ORIGINAL FORECAST TARGET TIMING
# ---------------------------------------------------------------------

df = pd.read_csv(DATA_FILE)
if "date" not in df.columns:
    raise KeyError("The input CSV must contain a 'date' column.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

TARGET = "fare_inflation_yoy"
ARX_FEATURES = ["fare_lag_1", "fuel_lag_1"]

missing = [c for c in [TARGET] + ARX_FEATURES if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in forecast_features.csv: {missing}")

# The original project predicts next month's year-over-year fare inflation.
df["target_next_month"] = df[TARGET].shift(-1)

# Use a target-month index for break-period reporting. Predictor rows remain
# ---------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------

def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def score_row(model_name, actual, predicted, period="All"):
    return {
        "Model": model_name,
        "Period": period,
        "N": int(len(actual)),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": rmse(actual, predicted),
    }

def chow_test(y_values, X_values, break_mask):
    """
    Classical Chow test for a known break date.
    Fits pooled OLS and separate OLS on the two subsamples.
    """
    yv = np.asarray(y_values, dtype=float)
    Xv = np.asarray(X_values, dtype=float)
    break_mask = np.asarray(break_mask, dtype=bool)

    X1, y1 = Xv[~break_mask], yv[~break_mask]
    X2, y2 = Xv[break_mask], yv[break_mask]

    k = Xv.shape[1] + 1  # regressors plus intercept
    if min(len(y1), len(y2)) <= k or len(yv) <= 2 * k:
        return {"F_statistic": np.nan, "p_value": np.nan,
                "N_pre": len(y1), "N_post": len(y2),
                "Note": "Insufficient observations for Chow test"}

    pooled = LinearRegression().fit(Xv, yv)
    pre = LinearRegression().fit(X1, y1)
    post = LinearRegression().fit(X2, y2)

    rss_pooled = np.sum((yv - pooled.predict(Xv)) ** 2)
    rss_separate = (
        np.sum((y1 - pre.predict(X1)) ** 2)
        + np.sum((y2 - post.predict(X2)) ** 2)
    )
    numerator_df = k
    denominator_df = len(yv) - 2 * k
    if denominator_df <= 0 or rss_separate <= 0:
        return {"F_statistic": np.nan, "p_value": np.nan,
                "N_pre": len(y1), "N_post": len(y2),
                "Note": "Invalid residual degrees of freedom"}

    F_stat = ((rss_pooled - rss_separate) / numerator_df) / (
        rss_separate / denominator_df
    )
    p_value = f_distribution.sf(max(F_stat, 0), numerator_df, denominator_df)
    return {
        "F_statistic": float(F_stat),
        "p_value": float(p_value),
        "N_pre": len(y1),
        "N_post": len(y2),
        "Note": "",
    }

def expanding_predictions(X, target, model_factory, min_train=None):
    """Generate strictly forward-looking expanding-window predictions."""
    n_splits = OUTER_SPLITS
    cv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        if min_train is not None and len(train_idx) < min_train:
            continue
        model = model_factory()
        model.fit(X.iloc[train_idx], target.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        for idx, actual, forecast in zip(test_idx, target.iloc[test_idx], pred):
            records.append({
                "row_index": X.index[idx],
                "target_date": data.loc[X.index[idx], "target_date"],
                "Fold": fold,
                "Actual": float(actual),
                "Predicted": float(forecast),
                "Residual": float(actual - forecast),
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------
# 4. STRUCTURAL BREAK CHECKS: COEFFICIENTS AND CHOW TESTS
# ---------------------------------------------------------------------
from pathlib import Path

# Locate project root (script is inside Code/)
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "forecast_features.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

# Original forecasting target: next month's YoY fare inflation
TARGET = "fare_inflation_yoy"
ARX_FEATURES = ["fare_lag_1", "fuel_lag_1"]

df["target_next_month"] = df[TARGET].shift(-1)

# Identify the month being forecast
target_dates = df.index.to_series().shift(-1)

# Add time trend
df["time_trend"] = np.arange(len(df), dtype=float)

# Add target month
df["target_month"] = target_dates.dt.month

# Create the analysis dataset
data = df[
    ["target_next_month"] + ARX_FEATURES +
    ["time_trend", "target_month"]
].copy()

data["target_date"] = target_dates

data = data.dropna().copy()
data["target_date"] = pd.to_datetime(data["target_date"])

# Define dependent and independent variables
y = data["target_next_month"].astype(float)

X_arx = data[ARX_FEATURES].astype(float)


break_rows = []
coefficient_rows = []

for break_name, break_date in BREAK_DATES.items():
    mask = data["target_date"] >= break_date
    test = chow_test(y, X_arx, mask)
    break_rows.append({
        "Break": break_name,
        "Break_date": break_date.date().isoformat(),
        **test,
    })

    for period_name, period_mask in [
        ("Before", ~mask),
        ("After", mask),
    ]:
        if int(period_mask.sum()) <= len(ARX_FEATURES) + 2:
            continue
        model = LinearRegression().fit(X_arx.loc[period_mask], y.loc[period_mask])
        row = {
            "Break": break_name,
            "Period": period_name,
            "N": int(period_mask.sum()),
            "Intercept": float(model.intercept_),
        }
        row.update({
            f"Coefficient_{feature}": float(coef)
            for feature, coef in zip(ARX_FEATURES, model.coef_)
        })
        coefficient_rows.append(row)

break_results = pd.DataFrame(break_rows)
break_coefficients = pd.DataFrame(coefficient_rows)
break_results.to_csv(OUTPUT_DIR / "structural_break_chow_tests.csv", index=False)
break_coefficients.to_csv(OUTPUT_DIR / "arx_coefficients_before_after_breaks.csv", index=False)

# ---------------------------------------------------------------------
# 5. COMPARE ARX(1) WITH TREND AND MONTH-OF-YEAR CONTROLS
# ---------------------------------------------------------------------

classical_specs = {
    "ARX(1)": lambda: LinearRegression(),
    "ARX(1) + trend": lambda: LinearRegression(),
    "ARX(1) + month indicators": lambda: LinearRegression(),
    "ARX(1) + trend + month indicators": lambda: LinearRegression(),
}

# ---------------------------------------------------------------------
# 6. EXPANDING-WINDOW FORECAST COMPARISON
# ---------------------------------------------------------------------

# Define the predictors for each specification.
# Month indicators are represented using one-hot encoding.
month_dummies = pd.get_dummies(
    data["target_month"].astype(int),
    prefix="month",
    drop_first=True,
    dtype=float
)

X_trend = data[ARX_FEATURES + ["time_trend"]].astype(float)

X_seasonal = pd.concat(
    [data[ARX_FEATURES], month_dummies],
    axis=1
).astype(float)

X_full = pd.concat(
    [data[ARX_FEATURES + ["time_trend"]], month_dummies],
    axis=1
).astype(float)

model_inputs = {
    "ARX(1)": X_arx,
    "ARX(1) + trend": X_trend,
    "ARX(1) + month indicators": X_seasonal,
    "ARX(1) + trend + month indicators": X_full,
}

# Store predictions from each model.
prediction_results = []

for model_name, X_model in model_inputs.items():

    predictions = expanding_predictions(
        X_model,
        y,
        model_factory=lambda: LinearRegression()
    )

    predictions["Model"] = model_name
    prediction_results.append(predictions)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)

predictions_df["target_date"] = pd.to_datetime(
    predictions_df["target_date"]
)

predictions_df.to_csv(
    OUTPUT_DIR / "expanding_window_predictions.csv",
    index=False
)

# ---------------------------------------------------------------------
# 7. FORECAST PERFORMANCE SUMMARY
# ---------------------------------------------------------------------

performance_rows = []

for model_name, group in predictions_df.groupby("Model"):

    performance_rows.append(
        score_row(
            model_name,
            group["Actual"],
            group["Predicted"]
        )
    )

performance = pd.DataFrame(performance_rows)
performance = performance.sort_values("RMSE")

performance.to_csv(
    OUTPUT_DIR / "forecast_performance_summary.csv",
    index=False
)

print("\nFORECAST PERFORMANCE")
print(performance.to_string(index=False))


# ---------------------------------------------------------------------
# 8. FIGURE 1: ACTUAL VS PREDICTED FARE INFLATION
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 11),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, (model_name, group) in zip(
    axes,
    predictions_df.groupby("Model")
):

    group = group.sort_values("target_date")

    ax.plot(
        group["target_date"],
        group["Actual"],
        color="black",
        linewidth=1.8,
        label="Actual"
    )

    ax.plot(
        group["target_date"],
        group["Predicted"],
        color="royalblue",
        linewidth=1.5,
        linestyle="--",
        label="Predicted"
    )

    ax.set_title(model_name, fontsize=11, fontweight="bold")
    ax.set_ylabel("Fare inflation (YoY)")
    ax.grid(alpha=0.25)
    ax.legend()

    for break_date in BREAK_DATES.values():
        ax.axvline(
            break_date,
            color="red",
            linestyle=":",
            alpha=0.65
        )

fig.suptitle(
    "Actual vs Predicted Fare Inflation: Expanding-Window Forecasts",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    GRAPH_DIR / "01_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# 9. FIGURE 2: FORECAST ERRORS OVER TIME
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 10),
    sharex=True
)

axes = axes.flatten()

for ax, (model_name, group) in zip(
    axes,
    predictions_df.groupby("Model")
):

    group = group.sort_values("target_date")

    ax.plot(
        group["target_date"],
        group["Residual"],
        color="darkorange",
        linewidth=1.2
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(model_name, fontweight="bold")
    ax.set_ylabel("Forecast error")
    ax.grid(alpha=0.25)

    for break_date in BREAK_DATES.values():
        ax.axvline(
            break_date,
            color="red",
            linestyle=":",
            alpha=0.65
        )

fig.suptitle(
    "Forecast Residuals Over Time",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    GRAPH_DIR / "02_forecast_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# 10. FIGURE 3: MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

performance_plot = performance.sort_values("RMSE", ascending=True)

axes[0].barh(
    performance_plot["Model"],
    performance_plot["MAE"],
    color="steelblue"
)

axes[0].set_title("Mean Absolute Error (MAE)")
axes[0].set_xlabel("MAE")
axes[0].grid(axis="x", alpha=0.25)

axes[1].barh(
    performance_plot["Model"],
    performance_plot["RMSE"],
    color="darkorange"
)

axes[1].set_title("Root Mean Squared Error (RMSE)")
axes[1].set_xlabel("RMSE")
axes[1].grid(axis="x", alpha=0.25)

fig.suptitle(
    "Forecast Performance Across Model Specifications",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    GRAPH_DIR / "03_model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# 11. FIGURE 4: STRUCTURAL BREAK COEFFICIENT COMPARISON
# ---------------------------------------------------------------------

if not break_coefficients.empty:

    coefficient_columns = [
        "Coefficient_fare_lag_1",
        "Coefficient_fuel_lag_1"
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6)
    )

    for ax, coefficient in zip(
        axes,
        coefficient_columns
    ):

        plot_data = break_coefficients.melt(
            id_vars=["Break", "Period"],
            value_vars=[coefficient],
            var_name="Variable",
            value_name="Coefficient"
        )

        plot_data["Group"] = (
            plot_data["Break"] + " - " + plot_data["Period"]
        )

        ax.bar(
            plot_data["Group"],
            plot_data["Coefficient"],
            color=[
                "steelblue" if p == "Before" else "darkorange"
                for p in plot_data["Period"]
            ]
        )

        ax.set_title(
            coefficient.replace("Coefficient_", "")
        )

        ax.set_ylabel("Estimated coefficient")
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "ARX(1) Coefficients Before and After Structural Breaks",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "04_structural_break_coefficients.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# 12. FIGURE 5: CHOW TEST RESULTS
# ---------------------------------------------------------------------

if not break_results.empty:

    fig, ax = plt.subplots(figsize=(10, 6))

    chow_plot = break_results.copy()

    chow_plot["Label"] = chow_plot["Break"].str.replace(
        "_", " ", regex=False
    )

    colors = [
        "firebrick" if pd.notna(p) and p < 0.05 else "steelblue"
        for p in chow_plot["p_value"]
    ]

    bars = ax.bar(
        chow_plot["Label"],
        chow_plot["p_value"],
        color=colors
    )

    ax.axhline(
        0.05,
        color="red",
        linestyle="--",
        label="5% significance level"
    )

    ax.set_ylabel("Chow test p-value")
    ax.set_title(
        "Structural Break Tests: Chow Test p-values",
        fontweight="bold"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, chow_plot["p_value"]):

        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom"
            )

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "05_chow_test_results.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# 13. FIGURE 6: ACTUAL VS PREDICTED SCATTER PLOT
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 11)
)

axes = axes.flatten()

for ax, (model_name, group) in zip(
    axes,
    predictions_df.groupby("Model")
):

    ax.scatter(
        group["Actual"],
        group["Predicted"],
        alpha=0.65,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3
    )

    lower = min(
        group["Actual"].min(),
        group["Predicted"].min()
    )

    upper = max(
        group["Actual"].max(),
        group["Predicted"].max()
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        color="red",
        linestyle="--",
        label="Perfect forecast"
    )

    ax.set_title(model_name, fontweight="bold")
    ax.set_xlabel("Actual fare inflation")
    ax.set_ylabel("Predicted fare inflation")
    ax.legend()
    ax.grid(alpha=0.25)

fig.suptitle(
    "Actual vs Predicted Values",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    GRAPH_DIR / "06_actual_predicted_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# 14. FIGURE 7: FORECAST ERROR DISTRIBUTIONS
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

for model_name, group in predictions_df.groupby("Model"):

    ax.hist(
        group["Residual"],
        bins=25,
        alpha=0.4,
        density=True,
        label=model_name
    )

ax.axvline(
    0,
    color="black",
    linestyle="--"
)

ax.set_title(
    "Distribution of Forecast Errors",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Forecast error")
ax.set_ylabel("Density")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    GRAPH_DIR / "07_residual_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# 15. FIGURE 8: FORECAST PERFORMANCE BY HISTORICAL PERIOD
# ---------------------------------------------------------------------

period_definitions = {
    "Pre-financial crisis": (
        pd.Timestamp("1900-01-01"),
        pd.Timestamp("2008-09-01")
    ),
    "Post-financial crisis / Pre-COVID": (
        pd.Timestamp("2008-09-01"),
        pd.Timestamp("2020-03-01")
    ),
    "Post-COVID": (
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2100-01-01")
    )
}

period_performance = []

for model_name, group in predictions_df.groupby("Model"):

    for period_name, (start, end) in period_definitions.items():

        subset = group[
            (group["target_date"] >= start) &
            (group["target_date"] < end)
        ]

        if len(subset) == 0:
            continue

        period_performance.append(
            score_row(
                model_name,
                subset["Actual"],
                subset["Predicted"],
                period=period_name
            )
        )

period_performance = pd.DataFrame(period_performance)

period_performance.to_csv(
    OUTPUT_DIR / "forecast_performance_by_period.csv",
    index=False
)

if not period_performance.empty:

    pivot_rmse = period_performance.pivot(
        index="Model",
        columns="Period",
        values="RMSE"
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    pivot_rmse.plot(
        kind="bar",
        ax=ax,
        width=0.8
    )

    ax.set_title(
        "Forecast RMSE Across Structural Regimes",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("RMSE")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Historical period")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "08_performance_by_period.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# 16. SAVE FINAL RESULTS AND PRINT SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 65)
print("RESIDUAL MODEL 1: ARX(1) ANALYSIS COMPLETED")
print("=" * 65)

print("\nOverall forecast performance:")
print(performance.to_string(index=False))

print("\nChow structural break tests:")
print(break_results.to_string(index=False))

print("\nARX coefficients before and after breaks:")
print(break_coefficients.to_string(index=False))

print("\nForecast performance by historical period:")
print(period_performance.to_string(index=False))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGraph directory:")
print(GRAPH_DIR)

print("\nGenerated figures:")
for graph_file in sorted(GRAPH_DIR.glob("*.png")):
    print(" -", graph_file.name)

print("\nAnalysis successfully completed.")