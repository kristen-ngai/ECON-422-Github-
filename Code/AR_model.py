
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. PREPARE DATA
# --------------------------------------------------

forecast_features=pd.read_csv("data/processed/forecast_features.csv")
df=forecast_features.copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

# Predict next month's year-over-year airline CPI inflation
target = "fare_inflation_yoy"
df["target_next_month"] = df[target].shift(-1)

# ARX(1): one-month lag of fare inflation and fuel changes
arx_features = [
    "fare_lag_1",
    "fuel_lag_1"
]

# Machine learning feature groups
fare_features = [
    c for c in df.columns
    if c.startswith("fare_lag_")
]

fuel_features = [
    c for c in df.columns
    if c.startswith(("fuel_lag_", "fuel_level_lag_",
                     "fuel_change_", "fuel_rolling_"))
]

macro_features = [
    c for c in ["unemployment", "core_cpi"]
    if c in df.columns
]

# Combined fare + fuel features
joint_features = list(dict.fromkeys(
    fare_features + fuel_features
))

# Remove missing values
required = list(dict.fromkeys(
    ["target_next_month"] + arx_features + joint_features
))

data = df[required].dropna()

X_arx = data[arx_features]
X_joint = data[joint_features]
y = data["target_next_month"]

# --------------------------------------------------
# 2. DEFINE BASE MODELS
# --------------------------------------------------

def get_models():
    return {
        "ARX(1)": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    }


# --------------------------------------------------
# 3. EXPANDING-WINDOW EVALUATION
# --------------------------------------------------

outer_cv = TimeSeriesSplit(n_splits=5)

results = []
prediction_records = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(data), start=1
):

    X_train_arx = X_arx.iloc[train_idx]
    X_test_arx = X_arx.iloc[test_idx]

    X_train_joint = X_joint.iloc[train_idx]
    X_test_joint = X_joint.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --------------------------------------------------
    # A. ARX(1) MODEL
    # --------------------------------------------------

    arx_model = LinearRegression()

    arx_model.fit(X_train_arx, y_train)

    arx_pred = arx_model.predict(X_test_arx)

    # --------------------------------------------------
    # B. INDIVIDUAL ML MODELS
    # --------------------------------------------------

    ml_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    }

    ml_predictions = {}

    for name, model in ml_models.items():

        model.fit(X_train_joint, y_train)

        ml_predictions[name] = model.predict(X_test_joint)

    # --------------------------------------------------
    # C. JOINT MODELING
    # Combine fare + fuel features within one model
    # --------------------------------------------------

    joint_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42
    )

    joint_model.fit(X_train_joint, y_train)

    joint_pred = joint_model.predict(X_test_joint)

    # --------------------------------------------------
    # D. SUPER LEARNER
    # Combine predictions at the prediction level
    # --------------------------------------------------

    # Generate out-of-fold predictions within outer training data.
    # These predictions train the meta-model without using
    # in-sample predictions.

    inner_cv = TimeSeriesSplit(n_splits=4)

    base_names = [
        "ARX(1)",
        "Random Forest",
        "Gradient Boosting"
    ]

    meta_train = pd.DataFrame(
        index=X_train_arx.index,
        columns=base_names,
        dtype=float
    )

    for inner_train_idx, inner_val_idx in inner_cv.split(
        X_train_arx
    ):

        inner_y_train = y_train.iloc[inner_train_idx]

        # ARX(1)
        inner_arx = LinearRegression()
        inner_arx.fit(
            X_train_arx.iloc[inner_train_idx],
            inner_y_train
        )

        meta_train.iloc[inner_val_idx, 0] = (
            inner_arx.predict(
                X_train_arx.iloc[inner_val_idx]
            )
        )

        # Random Forest
        inner_rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42
        )

        inner_rf.fit(
            X_train_joint.iloc[inner_train_idx],
            inner_y_train
        )

        meta_train.iloc[inner_val_idx, 1] = (
            inner_rf.predict(
                X_train_joint.iloc[inner_val_idx]
            )
        )

        # Gradient Boosting
        inner_gb = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )

        inner_gb.fit(
            X_train_joint.iloc[inner_train_idx],
            inner_y_train
        )

        meta_train.iloc[inner_val_idx, 2] = (
            inner_gb.predict(
                X_train_joint.iloc[inner_val_idx]
            )
        )

    # Only use observations with valid out-of-fold predictions
    valid_meta = meta_train.dropna()

    # Fit the meta-model using historical predictions
    meta_model = Ridge(alpha=1.0)

    meta_model.fit(
        valid_meta,
        y_train.loc[valid_meta.index]
    )

    # Create test predictions from the refitted base models
    test_meta = pd.DataFrame({
        "ARX(1)": arx_pred,
        "Random Forest": ml_predictions["Random Forest"],
        "Gradient Boosting": ml_predictions["Gradient Boosting"]
    }, index=X_test_arx.index)

    # Final Super Learner prediction
    super_learner_pred = meta_model.predict(test_meta)

    # --------------------------------------------------
    # E. COLLECT ALL PREDICTIONS
    # --------------------------------------------------

    predictions = {
        "ARX(1)": arx_pred,
        "Random Forest": ml_predictions["Random Forest"],
        "Gradient Boosting": ml_predictions["Gradient Boosting"],
        "Joint Model (Fare + Fuel)": joint_pred,
        "Super Learner": super_learner_pred
    }

    for name, pred in predictions.items():

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "Fold": fold,
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

        for date, actual, forecast in zip(
            y_test.index, y_test, pred
        ):
            prediction_records.append({
                "Date": date,
                "Fold": fold,
                "Model": name,
                "Actual": actual,
                "Predicted": forecast
            })


# --------------------------------------------------
# 4. SUMMARIZE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("Model")[["MAE", "RMSE"]]
    .mean()
    .sort_values("MAE")
)

print("\nONE-MONTH-AHEAD MODEL COMPARISON")
print(summary.round(4))

# Save results
summary.to_csv("one_month_AR_superlearner_summary.csv")

pd.DataFrame(prediction_records).to_csv(
    "one_month_AR_superlearner_predictions.csv",
    index=False
)


import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 5. CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("results/AR_superlearner", exist_ok=True)

# Consistent model ordering
model_order = [
    "ARX(1)",
    "Random Forest",
    "Gradient Boosting",
    "Joint Model (Fare + Fuel)",
    "Super Learner"
]

# --------------------------------------------------
# GRAPH 1: MAE COMPARISON
# --------------------------------------------------

plot_mae = summary.reindex(model_order)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    plot_mae.index,
    plot_mae["MAE"]
)

plt.title("One-Month-Ahead Forecasting: MAE Comparison")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Model")
plt.xticks(rotation=20, ha="right")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    "results/AR_superlearner/MAE_comparison.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# GRAPH 2: RMSE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plot_rmse = summary.reindex(model_order)

bars = plt.bar(
    plot_rmse.index,
    plot_rmse["RMSE"]
)

plt.title("One-Month-Ahead Forecasting: RMSE Comparison")
plt.ylabel("Root Mean Squared Error")
plt.xlabel("Model")
plt.xticks(rotation=20, ha="right")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    "results/AR_superlearner/RMSE_comparison.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# GRAPH 3: ACTUAL VS. PREDICTED VALUES
# --------------------------------------------------

pred_df = pd.DataFrame(prediction_records)
pred_df["Date"] = pd.to_datetime(pred_df["Date"])

# Average predictions across the five test folds
# Each date should have one out-of-sample prediction.
actual_values = (
    pred_df.groupby("Date")["Actual"].first()
)

plt.figure(figsize=(13, 7))

plt.plot(
    actual_values.index,
    actual_values.values,
    label="Actual Airline CPI Inflation",
    linewidth=2.5
)

for model in [
    "ARX(1)",
    "Joint Model (Fare + Fuel)",
    "Super Learner"
]:

    model_data = pred_df[
        pred_df["Model"] == model
    ].sort_values("Date")

    plt.plot(
        model_data["Date"],
        model_data["Predicted"],
        label=model,
        linewidth=1.5
    )

plt.title("Actual vs. Predicted Airline CPI Inflation")
plt.xlabel("Date")
plt.ylabel("Year-over-Year Airline CPI Inflation (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/AR_superlearner/actual_vs_predicted.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# GRAPH 4: FORECAST ERRORS OVER TIME
# --------------------------------------------------

plt.figure(figsize=(13, 6))

for model in [
    "ARX(1)",
    "Joint Model (Fare + Fuel)",
    "Super Learner"
]:

    model_data = pred_df[
        pred_df["Model"] == model
    ].sort_values("Date").copy()

    model_data["Error"] = (
        model_data["Actual"] - model_data["Predicted"]
    )

    plt.plot(
        model_data["Date"],
        model_data["Error"],
        label=model,
        linewidth=1.5
    )

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Forecast Errors Over Time")
plt.xlabel("Date")
plt.ylabel("Actual - Predicted (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/AR_superlearner/forecast_errors.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# GRAPH 5: MAE BY CROSS-VALIDATION FOLD
# --------------------------------------------------

fold_mae = results_df.pivot(
    index="Fold",
    columns="Model",
    values="MAE"
)

fold_mae = fold_mae.reindex(columns=model_order)

plt.figure(figsize=(11, 6))

fold_mae.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Forecasting MAE Across Time-Series Validation Folds")
plt.xlabel("Validation Fold")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=0)
plt.legend(title="Model", bbox_to_anchor=(1.02, 1))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/AR_superlearner/MAE_by_fold.png",
    dpi=300
)
plt.show()


print("\nAll graphs saved in results/AR_superlearner/")

#Shap analysis for AR model

import shap
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# SHAP STORAGE
# --------------------------------------------------

shap_results = {
    "ARX(1)": [],
    "Joint Model": [],
    "Super Learner": []
}

os.makedirs("results/AR_superlearner/SHAP", exist_ok=True)

super_learner_pred = meta_model.predict(test_meta)


# --------------------------------------------------
# SHAP ANALYSIS: ARX(1), JOINT MODEL, SUPER LEARNER
# --------------------------------------------------

# A. ARX(1): Linear SHAP
arx_explainer = shap.LinearExplainer(
    arx_model,
    X_train_arx
)

arx_shap = arx_explainer(X_test_arx)

shap_results["ARX(1)"].append(
    pd.DataFrame(
        np.abs(arx_shap.values),
        columns=arx_features,
        index=X_test_arx.index
    )
)


# B. JOINT MODEL: Random Forest Tree SHAP
joint_explainer = shap.TreeExplainer(joint_model)

joint_shap = joint_explainer(X_test_joint)

shap_results["Joint Model"].append(
    pd.DataFrame(
        np.abs(joint_shap.values),
        columns=joint_features,
        index=X_test_joint.index
    )
)


# C. SUPER LEARNER: Explain the complete ensemble
# The input is the original fare + fuel feature set.
# The function runs all three base models and the Ridge
# meta-model, so SHAP explains the final ensemble forecast.

def super_learner_predict(X):
    X_frame = pd.DataFrame(
        np.asarray(X),
        columns=joint_features
    )

    # Predictions from the three fitted base models
    arx_prediction = arx_model.predict(
        X_frame[arx_features]
    )

    rf_prediction = ml_models["Random Forest"].predict(
        X_frame[joint_features]
    )

    gb_prediction = ml_models["Gradient Boosting"].predict(
        X_frame[joint_features]
    )

    # Combine predictions using the fitted Ridge meta-model
    meta_input = pd.DataFrame({
        "ARX(1)": arx_prediction,
        "Random Forest": rf_prediction,
        "Gradient Boosting": gb_prediction
    })

    return meta_model.predict(meta_input)


# Use historical training observations as the background
background = shap.sample(
    X_train_joint,
    min(50, len(X_train_joint)),
    random_state=42
)

super_explainer = shap.PermutationExplainer(
    super_learner_predict,
    background
)

super_shap = super_explainer(
    X_test_joint,
    max_evals=2 * len(joint_features) + 1
)

shap_results["Super Learner"].append(
    pd.DataFrame(
        np.abs(super_shap.values),
        columns=joint_features,
        index=X_test_joint.index
    )
)


# --------------------------------------------------
# SHAP RESULTS: TOP 10 FEATURES
# --------------------------------------------------

shap_importance = {}

for model_name, fold_results in shap_results.items():

    # Combine out-of-sample SHAP values from all folds
    combined_shap = pd.concat(fold_results)

    # Average absolute SHAP value across all test observations
    importance = (
        combined_shap.mean()
        .sort_values(ascending=False)
    )

    shap_importance[model_name] = importance

    # Keep only the top 10 features
    top_10 = importance.head(10)

    print(f"\nTOP 10 SHAP FEATURES: {model_name}")
    print(top_10.round(4))

    # Plot
    plt.figure(figsize=(9, 5))

    top_10.sort_values().plot(
        kind="barh"
    )

    plt.title(f"Top 10 SHAP Features: {model_name}")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.savefig(
        f"results/AR_superlearner/SHAP/{filename}_top10.png",
        dpi=300
    )

    plt.show()


# --------------------------------------------------
# COMBINED SHAP IMPORTANCE TABLE
# --------------------------------------------------

shap_comparison = pd.DataFrame(shap_importance).fillna(0)

shap_comparison.to_csv(
    "results/AR_superlearner/SHAP/shap_comparison.csv"
)

print("\nSHAP IMPORTANCE COMPARISON")
print(shap_comparison.round(4).head(15))

