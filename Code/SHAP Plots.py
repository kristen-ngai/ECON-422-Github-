
# ============================================================
# 8. SHAP EXPLAINABILITY: SUMMARY, BEESWARM, FORCE,
#    AND DEPENDENCE PLOTS
# ============================================================
# Define feature groups for SHAP analysis
# Configuration
FUEL_COL = "jet_fuel"
FARE_COL = "airline_cpi"
DATE_COL = "date"

fare_features = [
    "fare_yoy_lag1",
    "fare_yoy_lag2",
    "fare_yoy_lag3",
    "fare_yoy_lag6",
    "fare_yoy_lag12",
    "fare_mom_lag1",
    "fare_mom_lag2",
    "fare_mom_lag3",
    "fare_mom_lag6",
    "fare_mom_lag12",
    "month_sin",
    "month_cos",
]

fuel_features = [
    f"{FUEL_COL}_lag1",
    "fuel_change_1m_lag1",
    "fuel_change_3m_lag1",
    "fuel_change_12m_lag1",
]

# Keep only features that actually exist in the dataset
fare_features = [
    c for c in fare_features if c in df.columns
]

fuel_features = [
    c for c in fuel_features if c in df.columns
]
try:
    import shap

    if fuel_features:

        # ----------------------------------------------------
        # 8.1 Prepare training and testing data
        # ----------------------------------------------------

        features = fare_features + fuel_features

        shap_data = df.dropna(
            subset=features + ["fare_yoy"]
        ).copy()

        split = int(len(shap_data) * 0.8)

        train = shap_data.iloc[:split]
        test = shap_data.iloc[split:]

        if len(train) >= 30 and len(test) >= 5:

            # ------------------------------------------------
            # 8.2 Train Random Forest
            # ------------------------------------------------

            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                train[features],
                train["fare_yoy"]
            )

            # ------------------------------------------------
            # 8.3 Calculate SHAP values
            # ------------------------------------------------

            explainer = shap.TreeExplainer(model)

            shap_values = explainer(test[features])

            # ------------------------------------------------
            # FIGURE 05: SHAP SUMMARY PLOT
            # ------------------------------------------------

            plt.figure()

            shap.plots.bar(
                shap_values,
                max_display=15,
                show=False
            )

            plt.title(
                "SHAP Feature Importance",
                fontsize=14
            )

            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "05_shap_summary.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close("all")

            # ------------------------------------------------
            # FIGURE 06: SHAP BEESWARM PLOT
            # ------------------------------------------------

            plt.figure()

            shap.plots.beeswarm(
                shap_values,
                max_display=15,
                show=False
            )

            plt.title(
                "SHAP Beeswarm: Feature Impact on Airline Inflation",
                fontsize=14
            )

            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "06_shap_beeswarm.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close("all")

            # ------------------------------------------------
            # FIGURE 07: SHAP FORCE PLOT
            # ------------------------------------------------

            # Select one observation from the test dataset.
            observation = 0

            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values.values[observation],
                test[features].iloc[observation],
                matplotlib=True,
                show=False
            )

            plt.title(
                "SHAP Force Plot: Individual Airline Inflation Forecast",
                fontsize=13
            )

            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "07_shap_force.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close("all")

            # ------------------------------------------------
            # FIGURE 08: SHAP DEPENDENCE PLOT - FUEL
            # ------------------------------------------------

            fuel_col = f"{FUEL_COL}_lag1"

            if fuel_col in features:

                plt.figure()

                shap.plots.scatter(
                    shap_values[:, fuel_col],
                    color=shap_values,
                    show=False
                )

                plt.title(
                    "SHAP Dependence: Lagged Jet Fuel Prices",
                    fontsize=14
                )

                plt.xlabel("Lagged Jet Fuel Price")
                plt.ylabel("SHAP Value")

                plt.tight_layout()

                plt.savefig(
                    OUTPUT_DIR / "08_shap_fuel_dependence.png",
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close("all")

            # ------------------------------------------------
            # FIGURE 09: SHAP DEPENDENCE PLOT - FARE INFLATION
            # ------------------------------------------------

            fare_col = "fare_yoy_lag1"

            if fare_col in features:

                plt.figure()

                shap.plots.scatter(
                    shap_values[:, fare_col],
                    color=shap_values,
                    show=False
                )

                plt.title(
                    "SHAP Dependence: Lagged Airline Inflation",
                    fontsize=14
                )

                plt.xlabel("Lagged Airline Inflation")
                plt.ylabel("SHAP Value")

                plt.tight_layout()

                plt.savefig(
                    OUTPUT_DIR / "09_shap_fare_dependence.png",
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close("all")

            # ------------------------------------------------
            # 8.4 Export SHAP values
            # ------------------------------------------------

            shap_importance = pd.DataFrame({
                "feature": features,
                "mean_absolute_shap": np.abs(
                    shap_values.values
                ).mean(axis=0)
            })

            shap_importance = shap_importance.sort_values(
                "mean_absolute_shap",
                ascending=False
            )

            shap_importance.to_csv(
                OUTPUT_DIR.parent / "shap_feature_importance.csv",
                index=False
            )

            print("\nSHAP FEATURE IMPORTANCE")
            print(shap_importance.to_string(index=False))

            print("\nSHAP plots successfully generated.")

        else:
            print(
                "Not enough observations for SHAP holdout plots."
            )

    else:
        print(
            "SHAP plots skipped: fuel features unavailable."
        )

except ImportError:
    print(
        "SHAP not installed. Run: pip install shap"
    )

except Exception as e:
    print(f"SHAP visualization error: {e}")
