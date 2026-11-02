#Data Examination

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load your engineered dataset
df = pd.read_csv("data/processed/forecast_features.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").copy()

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# GRAPH 1: Airline CPI vs. Jet Fuel
# ---------------------------------------
fig, ax1 = plt.subplots(figsize=(13, 6))

ax1.plot(
    df["date"], df["airline_cpi"],
    label="Airline CPI", linewidth=2
)
ax1.set_xlabel("Date")
ax1.set_ylabel("Airline CPI")

ax2 = ax1.twinx()
ax2.plot(
    df["date"], df["jet_fuel"],
    label="Jet Fuel Price", linestyle="--", linewidth=2
)
ax2.set_ylabel("Jet Fuel Price")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title("Airline CPI vs. Jet Fuel Prices")
plt.tight_layout()
plt.savefig(output_dir / "fare_fuel_dual_axis.png", dpi=300)
plt.show()


# ---------------------------------------
# GRAPH 2: Indexed comparison
# Both series start at 100
# ---------------------------------------
indexed = df[["date", "airline_cpi", "jet_fuel"]].dropna().copy()

indexed["fare_index"] = (
    indexed["airline_cpi"] / indexed["airline_cpi"].iloc[0]
) * 100

indexed["fuel_index"] = (
    indexed["jet_fuel"] / indexed["jet_fuel"].iloc[0]
) * 100

plt.figure(figsize=(13, 6))

plt.plot(
    indexed["date"], indexed["fare_index"],
    label="Airline CPI", linewidth=2
)

plt.plot(
    indexed["date"], indexed["fuel_index"],
    label="Jet Fuel", linewidth=2, linestyle="--"
)

plt.axhline(100, linestyle=":", linewidth=1)

plt.title("Indexed Airline CPI vs. Jet Fuel Prices")
plt.xlabel("Date")
plt.ylabel("Index (Starting Value = 100)")
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "fare_fuel_indexed.png", dpi=300)
plt.show()


# ---------------------------------------
# GRAPH 3: Monthly changes
# ---------------------------------------
monthly = df.dropna(
    subset=["fare_inflation_mom", "fuel_change_mom"]
)

plt.figure(figsize=(8, 6))

plt.scatter(
    monthly["fuel_change_mom"],
    monthly["fare_inflation_mom"],
    alpha=0.6
)

plt.axhline(0, linestyle=":", linewidth=1)
plt.axvline(0, linestyle=":", linewidth=1)

corr = monthly[
    ["fare_inflation_mom", "fuel_change_mom"]
].corr().iloc[0, 1]

plt.title(f"Monthly Fare Inflation vs. Fuel Changes (r={corr:.3f})")
plt.xlabel("Monthly Fuel Price Change")
plt.ylabel("Monthly Airline Fare Inflation")
plt.tight_layout()

plt.savefig(output_dir / "fare_fuel_monthly_changes.png", dpi=300)
plt.show()


# ---------------------------------------
# GRAPH 4: Fuel lead-lag correlation
# ---------------------------------------
# Compare fuel changes from earlier months
# against current monthly fare inflation.

lag_results = []

for lag in range(0, 13):

    temp = df[["fare_inflation_mom", "fuel_change_mom"]].copy()

    temp["fuel_prior"] = temp["fuel_change_mom"].shift(lag)

    corr = temp[
        ["fare_inflation_mom", "fuel_prior"]
    ].corr().iloc[0, 1]

    lag_results.append({
        "Fuel lag (months)": lag,
        "Correlation": corr
    })

lag_df = pd.DataFrame(lag_results)

plt.figure(figsize=(10, 5))

plt.bar(
    lag_df["Fuel lag (months)"],
    lag_df["Correlation"]
)

plt.axhline(0, linestyle=":", linewidth=1)

plt.title("Fuel Price Changes vs. Current Fare Inflation")
plt.xlabel("Months Earlier Fuel Prices Changed")
plt.ylabel("Correlation")

plt.xticks(range(13))
plt.tight_layout()

plt.savefig(output_dir / "fuel_lead_lag_correlation.png", dpi=300)
plt.show()

print(lag_df.to_string(index=False))
print(f"\nGraphs saved to: {output_dir}")
