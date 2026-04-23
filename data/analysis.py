import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Churn.csv")

# Clean data
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Calculate churn rate
churn_rate = df["Churn"].value_counts(normalize=True)

print("Churn Rate:")
print(churn_rate)

# Plot
churn_rate.plot(kind="bar")

plt.title("Churn Rate")
plt.xlabel("Churn")
plt.ylabel("Percentage")

plt.savefig("churn_chart.png")

print("Chart saved")