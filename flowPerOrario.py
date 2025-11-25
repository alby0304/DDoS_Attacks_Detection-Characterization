import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ddos_dataset.csv")

# if you removed spaces from the CSV header, skip this line
# otherwise keep it, it doesn't hurt:
df.columns = df.columns.str.strip()

# 1) Timestamp must be datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# 2) Make sure the label column exists and create is_attack
print(df.columns)          # just to see names in the terminal
print(df["label"].head())  # should print labels

df["is_attack"] = np.where(df["label"] == "benign", "benign", "attack")

# 3) Make sure FlowID (or Flow ID) exists
print(df[["FlowID"]].head())

flows_time_15m = (
    df
    .set_index("Timestamp")
    .groupby("is_attack")
    .resample("15T")["FlowID"]      # or "Flow ID" depending on your column name
    .nunique()
    .unstack(0)
    .fillna(0)
)

ft_long = (flows_time_15m
           .reset_index()
           .melt(id_vars="Timestamp",
                 value_vars=["benign", "attack"],
                 var_name="is_attack",
                 value_name="flows"))

plt.figure(figsize=(10,4))
sns.barplot(data=ft_long, x="Timestamp", y="flows", hue="is_attack")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Time (15-minute windows)")
plt.ylabel("Number of flows")
plt.title("Flows per 15 minutes (benign vs attack)")
plt.tight_layout()
plt.show()



# print(flows_time.head())  # check that we actually have data

# plt.figure(figsize=(10,4))
# flows_time.iloc[30:300].plot(kind="bar")   # first 50 seconds
# plt.xlabel("Time")
# plt.ylabel("Flows per second")
# plt.title("Flows per second (first 50 seconds, benign vs attack)")
# plt.tight_layout()
# plt.show()

