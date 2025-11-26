import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('ddos_dataset.csv')
df.columns = [col.strip() for col in df.columns]

print(df.columns[2])
print(df.columns[4])


# 1. BASIC TRAFFIC STATISTICS
print("=== BASIC TRAFFIC STATISTICS ===")
n_flows = df["FlowID"].nunique()
n_src_ip = df["SourceIP"].nunique()
n_dst_ip = df["DestinationIP"].nunique()
print(f"Number of flows: {n_flows}")
print(f"Number of unique source IPs: {n_src_ip}")
print(f"Number of unique destination IPs: {n_dst_ip}")

# 2. CLASS DISTRIBUTION
print("=== CLASS DISTRIBUTION ===")
labels = df["label"].unique()
print(f"Labels: {labels}")

# n_benign = df[df["label"] == 'benign'].shape[0] 
# print(f"Number of benign packets: {n_benign}")
# n_benign_flows = df[df["label"] == 'benign'].groupby("FlowID").count().shape[0]
# print(f"Number of benign flows: {n_benign_flows}")

n_attack_per_label = {}
n_flows_attack_per_label = {}
for label in labels:
    n_attack_per_label[label] = df[df["label"] == label].shape[0]
    n_flows_attack_per_label[label] = df[df["label"] == label].groupby("FlowID").count().shape[0]

print(n_attack_per_label)
print(n_flows_attack_per_label)

plot_df_n_labels = pd.DataFrame({
    'Number of attacked packets': n_attack_per_label,
    'Number of attacked flows': n_flows_attack_per_label
})


plot_df_n_labels.plot(kind='bar')
plt.title('Distribuition of flow labels')
plt.xlabel('Flow Labels')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


### NON FUNZIONANTE ###
# 3. Distribution of features
print("=== DISTRIBUTION OF FEATURES ===")

features_to_plot = [ 
    "DestinationPort",
    "SourcePort",
    #"FlowDuration", 
    #"TotalFwdPackets",
    #"TotalBackwardPackets",
    "FlowPackets/s",
    #"FlowBytes/s",
    #"PacketLengthMean",
    #"PacketLengthStd",
    #"FwdIATMean",
    #"BwdIATMean",
    #"AveragePacketSize"
]
scaler = StandardScaler()

# Copio tutto il df per mantenere anche 'label'
df_std = df.copy()

# Standardizzo SOLO le feature, rimanendo in un DataFrame
# df_std[features_to_plot] = scaler.fit_transform(df_std[features_to_plot])

n_features = len(features_to_plot)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

plt.figure(figsize=(6 * n_cols, 4 * n_rows))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(n_rows, n_cols, i)
    for label in labels:
        subset = df_std[df_std["label"] == label][feature]
        sns.kdeplot( subset, label=label, linewidth=1.2)
    
    plt.title(f"KDE distribution of {feature} per attack type")
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.tight_layout()

    # Mostra la legenda solo nel primo plot
    if i == 1:
        plt.legend(title="Attack type", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.legend([],[], frameon=False)

plt.suptitle("Distribution of Key Features (Generic Traffic Level)", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # lascia spazio per il suptitle
plt.show()


### Generate additional features ####


