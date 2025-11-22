import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ddos_dataset.csv')
df.columns = [col.strip() for col in df.columns]

print(df.columns[2])
print(df.columns[4])

n_flows = df[df.columns[1]].nunique()
n_src_ip = df[df.columns[2]].nunique()
n_dst_ip = df[df.columns[4]].nunique()
print(f"Number of flows: {n_flows}")
print(f"Number of unique source IPs: {n_src_ip}")
print(f"Number of unique destination IPs: {n_dst_ip}")

n_benign = df[df.columns[-1]].value_counts().get('benign', 0)
print(f"Number of benign packets: {n_benign}")
n_benign_flows = df[df[df.columns[-1]] == 'benign'].groupby(df.columns[1]).size()
print(f"Number of benign flows: {n_benign_flows.nunique()}")


#print(data.head())