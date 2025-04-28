import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

# Load results
with open("results/evaluation_results.json", "r") as f:
    results = json.load(f)

# Create a DataFrame for easier analysis
df = pd.DataFrame([
    {
        "example_id": r["example_id"],
        "sentence": r["sentence"],
        "entity_token": r["entity_token"],
        "entity_type": r["entity_type"],
        "lfs": r["lfs"] if r["lfs"] is not None else np.nan,
        "has_lime": r["lime_explanation"] is not None
    }
    for r in results
])

# Overall statistics
print("=== LIAP & LFS Analysis ===")
print(f"Total examples analyzed: {len(df)}")
print(f"Average LFS: {df['lfs'].mean():.4f}")
print(f"Median LFS: {df['lfs'].median():.4f}")
print(f"LFS Standard Deviation: {df['lfs'].std():.4f}")
print()

# LFS by entity type
print("=== LFS by Entity Type ===")
entity_stats = df.groupby("entity_type")["lfs"].agg(["mean", "median", "std", "count"])
entity_stats = entity_stats.round(4)
print(tabulate(entity_stats, headers="keys", tablefmt="grid"))
print()

# Find best and worst examples
print("=== Top 5 Examples by LFS ===")
top_examples = df.sort_values("lfs", ascending=False).head(5)
for _, row in top_examples.iterrows():
    print(f"Example {row['example_id']}: Entity '{row['entity_token']}' ({row['entity_type']}) - LFS: {row['lfs']:.4f}")
    print(f"Sentence: {row['sentence']}")
    print()

print("=== Bottom 5 Examples by LFS ===")
bottom_examples = df.dropna(subset=["lfs"]).sort_values("lfs").head(5)
for _, row in bottom_examples.iterrows():
    print(f"Example {row['example_id']}: Entity '{row['entity_token']}' ({row['entity_type']}) - LFS: {row['lfs']:.4f}")
    print(f"Sentence: {row['sentence']}")
    print()

# Compare LIAP with LIME
lime_comparison = df.groupby("has_lime")["lfs"].agg(["mean", "median", "std", "count"])
print("=== LIAP vs LIME Comparison ===")
print(tabulate(lime_comparison, headers="keys", tablefmt="grid"))
print()

# Add strategy analysis
if "best_strategy" in df.columns:
    print("=== Pruning Strategy Analysis ===")
    strategy_stats = df.groupby("best_strategy")["lfs"].agg(["mean", "median", "std", "count"])
    strategy_stats = strategy_stats.round(4)
    print(tabulate(strategy_stats, headers="keys", tablefmt="grid"))
    print()
    
    # Visualize strategy performance
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="best_strategy", y="lfs", data=df)
    plt.title("LFS by Pruning Strategy")
    plt.tight_layout()
    plt.savefig("results/lfs_by_strategy.png")

# Generate additional visualizations
plt.figure(figsize=(12, 8))
sns.boxplot(x="entity_type", y="lfs", data=df)
plt.title("LFS Distribution by Entity Type")
plt.tight_layout()
plt.savefig("results/lfs_boxplot_by_entity.png")

# Correlation between sentence length and LFS
df["sentence_length"] = df["sentence"].apply(len)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="sentence_length", y="lfs", hue="entity_type", data=df)
plt.title("LFS vs Sentence Length")
plt.tight_layout()
plt.savefig("results/lfs_vs_sentence_length.png")

print("Analysis complete. Additional visualizations saved to the 'results' directory.")
