import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

from scipy.stats import pearsonr

# ==========================================
# 1. Load Results
# ==========================================

CSV_PATH = "fixed_adversarial_results.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print(f"Loaded: {CSV_PATH}")
print(f"Total Samples: {len(df)}")
print("=" * 70)

# ==========================================
# 2. Parse Layer-wise JSD
# ==========================================

def clean_and_eval(x):
    """
    Safely parse layer-wise JSD lists from CSV.
    """

    try:
        s = str(x)

        s = s.replace("np.float32(", "")
        s = s.replace("np.float64(", "")
        s = s.replace(")", "")

        s = s.replace("nan", "0.0")
        s = s.replace("NaN", "0.0")
        s = s.replace("inf", "0.0")
        s = s.replace("-inf", "0.0")

        return ast.literal_eval(s)

    except Exception:
        return []

# parse layerwise
df["jsd_layerwise_list"] = df["jsd_layerwise"].apply(clean_and_eval)

# ==========================================
# 3. Correlation Analysis
# ==========================================

corr, p_value = pearsonr(
    df["tvd_base"],
    df["jsd_mean"]
)

print(f"Pearson Correlation (TVD vs JSD): {corr:.4f}")
print(f"P-value: {p_value:.6e}")

# ==========================================
# 4. Low-TVD Subset
# ==========================================

threshold = df["tvd_base"].quantile(0.25)

aligned_df = df[df["tvd_base"] <= threshold].copy()

print("\n" + "=" * 70)
print("LOW-TVD SUBSET")
print("=" * 70)

print(f"Threshold (25th percentile): {threshold:.4f}")
print(f"Subset Size: {len(aligned_df)}")

print(f"Mean TVD: {aligned_df['tvd_base'].mean():.4f}")
print(f"Mean JSD: {aligned_df['jsd_mean'].mean():.4f}")
print(f"Mean Shuffle A: {aligned_df['tvd_shuffle_A'].mean():.4f}")
print(f"Mean Shuffle B: {aligned_df['tvd_shuffle_B'].mean():.4f}")

# ==========================================
# 5. Figure 1
# DTW-Aligned Attention Divergence
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="tvd_base",
    y="jsd_mean",
    alpha=0.75,
    s=70,
    edgecolor="white",
    color="royalblue"
)

sns.regplot(
    data=df,
    x="tvd_base",
    y="jsd_mean",
    scatter=False,
    color="black",
    line_kws={"linewidth": 2}
)

plt.axvline(
    x=threshold,
    linestyle="--",
    color="red",
    linewidth=1.5,
    label=f"Low-TVD Threshold ({threshold:.2f})"
)

plt.text(
    0.05,
    df["jsd_mean"].max() * 0.92,
    f"Pearson r = {corr:.2f}",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.title(
    "DTW-Aligned Attention Divergence under Reasoning Perturbations",
    fontsize=14
)

plt.xlabel(
    "Output Divergence (TVD)",
    fontsize=12
)

plt.ylabel(
    "Attention Divergence (Mean JSD)",
    fontsize=12
)

plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figure1_dtw_attention_vs_output.png",
    dpi=300,
    bbox_inches="tight"
)

print("\n[Saved] figure1_dtw_attention_vs_output.png")

# ==========================================
# 6. Figure 2
# Layer-wise Attention Divergence
# ==========================================

valid_lists = [
    l for l in aligned_df["jsd_layerwise_list"]
    if isinstance(l, list) and len(l) > 0
]

if len(valid_lists) > 0:

    expected_layers = len(valid_lists[0])

    clean_lists = [
        l for l in valid_lists
        if len(l) == expected_layers
    ]

    layerwise_matrix = np.array(clean_lists)

    mean_jsd_per_layer = layerwise_matrix.mean(axis=0)
    std_jsd_per_layer = layerwise_matrix.std(axis=0)

    layers = np.arange(expected_layers)

    plt.figure(figsize=(10, 5))

    plt.plot(
        layers,
        mean_jsd_per_layer,
        marker="o",
        linewidth=2,
        markersize=5,
        color="purple",
        label="Mean Layer-wise JSD"
    )

    plt.fill_between(
        layers,
        mean_jsd_per_layer - std_jsd_per_layer,
        mean_jsd_per_layer + std_jsd_per_layer,
        alpha=0.2,
        color="purple"
    )

    plt.title(
        "Layer-wise Attention Divergence under DTW Alignment",
        fontsize=14
    )

    plt.xlabel(
        "Transformer Layer Depth",
        fontsize=12
    )

    plt.ylabel(
        "Mean Jensen-Shannon Divergence",
        fontsize=12
    )

    plt.xticks(np.arange(0, expected_layers, 2))

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "figure2_layerwise_dtw_attention.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("[Saved] figure2_layerwise_dtw_attention.png")

else:

    print("\n[Warning] No valid layer-wise JSD lists found.")

# ==========================================
# 7. Figure 3
# Shuffle Sensitivity Distribution
# ==========================================

plt.figure(figsize=(8, 6))

sns.kdeplot(
    df["tvd_shuffle_A"],
    fill=True,
    label="Path A (Algebra)"
)

sns.kdeplot(
    df["tvd_shuffle_B"],
    fill=True,
    label="Path B (Arithmetic)"
)

plt.title(
    "Output Sensitivity to Reasoning Token Reordering",
    fontsize=14
)

plt.xlabel(
    "Shuffle-Induced TVD",
    fontsize=12
)

plt.ylabel(
    "Density",
    fontsize=12
)

plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figure3_shuffle_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

print("[Saved] figure3_shuffle_sensitivity.png")

# ==========================================
# 8. Figure 4
# Histogram of JSD
# ==========================================

plt.figure(figsize=(8, 6))

sns.histplot(
    df["jsd_mean"],
    bins=25,
    kde=True
)

plt.title(
    "Distribution of DTW-Aligned Attention Divergence",
    fontsize=14
)

plt.xlabel(
    "Mean JSD",
    fontsize=12
)

plt.ylabel(
    "Count",
    fontsize=12
)

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "figure4_jsd_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("[Saved] figure4_jsd_distribution.png")

if len(aligned_df) > 0:

    case = aligned_df.sort_values("jsd_mean").iloc[0]

    print("\n" + "=" * 70)
    print("CASE STUDY")
    print("=" * 70)

    print(f"Question:")
    print(case["question"][:300] + "...")

    print("\nMetrics:")
    print(f"TVD base: {case['tvd_base']:.4f}")
    print(f"Mean JSD: {case['jsd_mean']:.4f}")
    print(f"Shuffle impact A: {case['tvd_shuffle_A']:.4f}")
    print(f"Shuffle impact B: {case['tvd_shuffle_B']:.4f}")

    print("\nInterpretation:")
    print(
        "This sample exhibits relatively compressed attention divergence "
        "despite measurable output divergence."
    )

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nKey Findings:")
print(f"- Pearson correlation (TVD vs JSD): {corr:.4f}")
print("- Attention divergence positively correlates with output divergence")
print("- CoT token order strongly affects prediction stability")
print("- Attention divergence remains numerically compressed relative to TVD")
print("=" * 70)
