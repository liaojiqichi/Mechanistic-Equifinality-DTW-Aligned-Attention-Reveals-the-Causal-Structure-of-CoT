import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

try:
    df = pd.read_csv("results.csv")
    print(f"Successfully loaded {len(df)} samples.")
except FileNotFoundError:
    print("Error: Could not find 'results.csv'. ")
    exit()

plt.figure(figsize=(8, 6))

tvd_data = pd.DataFrame({
    'Condition': ['Base (Alg vs Arith)'] * len(df) +
                 ['Shuffle Alg'] * len(df) +
                 ['Shuffle Arith'] * len(df),
    'TVD': np.concatenate([df['tvd_base'], df['tvd_shuffle_A'], df['tvd_shuffle_B']])
})

ax1 = sns.boxplot(x='Condition', y='TVD', data=tvd_data, palette="Set2", showfliers=False, width=0.5)
sns.stripplot(x='Condition', y='TVD', data=tvd_data, color=".25", alpha=0.3, size=4, jitter=True)

plt.title("Behavioral Shift Induced by CoT Perturbations", pad=15)
plt.ylabel("Total Variation Distance (TVD)")
plt.xlabel("")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("Fig1_Behavioral_TVD.pdf", dpi=300)
print("Saved Figure 1: Fig1_Behavioral_TVD.pdf")


plt.figure(figsize=(8, 6))

sns.regplot(
    data=df,
    x='tvd_base',
    y='cosine_mean',
    scatter_kws={'alpha':0.6, 'color': '#4C72B0', 's': 50},
    line_kws={'color': '#C44E52', 'linewidth': 2}
)

plt.title("Mechanistic Equifinality: Divergent Outputs,\nConvergent Representations", pad=15)
plt.xlabel("Output Prediction Divergence (TVD)")
plt.ylabel("Hidden State Cosine Similarity")

plt.ylim(0.85, 1.02)
plt.xlim(-0.05, 1.05)

plt.axhline(y=df['cosine_mean'].mean(), color='gray', linestyle='--', alpha=0.7)
plt.text(0.6, df['cosine_mean'].mean() + 0.005, f"Mean Cosine = {df['cosine_mean'].mean():.3f}",
         color='gray', fontsize=12)

plt.tight_layout()
plt.savefig("Fig2_Mechanistic_Equifinality_Scatter.pdf", dpi=300)
print("Saved Figure 2: Fig2_Mechanistic_Equifinality_Scatter.pdf")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.kdeplot(df['cosine_mean'], ax=axes[0], fill=True, color='#55A868', alpha=0.6, linewidth=2)
axes[0].set_title("Cosine Similarity", pad=10)
axes[0].set_xlabel("Mean Cosine")
axes[0].set_ylabel("Density")
axes[0].axvline(x=df['cosine_mean'].mean(), color='black', linestyle='--', alpha=0.5)

sns.kdeplot(df['cka_mean'], ax=axes[1], fill=True, color='#DD8452', alpha=0.6, linewidth=2)
axes[1].set_title("Representation CKA", pad=10)
axes[1].set_xlabel("Mean Linear CKA")
axes[1].set_ylabel("Density")
axes[1].axvline(x=df['cka_mean'].mean(), color='black', linestyle='--', alpha=0.5)

sns.kdeplot(df['attn_jsd_mean'], ax=axes[2], fill=True, color='#8172B3', alpha=0.6, linewidth=2)
axes[2].set_title("Attention Routing", pad=10)
axes[2].set_xlabel("Mean Rollout JSD")
axes[2].set_ylabel("Density")
axes[2].axvline(x=df['attn_jsd_mean'].mean(), color='black', linestyle='--', alpha=0.5)

plt.suptitle("Internal Dynamics under Distinct Reasoning Trajectories", fontsize=18, y=1.08)

plt.tight_layout()
plt.savefig("Fig3_Internal_Dynamics_Density.pdf", dpi=300, bbox_inches='tight')
print("Saved Figure 3: Fig3_Internal_Dynamics_Density.pdf")

print("\nVisualizations generated successfully.")
