# 01_cleaning
# %% [markdown]
# # Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
import os
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import OLSInfluence

# %%
pd.set_option("display.max_columns", None)

# %% [markdown]
# # Load Data

# %%
products_df = pd.read_csv("../data/products_kaggle.csv")

# %%
products_df.head()

# %% [markdown]
# # Data Cleaning

# %%
products = products_df.copy()

# %%
predictor = ['price_usd']
features = ['brand_name', 'size', 'ingredients', 'limited_edition', 'new', 'online_only', 'out_of_stock', 'sephora_exclusive'
            , 'highlights', 'primary_category', 'secondary_category']

# %% [markdown]
# # Check for Duplicates

# %%
products.duplicated().sum()

# %%
products.duplicated(subset=['product_id']).sum()

# %% [markdown]
# # Check for Missing Values

# %%
products[predictor + features].isna().sum()

# %%
new_products = products[predictor + features].dropna()

# %%
new_products.shape

# %% [markdown]
# # Feature Engineering

# %%
new_products["primary_category"].value_counts()

# %%
new_products["primary_category"] = new_products["primary_category"].astype(str).str.strip()

# %%
if "brand_name" in new_products.columns:
       new_products["brand_name"] = new_products["brand_name"].astype(str).str.strip().str.title()

# %%
INGREDIENT_FLAGS = {
        "has_retinol":       r"retinol|retinoid|tretinoin|retinal\b",
        "has_niacinamide":   r"niacinamide|nicotinamide",
        "has_hyaluronic":    r"hyaluronic acid|sodium hyaluronate",
        "has_vitamin_c":     r"ascorbic acid|vitamin c|l-ascorbic|ascorbyl",
        "has_aha_bha":       r"glycolic acid|lactic acid|salicylic acid|mandelic acid|"
                              r"tartaric acid|malic acid|citric acid|aha|bha",
        "has_peptides":      r"peptide|palmitoyl|matrixyl|argireline|acetyl",
        "has_spf":           r"\bspf\b|sunscreen|zinc oxide|titanium dioxide|"
                              r"avobenzone|octinoxate",
        "has_fragrance":     r"\bfragrance\b|\bparfum\b|\bperfume\b",
        "has_alcohol":       r"alcohol denat|sd alcohol|denatured alcohol",
        "has_ceramides":     r"ceramide",
        "has_collagen":      r"collagen|hydrolyzed collagen",
        "has_bakuchiol":     r"bakuchiol",
        "has_vitamin_e":     r"tocopherol|vitamin e",
    }
 

ing = new_products["ingredients"].fillna("").astype(str).str.lower()
for flag, pattern in INGREDIENT_FLAGS.items():
    new_products[flag] = ing.str.contains(pattern, regex=True, na=False).astype(int)
    n = new_products[flag].sum()
    print(f"      {flag:<25}: {n:,} products ({n/len(new_products):.1%})")

# %%
new_products['size'].value_counts()

# %%
import re
import numpy as np

def extract_ml(text):
    """
    Extract size in mL from any volume/weight unit.
    Converts fl oz and oz → mL. Leaves g as-is (approx 1g ≈ 1mL for creams).
    Returns 0.0 if the product is a count item or unparseable.
    """
    if pd.isna(text):
        return 0.0
    s = str(text).lower().strip()

    # Skip count-based products entirely — return 0 so size_count handles them
    if re.search(r"[\d\.]+\s*(?:count|capsule|tablet|piece|ct\.?)", s):
        return 0.0

    # mL (no conversion needed)
    ml_match = re.search(r"([\d\.]+)\s*ml", s)
    if ml_match:
        return float(ml_match.group(1))

    # fl oz → mL (must check before plain oz)
    floz_match = re.search(r"([\d\.]+)\s*fl\.?\s*oz", s)
    if floz_match:
        return round(float(floz_match.group(1)) * 29.5735, 2)

    # oz → mL
    oz_match = re.search(r"([\d\.]+)\s*oz", s)
    if oz_match:
        return round(float(oz_match.group(1)) * 29.5735, 2)

    # g (treat as mL — reasonable for creams/powders)
    g_match = re.search(r"([\d\.]+)\s*g\b", s)
    if g_match:
        return float(g_match.group(1))
    
    slash_match = re.search(r"([\d\.]+)\s*/\s*([\d\.]+)", s)
    if slash_match:
        a, b = float(slash_match.group(1)), float(slash_match.group(2))
        return max(a, b)

    return 0.0


def extract_count(text):
    """
    Extract numeric count for count-based products.
    Returns 0.0 if volume/weight based or unparseable.
    """
    if pd.isna(text):
        return 0.0
    s = str(text).lower().strip()

    # Pattern 1: "N x Item" (e.g. "6 x Eye Masks")
    x_match = re.search(r"([\d\.]+)\s*x\s+\w+", s)
    if x_match:
        return float(x_match.group(1))
    
    slash_count_match = re.search(r"([\d\.]+)\s*/\s*[a-z]", s)
    if slash_count_match:
        return float(slash_count_match.group(1))

    # Pattern 2: explicit count-like units (expanded)
    count_match = re.search(
        r"([\d\.]+)\s*"
        r"(?:count|capsule|capsules|tablet|tablets|piece|pieces|ct\.?|"
        r"pad|pads|wipe|wipes|mask|masks|treatment|treatments|textured towels|individual masks|"
        r"sponge|sponges|pair|pairs|roller|rollers|vials|patches|"
        r"towel|towels|ampoule|ampoules|refill|refills|"
        r"pack|packs|daily pack|daily packs|healing dots|microneedling|stick packs|"
        r"softgel|softgels|soft gel|soft gels|"
        r"gummy|gummies|gummy heart|gummy hearts|"
        r"berry gummy|tangerine|"
        r"vegetarian capsule|vegetarian capsules|"
        r"vegan capsule|vegan capsules|vegan softgel|vegan softgels|"
        r"vegan gummy|vegan gummies|vegan berry|"
        r"week supply|day supply|day\b)",
        s
    )
    if count_match:
        return float(count_match.group(1))

    return 0.0


# Apply
new_products["size_ml"]    = new_products["size"].apply(extract_ml)
new_products["size_count"] = new_products["size"].apply(extract_count)

# ── Checks ───────────────────────────────────────────────────────────────────

size_cols = ["size_ml", "size_count"]
print("Products per size type:")
for c in size_cols:
    n = (new_products[c] > 0).sum()
    print(f"  {c:<20}: {n:,} ({n/len(new_products):.1%})")

# Should be zero — no product should have both a volume and a count
both = ((new_products["size_ml"] > 0) & (new_products["size_count"] > 0)).sum()
print(f"\nProducts with both size_ml and size_count > 0 (should be 0): {both}")

# Unparseable — neither column got a value, but size field exists
neither = (
    new_products["size"].notna() &
    (new_products["size_ml"] == 0) &
    (new_products["size_count"] == 0)
)
print(f"\nUnparseable (has size string but neither column filled): {neither.sum()}")
print(new_products[neither]["size"].value_counts().head(20).to_string())

# %%
new_products['log_price'] = np.log(new_products['price_usd'])

# %% [markdown]
# # EDA

# %%
# Outcome Distribution
sns.histplot(new_products["price_usd"])
plt.title("Price Distibution")
plt.xlabel("Price (USD)")

# %% [markdown]
# Right skewed distribution --> log transformation to make more normal

# %%
sns.histplot(np.log(new_products["price_usd"]))
plt.title("Log Price Distibution")
plt.xlabel("log(Price (USD))")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("QQ Plot of Product Price (Outcome)", fontsize=14, fontweight="bold")
stats.probplot(new_products["price_usd"].dropna(), dist="norm", plot=axes[0])
axes[0].set_title("QQ-Plot of Price")
axes[0].get_lines()[0].set(color='blue', markersize=2, alpha=0.5)
axes[0].get_lines()[1].set(color='red')

stats.probplot(np.log(new_products["price_usd"]).dropna(), dist="norm", plot=axes[1])
axes[1].set_title("QQ-Plot of log(Price)")
axes[1].get_lines()[0].set(color='blue', markersize=2, alpha=0.5)
axes[1].get_lines()[1].set(color='red')

plt.tight_layout()

# %%
# Price by binary features
bin_cols = [
    "sephora_exclusive", "limited_edition", "online_only", "new",
    "has_retinol", "has_niacinamide", "has_hyaluronic", "has_vitamin_c",
    "has_aha_bha", "has_peptides", "has_spf", "has_fragrance",
    "has_ceramides", "has_vitamin_e"
]

# Melt to long format: one row per product × feature combination
long_df = pd.melt(
    new_products[bin_cols + ['log_price']],
    id_vars="log_price",
    var_name="feature",
    value_name="present"
)
long_df["present"] = long_df["present"].map({1: "Yes", 0: "No"})
long_df["feature"] = long_df["feature"].str.replace("has_", "").str.replace("_", " ").str.title()

fig, ax = plt.subplots(figsize=(26, 8))
sns.boxplot(
    data=long_df, x="feature", y="log_price", hue="present",
    hue_order=["Yes", "No"],
    palette= ['#9ad49f', '#ffada9'],
    flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
    ax=ax
)
ax.set(title="log(Price) by Feature Presence — All Binary Predictors",
       xlabel="Binary Features", ylabel="log(Price)")

ax.legend(title="Feature Present", loc="upper right")
plt.tight_layout()

# %%
# Price by categorical features
cat_cols = ["primary_category"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, col in zip(axes, cat_cols):
    order = (new_products.groupby(col)["log_price"]
               .median()
               .sort_values(ascending=False)
               .index)

    sns.boxplot(
        data=new_products, x="log_price", y=col,
        order=order, palette="muted", ax=ax,
        flierprops={"marker": ".", "alpha": 0.2, "markersize": 3},
        orient = 'h'
    )

plt.tight_layout()
plt.show()

# %%
# Correlation Heatmap
numeric_cols = (
    ["log_price"]
    + bin_cols + ["size_ml", "size_count"]
)
numeric_cols = [c for c in numeric_cols if c in new_products.columns]
corr = new_products[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu",
    center=0, vmin=-0.5, vmax=0.5, ax=ax,
    linewidths=0.5, annot_kws={"size": 8}
)
ax.set_title("Correlation Matrix — Numeric Predictors & Outcome", fontsize=13, fontweight="bold")
plt.tight_layout()

# Print high correlations (|r| > 0.3) — potential multicollinearity
high_corr = (corr.abs()
                .where(mask == False)
                .stack()
                .reset_index()
                .rename(columns={0: "r", "level_0": "var1", "level_1": "var2"}))
high_corr = high_corr[
    (high_corr["var1"] != high_corr["var2"]) &
    (high_corr["r"] > 0.30)
].sort_values("r", ascending=False)
if len(high_corr) > 0:
    print(f"\n      High correlations (|r| > 0.30) — watch for VIF issues:")
    print(high_corr.to_string(index=False))

# %%
# Justification for using skincare as the baseline
new_products['primary_category'] = new_products['primary_category'].replace({
    'Men': 'Other',
    'Tools & Brushes': 'Other',
    'Mini Size': 'Other'  # optional, debatable
})


# 1. Count the occurrences of each category
category_counts = new_products["primary_category"].value_counts()


# 2. Create fig and ax objects
fig, ax = plt.subplots(figsize=(8, 6))

# 3. Create the bar plot on the 'ax' object
bars = ax.bar(category_counts.index, category_counts.values, color='skyblue', edgecolor='black')

# 4. Customize the plot using 'ax' methods
ax.set_title('Fruit Sales Overview', fontsize=14)
ax.set_xlabel('Primary Category')
ax.set_ylabel('Count')

# Optional: Add data labels on top of bars
ax.bar_label(bars, padding=3)


# %%
# What % of Fragrance category products have has_fragrance = 1?
new_products.groupby('primary_category')['has_fragrance'].mean()

# %%
new_products.to_csv("../data/new_products.csv")



