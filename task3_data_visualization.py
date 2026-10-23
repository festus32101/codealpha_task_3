
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Set up
INPUT_FILE = "Reviews.csv"
OUTPUT_DIR = Path("task3_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# 2. Load data
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 3. Prepare columns for visualization
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

# Unix timestamps are in seconds
df["ReviewDate"] = pd.to_datetime(
    df["Time"], unit="s", errors="coerce", utc=True
).dt.tz_localize(None)

df["ReviewYear"] = df["ReviewDate"].dt.year

# Use valid ratings for rating-based charts
ratings = df.dropna(subset=["Score"]).copy()
ratings = ratings[ratings["Score"].between(1, 5)]

# 4. Calculate key indicators
total_reviews = len(df)
average_rating = ratings["Score"].mean()
five_star_pct = (ratings["Score"] == 5).mean() * 100
unique_products = df["ProductId"].nunique()

# 5. Chart 1: Rating distribution
rating_counts = ratings["Score"].value_counts().reindex(
    [1, 2, 3, 4, 5], fill_value=0
)

plt.figure(figsize=(8, 5))
sns.barplot(x=rating_counts.index, y=rating_counts.values)
plt.title("Amazon Fine Food Reviews: Rating Distribution")
plt.xlabel("Star rating")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_rating_distribution.png", dpi=180)
plt.close()

# 6. Chart 2: Review volume by year
yearly = (
    df.dropna(subset=["ReviewYear"])
    .groupby("ReviewYear")
    .agg(
        ReviewCount=("Id", "count"),
        AverageRating=("Score", "mean")
    )
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly, x="ReviewYear", y="ReviewCount", marker="o")
plt.title("Review Volume by Year")
plt.xlabel("Year")
plt.ylabel("Number of reviews")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_reviews_by_year.png", dpi=180)
plt.close()

# 7. Chart 3: Average rating by year
plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly, x="ReviewYear", y="AverageRating", marker="o")
plt.title("Average Star Rating by Year")
plt.xlabel("Year")
plt.ylabel("Average rating")
plt.ylim(1, 5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_average_rating_by_year.png", dpi=180)
plt.close()

# 8. Chart 4: Ten most-reviewed products
top_products = (
    df.groupby("ProductId")
    .size()
    .nlargest(10)
    .sort_values()
    .rename("ReviewCount")
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_products,
    x="ReviewCount",
    y="ProductId"
)
plt.title("Top 10 Products by Number of Reviews")
plt.xlabel("Number of reviews")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_top_products.png", dpi=180)
plt.close()

# 9. Dashboard: combine charts and key indicators
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Amazon Fine Food Reviews — Data Story Dashboard",
    fontsize=17,
    fontweight="bold"
)

# Rating distribution
sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Rating Distribution")
axes[0, 0].set_xlabel("Stars")
axes[0, 0].set_ylabel("Review count")

# Review volume trend
sns.lineplot(
    data=yearly, x="ReviewYear", y="ReviewCount",
    marker="o", ax=axes[0, 1]
)
axes[0, 1].set_title("Review Volume by Year")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Review count")

# Average rating trend
sns.lineplot(
    data=yearly, x="ReviewYear", y="AverageRating",
    marker="o", ax=axes[1, 0]
)
axes[1, 0].set_title("Average Rating by Year")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Average rating")
axes[1, 0].set_ylim(1, 5)

# Top products
sns.barplot(
    data=top_products,
    x="ReviewCount",
    y="ProductId",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Top 10 Products")
axes[1, 1].set_xlabel("Review count")
axes[1, 1].set_ylabel("Product ID")

fig.text(
    0.5, 0.01,
    f"Reviews: {total_reviews:,} | "
    f"Average rating: {average_rating:.2f}/5 | "
    f"5-star reviews: {five_star_pct:.1f}% | "
    f"Products: {unique_products:,}",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(
    OUTPUT_DIR / "05_reviews_dashboard.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close(fig)

# 10. Save supporting tables
yearly.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
top_products.to_csv(OUTPUT_DIR / "top_products.csv", index=False)
rating_counts.rename_axis("Score").rename(
    "ReviewCount"
).to_csv(OUTPUT_DIR / "rating_counts.csv")

print("Visualization task completed.")
print(f"Average rating: {average_rating:.2f}/5")
print(f"Five-star review percentage: {five_star_pct:.2f}%")
print(f"All charts saved in: {OUTPUT_DIR.resolve()}")
