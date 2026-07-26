from pathlib import Path

import pandas as pd


# Locate the project folder and customer segmentation CSV file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "outputs" / "rfm_with_predictions.csv"


def load_customer_data() -> pd.DataFrame:
    """Load and validate the customer segmentation dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset was not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Segment",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the customer count and average RFM values for each segment."""

    summary = (
        df.groupby("Segment")
        .agg(
            Customer_Count=("CustomerID", "nunique"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(2)
        .sort_values("Total_Revenue", ascending=False)
    )

    return summary


def get_highest_revenue_segment(df: pd.DataFrame) -> dict:
    """Return the segment contributing the highest total revenue."""

    revenue_by_segment = (
        df.groupby("Segment")["Monetary"]
        .sum()
        .sort_values(ascending=False)
    )

    if revenue_by_segment.empty:
        raise ValueError("The dataset does not contain segment revenue data.")

    segment_name = revenue_by_segment.index[0]
    revenue = float(revenue_by_segment.iloc[0])
    total_revenue = float(revenue_by_segment.sum())

    revenue_share = (
        revenue / total_revenue * 100
        if total_revenue > 0
        else 0
    )

    return {
        "segment": segment_name,
        "revenue": round(revenue, 2),
        "revenue_share_percentage": round(revenue_share, 2),
    }


def get_customer_count_by_segment(df: pd.DataFrame) -> dict:
    """Return the number of unique customers in every segment."""

    counts = (
        df.groupby("Segment")["CustomerID"]
        .nunique()
        .sort_values(ascending=False)
    )

    return counts.to_dict()


def get_marketing_recommendation(segment):
    recommendations = {
        "Loyal Customers": "Reward them with loyalty benefits...",
        "Potential Loyalists": "Use personalised offers...",
        "At Risk Customers": "Use win-back campaigns...",
        "Low Value Customers": "Use low-cost automated campaigns...",
    }

    return recommendations.get(
        segment,
        "No recommendation is available for this segment.",
    )

def get_all_marketing_recommendations():
    """
    Return marketing recommendations for every customer segment.
    """
    return {
        "Loyal Customers": (
            "Invest in loyalty rewards, early access, referrals, "
            "and personalised premium offers."
        ),
        "Potential Loyalists": (
            "Use personalised product recommendations, limited-time "
            "offers, and repeat-purchase incentives."
        ),
        "At Risk Customers": (
            "Use win-back campaigns, reminder emails, targeted discounts, "
            "and customer-feedback surveys."
        ),
        "Low Value Customers": (
            "Use inexpensive automated campaigns and bundle offers. "
            "Avoid assigning a large portion of the marketing budget."
        ),
    }