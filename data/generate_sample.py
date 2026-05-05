"""
Sample Dataset Generator — creates a realistic e-commerce/sales dataset
for testing the Autonomous Data Analyst Agent.

Run: python data/generate_sample.py
Output: data/sample_sales.csv (~1000 rows)
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ROWS = 1200
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_sales.csv")

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Online", "Retail", "Wholesale", "Direct Sales"]
CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
PRODUCTS = {
    "Electronics":   ["Laptop", "Smartphone", "Tablet", "Headphones", "Smart Watch"],
    "Clothing":      ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers"],
    "Home & Garden": ["Coffee Maker", "Blender", "Sofa", "Lamp", "Plant Pot"],
    "Books":         ["Business Strategy", "Python Guide", "Self Help", "Novel", "Cook Book"],
    "Sports":        ["Running Shoes", "Yoga Mat", "Dumbbell Set", "Bicycle", "Tennis Racket"],
    "Beauty":        ["Face Cream", "Perfume", "Lipstick", "Shampoo", "Sunscreen"],
}
CUSTOMER_SEGMENTS = ["Premium", "Standard", "Budget"]

START_DATE = datetime(2023, 1, 1)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_sales_data(n: int = N_ROWS) -> pd.DataFrame:
    records = []
    for i in range(n):
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        region = random.choice(REGIONS)
        channel = random.choice(CHANNELS)
        segment = random.choice(CUSTOMER_SEGMENTS)

        # Price logic
        base_prices = {
            "Electronics": (150, 1500),
            "Clothing": (20, 200),
            "Home & Garden": (30, 800),
            "Books": (10, 60),
            "Sports": (25, 600),
            "Beauty": (10, 150),
        }
        lo, hi = base_prices[category]
        unit_price = round(np.random.uniform(lo, hi), 2)

        # Quantity
        quantity = int(np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.1, 0.06, 0.04, 0.03, 0.02]))

        # Discount
        discount_pct = round(random.choice([0, 0, 0, 5, 10, 15, 20, 25]), 1)
        discount_amount = round(unit_price * quantity * discount_pct / 100, 2)
        revenue = round(unit_price * quantity - discount_amount, 2)

        # Cost and profit
        cost_ratio = np.random.uniform(0.45, 0.75)
        cost = round(unit_price * quantity * cost_ratio, 2)
        profit = round(revenue - cost, 2)

        # Date
        date = START_DATE + timedelta(days=random.randint(0, 364))

        # Customer satisfaction (1–5)
        satisfaction = max(1, min(5, int(np.random.normal(3.8, 0.8))))

        # Return flag (higher for low satisfaction)
        return_prob = 0.02 if satisfaction >= 4 else 0.12
        returned = random.random() < return_prob

        # Introduce some missing values (~3%)
        if random.random() < 0.03:
            satisfaction = None
        if random.random() < 0.02:
            discount_pct = None

        records.append(
            {
                "order_id": f"ORD-{10000 + i}",
                "date": date.strftime("%Y-%m-%d"),
                "region": region,
                "sales_channel": channel,
                "customer_segment": segment,
                "category": category,
                "product_name": product,
                "unit_price": unit_price,
                "quantity": quantity,
                "discount_pct": discount_pct,
                "discount_amount": discount_amount,
                "revenue": revenue,
                "cost": cost,
                "profit": profit,
                "profit_margin_pct": round(profit / revenue * 100, 2) if revenue > 0 else 0,
                "customer_satisfaction": satisfaction,
                "returned": returned,
            }
        )

    df = pd.DataFrame(records)

    # Sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating {N_ROWS}-row sales dataset...")
    df = generate_sales_data(N_ROWS)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nSample:")
    print(df.head(3).to_string())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print(f"\nMissing values:\n{df.isnull().sum()}")
