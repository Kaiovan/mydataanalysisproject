"""
Load processed Parquet data into PostgreSQL data warehouse.

Reads the Spark-processed Parquet files and loads them into the
PostgreSQL tables created by sql/schema.sql.

Usage:
    python src/load_to_postgres.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import yaml


def load_config():
    """Load database config from config.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_connection(config):
    """Create a PostgreSQL connection."""
    db = config["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        database=db["database"],
        user=db["username"],
        password=db["password"],
    )


def load_parquet(path):
    """Load a Parquet directory (may contain multiple part files)."""
    path = Path(path)
    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if parquet_files:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            return pd.concat(dfs, ignore_index=True)
    elif path.exists() and path.suffix == ".parquet":
        return pd.read_parquet(path)
    return None


def load_product_catalog(config):
    """Load product catalog CSV."""
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "product_catalog.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return None


def truncate_and_load(conn, table_name, df, columns=None):
    """Truncate table and bulk insert DataFrame rows."""
    if df is None or df.empty:
        print(f"  Skipping {table_name} - no data")
        return

    cur = conn.cursor()

    # Truncate with cascade to handle foreign keys
    cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")

    if columns:
        df = df[[c for c in columns if c in df.columns]]

    cols = list(df.columns)
    col_str = ", ".join(cols)
    template = "(" + ", ".join(["%s"] * len(cols)) + ")"

    # Convert to list of tuples, handling NaN -> None
    data = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

    # Bulk insert
    query = f"INSERT INTO {table_name} ({col_str}) VALUES %s ON CONFLICT DO NOTHING"
    execute_values(cur, query, data, template=template, page_size=1000)

    conn.commit()
    print(f"  Loaded {len(data)} rows into {table_name}")


def main():
    print("=" * 70)
    print("LOADING DATA INTO POSTGRESQL")
    print("=" * 70)

    config = load_config()
    base_path = Path(__file__).parent.parent / "data" / "processed"

    # Test connection
    try:
        conn = get_connection(config)
        print("  Connected to PostgreSQL")
    except Exception as e:
        print(f"  ERROR: Cannot connect to PostgreSQL: {e}")
        print("  Make sure PostgreSQL is running: docker-compose up -d postgres")
        sys.exit(1)

    # 1. Load dim_date (generate from daily metrics dates)
    print("\nLoading dim_date...")
    daily_df = load_parquet(base_path / "fact_daily_metrics")
    if daily_df is not None:
        date_col = "event_date" if "event_date" in daily_df.columns else "metric_date"
        dates = pd.to_datetime(daily_df[date_col].unique())
        date_df = pd.DataFrame({
            "date_key": dates,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "quarter": dates.quarter,
            "day_of_week": dates.dayofweek,
            "day_name": [d.strftime("%A") for d in dates],
            "month_name": [d.strftime("%B") for d in dates],
            "is_weekend": [d.dayofweek >= 5 for d in dates],
            "is_holiday": False,
        })
        truncate_and_load(conn, "dim_date", date_df)

    # 2. Load dim_users
    print("\nLoading dim_users...")
    users_df = load_parquet(base_path / "dim_user_metrics")
    if users_df is not None:
        user_cols = [
            "user_id", "first_seen", "last_seen", "num_sessions", "total_events",
            "num_active_days", "total_purchases", "total_revenue",
            "unique_products_viewed", "days_active", "avg_events_per_session",
            "avg_revenue_per_purchase", "user_segment",
        ]
        available = [c for c in user_cols if c in users_df.columns]
        truncate_and_load(conn, "dim_users", users_df, available)

    # 3. Load dim_products
    print("\nLoading dim_products...")
    products_df = load_product_catalog(config)
    if products_df is not None:
        prod_cols = [
            "product_id", "product_name", "category", "price",
            "stock_quantity", "rating", "num_reviews",
        ]
        available = [c for c in prod_cols if c in products_df.columns]
        truncate_and_load(conn, "dim_products", products_df, available)
    else:
        print("  WARNING: product_catalog.csv not found, skipping dim_products")

    # 4. Load dim_product_metrics
    print("\nLoading dim_product_metrics...")
    prod_metrics_df = load_parquet(base_path / "dim_product_metrics")
    if prod_metrics_df is not None:
        truncate_and_load(conn, "dim_product_metrics", prod_metrics_df, [
            "product_id", "category", "total_interactions", "unique_users",
            "unique_sessions", "num_clicks", "num_add_to_cart", "avg_price",
            "click_to_cart_rate",
        ])

    # 5. Load fact_events
    print("\nLoading fact_events...")
    events_df = load_parquet(base_path / "fact_events")
    if events_df is not None:
        event_cols = [
            "event_id", "session_id", "user_id", "timestamp", "event_date",
            "event_type", "page_type", "product_id", "category", "price",
            "cart_size", "cart_value", "num_items", "payment_method",
            "device", "browser", "ip_address", "referrer", "search_query",
            "event_hour", "event_day_of_week", "event_month", "event_year",
        ]
        available = [c for c in event_cols if c in events_df.columns]
        truncate_and_load(conn, "fact_events", events_df, available)

    # 6. Load fact_sessions
    print("\nLoading fact_sessions...")
    sessions_df = load_parquet(base_path / "fact_sessions")
    if sessions_df is not None:
        session_cols = [
            "session_id", "user_id", "event_date", "session_start", "session_end",
            "session_duration_seconds", "num_events", "num_unique_event_types",
            "device", "browser", "referrer", "num_page_views", "num_product_clicks",
            "num_add_to_cart", "num_purchases", "total_cart_value", "items_purchased",
            "avg_time_between_events", "converted",
        ]
        available = [c for c in session_cols if c in sessions_df.columns]
        truncate_and_load(conn, "fact_sessions", sessions_df, available)

    # 7. Load fact_daily_metrics
    print("\nLoading fact_daily_metrics...")
    if daily_df is not None:
        # Rename event_date to metric_date if needed
        if "event_date" in daily_df.columns and "metric_date" not in daily_df.columns:
            daily_df = daily_df.rename(columns={"event_date": "metric_date"})
        daily_cols = [
            "metric_date", "total_events", "unique_users", "unique_sessions",
            "total_purchases", "total_revenue", "avg_order_value", "conversion_rate",
        ]
        available = [c for c in daily_cols if c in daily_df.columns]
        truncate_and_load(conn, "fact_daily_metrics", daily_df, available)

    # 8. Load ML predictions if available
    ml_path = Path(__file__).parent.parent / "data" / "ml_output" / "predictions"

    for pred_file, table_name in [
        ("conversion_predictions.parquet", "ml_conversion_predictions"),
        ("churn_predictions.parquet", "ml_churn_predictions"),
        ("ltv_predictions.parquet", "ml_ltv_predictions"),
    ]:
        pred_path = ml_path / pred_file
        if pred_path.exists():
            print(f"\nLoading {table_name}...")
            pred_df = pd.read_parquet(pred_path)
            truncate_and_load(conn, table_name, pred_df)
        else:
            print(f"\nSkipping {table_name} - {pred_file} not found")

    conn.close()
    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
