"""
PySpark ETL Pipeline for Clickstream Data

This script demonstrates:
- Reading data with Spark
- Data cleaning and transformation
- Sessionization using window functions
- Aggregations and metrics calculation
- Joining with dimension tables
- Writing to data warehouse with partitioning
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, max as spark_max, min as spark_min,
    row_number, lag, unix_timestamp, when, round as spark_round,
    to_date, hour, dayofweek, month, year,
    countDistinct, first, last, dense_rank, datediff,
    current_timestamp, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType
)
from pathlib import Path
import sys


class ClickstreamETL:
    """ETL Pipeline for processing clickstream data"""

    def __init__(self, app_name: str = "ClickstreamETL"):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")

        print(f"Spark version: {self.spark.version}")
        print(f"Spark UI available at: {self.spark.sparkContext.uiWebUrl}")

    def read_clickstream_data(self, input_path: str):
        """
        Read clickstream data from CSV/JSON

        Args:
            input_path: Path to input data file

        Returns:
            DataFrame with clickstream events
        """
        print(f"\nReading clickstream data from {input_path}...")

        # Read with schema inference
        if input_path.endswith('.csv'):
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(input_path)
        elif input_path.endswith('.jsonl') or input_path.endswith('.json'):
            df = self.spark.read.json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

        print(f"Loaded {df.count():,} events")
        print("\nSchema:")
        df.printSchema()

        return df

    def clean_data(self, df):
        """
        Clean and validate clickstream data

        Args:
            df: Raw clickstream DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning data...")

        # Convert timestamp to proper timestamp type
        df_cleaned = df.withColumn(
            "timestamp",
            col("timestamp").cast(TimestampType())
        )

        # Add date partition column
        df_cleaned = df_cleaned.withColumn("event_date", to_date(col("timestamp")))

        # Add time-based features
        df_cleaned = df_cleaned \
            .withColumn("event_hour", hour(col("timestamp"))) \
            .withColumn("event_day_of_week", dayofweek(col("timestamp"))) \
            .withColumn("event_month", month(col("timestamp"))) \
            .withColumn("event_year", year(col("timestamp")))

        # Remove any null session_ids or user_ids
        df_cleaned = df_cleaned.filter(
            col("session_id").isNotNull() & col("user_id").isNotNull()
        )

        # Cast numeric columns
        numeric_cols = ['price', 'cart_value']
        for col_name in numeric_cols:
            if col_name in df_cleaned.columns:
                df_cleaned = df_cleaned.withColumn(
                    col_name,
                    col(col_name).cast(DoubleType())
                )

        print(f"After cleaning: {df_cleaned.count():,} events")

        return df_cleaned

    def sessionize_events(self, df):
        """
        Create session-level metrics using window functions

        Args:
            df: Cleaned clickstream DataFrame

        Returns:
            DataFrame with session metrics
        """
        print("\nSessionizing events...")

        # Define window for session ordering
        session_window = Window.partitionBy("session_id").orderBy("timestamp")

        # Add session sequence number and calculate time between events
        df_session = df \
            .withColumn("event_sequence", row_number().over(session_window)) \
            .withColumn("prev_timestamp", lag("timestamp", 1).over(session_window)) \
            .withColumn(
                "time_since_prev_event",
                when(
                    col("prev_timestamp").isNotNull(),
                    unix_timestamp("timestamp") - unix_timestamp("prev_timestamp")
                ).otherwise(0)
            )

        # Aggregate session-level metrics
        session_metrics = df_session.groupBy("session_id", "user_id", "event_date") \
            .agg(
                count("*").alias("num_events"),
                countDistinct("event_type").alias("num_unique_event_types"),
                spark_min("timestamp").alias("session_start"),
                spark_max("timestamp").alias("session_end"),
                first("device").alias("device"),
                first("browser").alias("browser"),
                first("referrer").alias("referrer"),
                spark_sum(
                    when(col("event_type") == "page_view", 1).otherwise(0)
                ).alias("num_page_views"),
                spark_sum(
                    when(col("event_type") == "product_click", 1).otherwise(0)
                ).alias("num_product_clicks"),
                spark_sum(
                    when(col("event_type") == "add_to_cart", 1).otherwise(0)
                ).alias("num_add_to_cart"),
                spark_sum(
                    when(col("event_type") == "purchase", 1).otherwise(0)
                ).alias("num_purchases"),
                spark_max("cart_value").alias("total_cart_value"),
                spark_max("num_items").alias("items_purchased"),
                avg("time_since_prev_event").alias("avg_time_between_events")
            )

        # Calculate session duration
        session_metrics = session_metrics.withColumn(
            "session_duration_seconds",
            unix_timestamp("session_end") - unix_timestamp("session_start")
        )

        # Add conversion flag (did user purchase?)
        session_metrics = session_metrics.withColumn(
            "converted",
            when(col("num_purchases") > 0, 1).otherwise(0)
        )

        print(f"Created metrics for {session_metrics.count():,} sessions")

        return session_metrics

    def calculate_product_metrics(self, df):
        """
        Calculate product-level metrics

        Args:
            df: Clickstream DataFrame

        Returns:
            DataFrame with product metrics
        """
        print("\nCalculating product metrics...")

        # Filter to product-related events
        product_events = df.filter(col("product_id").isNotNull())

        # Aggregate by product
        product_metrics = product_events.groupBy("product_id", "category") \
            .agg(
                count("*").alias("total_interactions"),
                countDistinct("user_id").alias("unique_users"),
                countDistinct("session_id").alias("unique_sessions"),
                spark_sum(
                    when(col("event_type") == "product_click", 1).otherwise(0)
                ).alias("num_clicks"),
                spark_sum(
                    when(col("event_type") == "add_to_cart", 1).otherwise(0)
                ).alias("num_add_to_cart"),
                avg("price").alias("avg_price")
            )

        # Calculate conversion rate
        product_metrics = product_metrics.withColumn(
            "click_to_cart_rate",
            when(
                col("num_clicks") > 0,
                spark_round(col("num_add_to_cart") / col("num_clicks"), 4)
            ).otherwise(0)
        )

        print(f"Calculated metrics for {product_metrics.count():,} products")

        return product_metrics

    def calculate_user_metrics(self, df):
        """
        Calculate user-level metrics and user segments

        Args:
            df: Clickstream DataFrame

        Returns:
            DataFrame with user metrics
        """
        print("\nCalculating user metrics...")

        # Aggregate by user
        user_metrics = df.groupBy("user_id") \
            .agg(
                countDistinct("session_id").alias("num_sessions"),
                count("*").alias("total_events"),
                countDistinct("event_date").alias("num_active_days"),
                spark_min("timestamp").alias("first_seen"),
                spark_max("timestamp").alias("last_seen"),
                spark_sum(
                    when(col("event_type") == "purchase", 1).otherwise(0)
                ).alias("total_purchases"),
                spark_sum(
                    when(col("cart_value").isNotNull(), col("cart_value")).otherwise(0)
                ).alias("total_revenue"),
                countDistinct(
                    when(col("product_id").isNotNull(), col("product_id"))
                ).alias("unique_products_viewed")
            )

        # Calculate lifetime value and engagement metrics
        user_metrics = user_metrics \
            .withColumn(
                "days_active",
                datediff(col("last_seen"), col("first_seen")) + 1
            ) \
            .withColumn(
                "avg_events_per_session",
                spark_round(col("total_events") / col("num_sessions"), 2)
            ) \
            .withColumn(
                "avg_revenue_per_purchase",
                when(
                    col("total_purchases") > 0,
                    spark_round(col("total_revenue") / col("total_purchases"), 2)
                ).otherwise(0)
            )

        # Create user segments based on behavior
        user_metrics = user_metrics.withColumn(
            "user_segment",
            when(col("total_purchases") >= 5, "High Value")
            .when(col("total_purchases") >= 1, "Converted")
            .when(col("num_sessions") >= 3, "Engaged")
            .otherwise("New/Inactive")
        )

        print(f"Calculated metrics for {user_metrics.count():,} users")

        return user_metrics

    def calculate_daily_metrics(self, df):
        """
        Calculate daily aggregated metrics

        Args:
            df: Clickstream DataFrame

        Returns:
            DataFrame with daily metrics
        """
        print("\nCalculating daily metrics...")

        daily_metrics = df.groupBy("event_date") \
            .agg(
                count("*").alias("total_events"),
                countDistinct("user_id").alias("unique_users"),
                countDistinct("session_id").alias("unique_sessions"),
                spark_sum(
                    when(col("event_type") == "purchase", 1).otherwise(0)
                ).alias("total_purchases"),
                spark_sum(
                    when(col("cart_value").isNotNull(), col("cart_value")).otherwise(0)
                ).alias("total_revenue"),
                avg(
                    when(col("cart_value").isNotNull(), col("cart_value"))
                ).alias("avg_order_value")
            )

        # Add conversion rate
        daily_metrics = daily_metrics.withColumn(
            "conversion_rate",
            spark_round(col("total_purchases") / col("unique_sessions") * 100, 2)
        )

        # Sort by date
        daily_metrics = daily_metrics.orderBy("event_date")

        print(f"Calculated metrics for {daily_metrics.count():,} days")

        return daily_metrics

    def write_to_parquet(self, df, output_path: str, partition_by: list = None):
        """
        Write DataFrame to Parquet with optional partitioning

        Args:
            df: DataFrame to write
            output_path: Output path
            partition_by: List of columns to partition by
        """
        print(f"\nWriting to {output_path}...")

        writer = df.write.mode("overwrite").option("compression", "snappy")

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        writer.parquet(output_path)
        print(f"Successfully wrote data to {output_path}")

    def run_pipeline(self, input_path: str, output_base_path: str):
        """
        Execute the complete ETL pipeline

        Args:
            input_path: Path to raw clickstream data
            output_base_path: Base path for output data
        """
        print("=" * 80)
        print("STARTING CLICKSTREAM ETL PIPELINE")
        print("=" * 80)

        # Step 1: Read data
        df_raw = self.read_clickstream_data(input_path)

        # Step 2: Clean data
        df_clean = self.clean_data(df_raw)

        # Cache the cleaned data as we'll use it multiple times
        df_clean.cache()

        # Step 3: Calculate various metrics
        session_metrics = self.sessionize_events(df_clean)
        product_metrics = self.calculate_product_metrics(df_clean)
        user_metrics = self.calculate_user_metrics(df_clean)
        daily_metrics = self.calculate_daily_metrics(df_clean)

        # Step 4: Write to data warehouse
        output_path = Path(output_base_path)

        # Write fact table (events) - partitioned by date
        self.write_to_parquet(
            df_clean,
            str(output_path / "fact_events"),
            partition_by=["event_date"]
        )

        # Write dimension/metric tables
        self.write_to_parquet(
            session_metrics,
            str(output_path / "fact_sessions"),
            partition_by=["event_date"]
        )

        self.write_to_parquet(
            product_metrics,
            str(output_path / "dim_product_metrics")
        )

        self.write_to_parquet(
            user_metrics,
            str(output_path / "dim_user_metrics")
        )

        self.write_to_parquet(
            daily_metrics,
            str(output_path / "fact_daily_metrics")
        )

        # Display sample insights
        self.display_insights(daily_metrics, session_metrics, user_metrics, product_metrics)

        # Unpersist cache
        df_clean.unpersist()

        print("\n" + "=" * 80)
        print("ETL PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)

    def display_insights(self, daily_metrics, session_metrics, user_metrics, product_metrics):
        """Display sample insights from the processed data"""
        print("\n" + "=" * 80)
        print("SAMPLE INSIGHTS")
        print("=" * 80)

        print("\n--- Daily Metrics (Last 7 Days) ---")
        daily_metrics.orderBy(col("event_date").desc()).limit(7).show()

        print("\n--- User Segment Distribution ---")
        user_metrics.groupBy("user_segment") \
            .agg(
                count("*").alias("num_users"),
                spark_sum("total_revenue").alias("total_revenue")
            ) \
            .orderBy(col("total_revenue").desc()) \
            .show()

        print("\n--- Top 10 Products by Interactions ---")
        product_metrics.orderBy(col("total_interactions").desc()).limit(10).show()

        print("\n--- Session Conversion Statistics ---")
        session_metrics.agg(
            count("*").alias("total_sessions"),
            spark_sum("converted").alias("converted_sessions"),
            spark_round(avg("converted") * 100, 2).alias("conversion_rate_pct"),
            spark_round(avg("session_duration_seconds"), 2).alias("avg_session_duration"),
            spark_round(avg("num_events"), 2).alias("avg_events_per_session")
        ).show()

    def stop(self):
        """Stop the Spark session"""
        self.spark.stop()


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    input_path = str(base_path / "data" / "raw" / "clickstream_events.csv")
    output_path = str(base_path / "data" / "processed")

    # Initialize and run ETL
    etl = ClickstreamETL(app_name="EcommerceClickstreamETL")

    try:
        etl.run_pipeline(input_path, output_path)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        etl.stop()


if __name__ == "__main__":
    main()
