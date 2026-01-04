# E-commerce Clickstream Analytics Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange.svg)](https://spark.apache.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data engineering portfolio project demonstrating end-to-end data pipeline development, ETL processing with Apache Spark, data warehouse design, and analytics for e-commerce clickstream data.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Data Warehouse Schema](#data-warehouse-schema)
- [Analytics & Insights](#analytics--insights)
- [Performance Optimizations](#performance-optimizations)
- [What I Learned](#what-i-learned)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project simulates a real-world data engineering scenario for an e-commerce platform. It implements a scalable data pipeline that:

1. **Generates** realistic user clickstream data (page views, clicks, cart actions, purchases)
2. **Processes** the data using Apache Spark with various transformations and aggregations
3. **Stores** the results in a star schema data warehouse
4. **Analyzes** the data to derive business insights and visualizations

The pipeline demonstrates proficiency in distributed data processing, data warehouse design, SQL analytics, and data visualization - key skills for data warehouse engineer roles.

## Architecture

```
┌─────────────────┐
│  Data Generator │ → Simulates user clickstream events
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Raw Data      │ → CSV/JSON files (10K+ sessions, 100K+ events)
│   (S3/Local)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Apache Spark   │ → ETL Processing
│  - Sessionization
│  - Aggregations │
│  - Window Fns   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Warehouse  │ → Star Schema (PostgreSQL/Parquet)
│  - Fact Tables  │                ┌──────────────────┐
│  - Dimensions   │──────────────→ │  Tableau Desktop │
│  - 8 Tableau    │                │  - Journey Funnel│
│    Views        │                │  - Touchpoints   │
└────────┬────────┘                │  - Cohort Analysis│
         │                         └──────────────────┘
         ▼
┌─────────────────┐
│   Analytics     │ → Static reports (Matplotlib/Seaborn)
│   Dashboard     │
└─────────────────┘
```

## Features

### Data Generation
- Realistic user behavior simulation (1000+ users, 500+ products)
- Multiple event types: page views, clicks, add-to-cart, purchases
- Session-based interactions with temporal patterns
- Configurable parameters for scale

### Spark ETL Pipeline
- **Data Cleaning**: Type casting, null handling, date parsing
- **Sessionization**: Using window functions to group events by session
- **Aggregations**: Session-level, user-level, product-level, and daily metrics
- **Joins**: Combining events with product catalogs
- **Partitioning**: Date-based partitioning for query performance
- **Optimization**: Broadcast joins, caching, adaptive query execution

### Data Warehouse Design
- **Star Schema** with fact and dimension tables
- **Fact Tables**:
  - `fact_events` (grain: one event)
  - `fact_sessions` (grain: one session)
  - `fact_daily_metrics` (grain: one day)
- **Dimension Tables**:
  - `dim_users` (user attributes and lifetime metrics)
  - `dim_products` (product catalog)
  - `dim_product_metrics` (aggregated product performance)
  - `dim_date` (date dimension for time analysis)

### Analytics Capabilities
- **Customer Journey Analytics**: Conversion funnel analysis with touchpoint performance
- **User Segmentation**: High Value, Converted, Engaged, New/Inactive customer segments
- **Product Performance**: Category trends, pricing insights, engagement scoring
- **Cohort Retention**: Month-over-month retention tracking
- **Time-based Trends**: Hourly, daily, weekly patterns with WoW comparisons
- **Cart Abandonment**: Recovery analysis with lost revenue tracking
- **Touchpoint Analysis**: Device/browser/referrer performance comparison
- **Tableau Dashboards**: 8 pre-built views for interactive visualization (see [TABLEAU_GUIDE.md](TABLEAU_GUIDE.md))

## Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Core Processing** | Apache Spark (PySpark) | Distributed data processing |
| **Languages** | Python 3.11 | Pipeline development |
| **Database** | PostgreSQL 16 | Data warehouse |
| **Data Formats** | Parquet, CSV, JSON | Storage formats |
| **Visualization** | Tableau | Interactive dashboards (customer journey analytics) |
| **Visualization** | Matplotlib, Seaborn, Plotly | Static analytics reports |
| **Containerization** | Docker, Docker Compose | Reproducible environment |
| **Version Control** | Git | Source control |

## Project Structure

```
mydataanalysisproject/
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/                     # Generated raw data
│   ├── processed/               # Spark output (Parquet)
│   ├── warehouse/               # Data warehouse files
│   └── analytics_output/        # Visualizations and reports
├── docker/
│   └── docker-compose.yml       # Multi-container setup
├── notebooks/                   # Jupyter notebooks for exploration
├── sql/
│   ├── schema.sql              # Data warehouse schema (+ Tableau views)
│   └── analytics_queries.sql   # Sample analytical queries
├── src/
│   ├── data_generation/
│   │   └── clickstream_generator.py
│   ├── spark_processing/
│   │   └── etl_pipeline.py
│   └── analytics/
│       └── dashboard.py
├── tests/                       # Unit tests
├── .env.example                # Environment variables template
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
├── run_pipeline.sh             # Pipeline runner (Linux/Mac)
├── run_pipeline.bat            # Pipeline runner (Windows)
├── TABLEAU_GUIDE.md            # Tableau integration guide
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8+ (3.11 recommended)
- Java 11+ (required for Spark)
- Docker (optional, for containerized setup)
- 8GB+ RAM recommended

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mydataanalysisproject.git
   cd mydataanalysisproject
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Spark installation**
   ```bash
   python -c "from pyspark.sql import SparkSession; print(SparkSession.builder.getOrCreate().version)"
   ```

### Option 2: Docker Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mydataanalysisproject.git
   cd mydataanalysisproject
   ```

2. **Build and start containers**
   ```bash
   docker-compose up -d
   ```

3. **Access Jupyter Lab**
   ```
   http://localhost:8888
   ```

## Usage

### Quick Start - Run Complete Pipeline

**Windows:**
```batch
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. Generate 10,000 sessions (~100K+ events)
2. Process data with Spark
3. Create analytics visualizations

### Step-by-Step Execution

**Step 1: Generate Data**
```bash
python src/data_generation/clickstream_generator.py
```

**Step 2: Run Spark ETL**
```bash
python src/spark_processing/etl_pipeline.py
```

**Step 3: Generate Analytics**
```bash
python src/analytics/dashboard.py
```

### Using Docker

```bash
# Start all services
docker-compose up -d

# Run pipeline inside container
docker exec -it clickstream-analytics bash
./run_pipeline.sh

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Data Pipeline

### Data Generation
The `clickstream_generator.py` script creates realistic e-commerce events:

- **Users**: 1,000 unique users
- **Products**: 500 products across 8 categories
- **Sessions**: 10,000 sessions with 3-20 events each
- **Events**: ~100K+ total events (page_view, product_click, add_to_cart, purchase)

### ETL Processing

The Spark pipeline (`etl_pipeline.py`) performs:

1. **Data Ingestion**: Read CSV/JSON with schema inference
2. **Data Cleaning**:
   - Type casting (timestamp, numeric fields)
   - Null value handling
   - Feature engineering (hour, day_of_week, month)
3. **Transformations**:
   - Session windows and sequencing
   - User lifetime value calculations
   - Product performance metrics
4. **Aggregations**:
   - Session-level metrics (duration, events, conversion)
   - Daily rollups (traffic, revenue, conversion rate)
   - User segments (High Value, Converted, Engaged)
5. **Output**: Partitioned Parquet files

### Key Spark Techniques Demonstrated

```python
# Window functions for sessionization
session_window = Window.partitionBy("session_id").orderBy("timestamp")
df.withColumn("event_sequence", row_number().over(session_window))

# Complex aggregations
session_metrics = df.groupBy("session_id").agg(
    count("*").alias("num_events"),
    max("timestamp").alias("session_end"),
    sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("conversions")
)

# Broadcast joins for small dimensions
df.join(broadcast(dim_products), "product_id")

# Partitioning for performance
df.write.partitionBy("event_date").parquet(output_path)
```

## Data Warehouse Schema

### Star Schema Design

**Fact Tables:**
- `fact_events`: Raw clickstream events (100K+ rows)
- `fact_sessions`: Session-level aggregates (10K rows)
- `fact_daily_metrics`: Daily rollups (30 rows)

**Dimension Tables:**
- `dim_users`: User profiles and lifetime metrics
- `dim_products`: Product catalog
- `dim_product_metrics`: Product performance aggregates
- `dim_date`: Date dimension for time analysis

### Sample Queries

See `sql/analytics_queries.sql` for 20+ analytical queries including:
- Conversion funnel analysis
- Cohort retention
- Product affinity (market basket analysis)
- User segmentation
- Time-series trends
- Cart abandonment rates

## Analytics & Insights

The analytics dashboard (`dashboard.py`) generates:

### Visualizations
1. **Daily Trends**: Users, sessions, revenue, conversion rate
2. **User Segments**: Distribution and revenue by segment
3. **Session Analysis**: Duration, events, device performance
4. **Product Performance**: Top products, category analysis, price correlation

### Key Metrics
- **Conversion Rate**: % of sessions resulting in purchase
- **Average Order Value (AOV)**: Revenue per purchase
- **Session Duration**: Time spent per visit
- **Cart Abandonment Rate**: % of carts not converted
- **User Lifetime Value (LTV)**: Total revenue per user

## Performance Optimizations

### Implemented Optimizations

1. **Partitioning**: Data partitioned by `event_date` for time-range queries
2. **Broadcast Joins**: Small dimension tables broadcasted to avoid shuffles
3. **Caching**: Intermediate results cached when reused
4. **Adaptive Query Execution**: Spark AQE enabled for dynamic optimization
5. **Compression**: Snappy compression for Parquet files
6. **Column Pruning**: Only required columns read from storage

### Performance Results

- **Data Volume**: 100K+ events processed
- **Processing Time**: ~2-3 minutes on local machine (4 cores, 8GB RAM)
- **Storage Efficiency**: 10MB raw CSV → 2MB compressed Parquet (80% reduction)
- **Query Performance**: Partitioned queries 10x faster than full scans

## What I Learned

Through building this project, I gained hands-on experience with:

### Technical Skills
- **Apache Spark**: DataFrames, transformations, window functions, optimization
- **Data Warehousing**: Star schema design, dimensional modeling, fact/dimension tables
- **SQL**: Complex analytical queries, CTEs, window functions, aggregations
- **Python**: OOP, data processing with pandas, visualization
- **Data Pipeline**: ETL design patterns, error handling, logging

### Engineering Concepts
- **Lazy Evaluation**: Understanding Spark's execution planning
- **Partitioning Strategies**: How to optimize for query patterns
- **Data Skew**: Handling unbalanced data distributions
- **Idempotency**: Designing rerunnable pipelines
- **Scalability**: Writing code that works at different data volumes

### Business Analytics
- **Conversion Funnels**: Tracking user journey from view to purchase
- **Cohort Analysis**: Understanding user retention and behavior over time
- **Product Analytics**: Identifying top performers and opportunities
- **User Segmentation**: Grouping users for targeted strategies

## Future Improvements

### Technical Enhancements
- [ ] Add Apache Kafka for real-time streaming
- [ ] Implement Apache Airflow for orchestration and scheduling
- [ ] Add data quality checks and monitoring
- [ ] Implement CDC (Change Data Capture) for incremental loads
- [ ] Add unit tests and integration tests
- [ ] Deploy to cloud (AWS/GCP/Azure)

### Analytics Features
- [ ] Machine learning models (churn prediction, recommendation engine)
- [ ] Real-time dashboard with Streamlit or Dash
- [ ] A/B testing framework
- [ ] Anomaly detection for fraud/unusual patterns
- [ ] Customer journey mapping visualization

### Data Engineering
- [ ] Add SCD Type 2 for dimension history tracking
- [ ] Implement data versioning with Delta Lake
- [ ] Add data catalog and documentation (dbt)
- [ ] Set up CI/CD pipeline
- [ ] Add performance benchmarking suite

## Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please feel free to:

1. Open an issue for bugs or feature requests
2. Submit a pull request with improvements
3. Star the repository if you find it helpful

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Jovan Chua**
Email: kaiovanchua@gmail.com
LinkedIn:(https://www.linkedin.com/in/kaijovan/)
GitHub: (https://github.com/Kaiovan/)

---
