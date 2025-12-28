# Quick Start Guide

Get up and running with the E-commerce Clickstream Analytics Pipeline in 5 minutes!

## Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed (Python 3.11 recommended)
- **Java 11+** installed (required for Spark)
- At least **8GB RAM** available
- **Git** for cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mydataanalysisproject.git
cd mydataanalysisproject
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PySpark 3.5.0
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- PostgreSQL connector
- And other dependencies

### 4. Verify Installation

```bash
python -c "from pyspark.sql import SparkSession; print('Spark version:', SparkSession.builder.getOrCreate().version)"
```

You should see: `Spark version: 3.5.0`

## Running the Pipeline

### Option 1: Run Everything (Recommended for First Time)

**Windows:**
```bash
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will:
1. Generate 10,000 sessions of clickstream data (~100K events)
2. Process the data with Apache Spark
3. Create analytics visualizations

**Total runtime: ~5-7 minutes**

### Option 2: Run Step-by-Step

If you want to run each step individually:

**Step 1: Generate Data (30 seconds)**
```bash
python src/data_generation/clickstream_generator.py
```

Output: `data/raw/clickstream_events.csv` (~10MB)

**Step 2: Process with Spark (2-3 minutes)**
```bash
python src/spark_processing/etl_pipeline.py
```

Output: `data/processed/` (Parquet files, ~2MB)

**Step 3: Create Analytics (30 seconds)**
```bash
python src/analytics/dashboard.py
```

Output: `data/analytics_output/` (PNG charts + summary report)

## View Results

After running the pipeline, check these locations:

### 1. Processed Data
```
data/processed/
├── fact_events/        # All clickstream events (partitioned by date)
├── fact_sessions/      # Session-level metrics
├── fact_daily_metrics/ # Daily aggregates
├── dim_user_metrics/   # User lifetime metrics
└── dim_product_metrics/ # Product performance
```

### 2. Analytics Visualizations
```
data/analytics_output/
├── daily_trends.png         # Traffic & revenue trends
├── user_segments.png        # User segmentation analysis
├── session_analysis.png     # Session behavior
├── product_performance.png  # Product metrics
└── summary_report.txt       # Key insights summary
```

### 3. Spark UI

While running the Spark pipeline, view the Spark UI:
```
http://localhost:4040
```

## Exploring with Jupyter

Launch Jupyter to interactively explore the data:

```bash
jupyter lab
```

Then open: `notebooks/01_data_exploration.ipynb`

## Common Issues & Solutions

### Issue: "Java not found"
**Solution:** Install Java 11 or higher
- Windows: Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or use [OpenJDK](https://adoptium.net/)
- Mac: `brew install openjdk@11`
- Linux: `sudo apt install openjdk-11-jdk`

### Issue: "Memory Error" during Spark execution
**Solution:** Reduce data size in `src/data_generation/clickstream_generator.py`:
```python
NUM_SESSIONS = 5000  # Instead of 10000
```

### Issue: Port 4040 already in use
**Solution:** Another Spark application is running. Stop it or Spark will use 4041, 4042, etc.

## Next Steps

Now that you have the pipeline running:

1. **Explore the Data**: Open the Jupyter notebook to dive deeper
2. **Run SQL Queries**: Check out `sql/analytics_queries.sql` for 20+ sample queries
3. **Customize**: Modify `config/config.yaml` to adjust pipeline parameters
4. **Scale Up**: Increase `NUM_SESSIONS` to 100K or 1M to test scalability
5. **Add Features**: Implement ideas from the "Future Improvements" section in README

## Using Docker (Alternative)

If you prefer Docker:

```bash
# Build and start all services
docker-compose up -d

# Access Jupyter Lab
open http://localhost:8888

# Run pipeline inside container
docker exec -it clickstream-analytics bash
./run_pipeline.sh

# View database
open http://localhost:5050  # pgAdmin
```

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review `sql/analytics_queries.sql` for SQL examples
- Explore `notebooks/01_data_exploration.ipynb` for analysis examples
- Open an issue on GitHub for bugs or questions

## Quick Reference

| Command | Purpose |
|---------|---------|
| `run_pipeline.bat/.sh` | Run complete pipeline |
| `python src/data_generation/clickstream_generator.py` | Generate data only |
| `python src/spark_processing/etl_pipeline.py` | Run Spark ETL only |
| `python src/analytics/dashboard.py` | Generate visualizations only |
| `jupyter lab` | Launch Jupyter |
| `pytest tests/` | Run unit tests |
| `docker-compose up -d` | Start Docker environment |

---

**You're all set!** Start exploring your e-commerce analytics pipeline. 🚀
