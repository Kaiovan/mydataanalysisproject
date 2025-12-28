@echo off
REM E-commerce Clickstream Analytics Pipeline Runner (Windows)
REM This script runs the complete data pipeline from generation to analytics

echo =========================================
echo E-commerce Clickstream Analytics Pipeline
echo =========================================
echo.

REM Create necessary directories
echo [STEP] Creating necessary directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\warehouse" mkdir data\warehouse
if not exist "data\analytics_output" mkdir data\analytics_output
if not exist "logs" mkdir logs
echo [SUCCESS] Directories created
echo.

REM Step 1: Generate clickstream data
echo [STEP] Step 1/3: Generating clickstream data...
python src\data_generation\clickstream_generator.py
if errorlevel 1 (
    echo [ERROR] Data generation failed
    exit /b 1
)
echo [SUCCESS] Data generation complete
echo.

REM Step 2: Run Spark ETL pipeline
echo [STEP] Step 2/3: Running Spark ETL pipeline...
python src\spark_processing\etl_pipeline.py
if errorlevel 1 (
    echo [ERROR] ETL pipeline failed
    exit /b 1
)
echo [SUCCESS] ETL pipeline complete
echo.

REM Step 3: Run analytics and generate visualizations
echo [STEP] Step 3/3: Running analytics and generating visualizations...
python src\analytics\dashboard.py
if errorlevel 1 (
    echo [ERROR] Analytics generation failed
    exit /b 1
)
echo [SUCCESS] Analytics complete
echo.

echo =========================================
echo Pipeline execution completed successfully!
echo =========================================
echo.
echo Generated outputs:
echo   - Processed data: data\processed\
echo   - Analytics visualizations: data\analytics_output\
echo.
echo To view Spark UI during execution, visit: http://localhost:4040
echo.

pause
