#!/bin/bash

# E-commerce Clickstream Analytics Pipeline Runner
# This script runs the complete data pipeline from generation to analytics

set -e  # Exit on error

echo "========================================="
echo "E-commerce Clickstream Analytics Pipeline"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create necessary directories
print_step "Creating necessary directories..."
mkdir -p data/raw data/processed data/warehouse data/analytics_output logs
print_success "Directories created"

# Step 1: Generate clickstream data
print_step "Step 1/3: Generating clickstream data..."
python3 src/data_generation/clickstream_generator.py
print_success "Data generation complete"

# Step 2: Run Spark ETL pipeline
print_step "Step 2/3: Running Spark ETL pipeline..."
python3 src/spark_processing/etl_pipeline.py
print_success "ETL pipeline complete"

# Step 3: Run analytics and generate visualizations
print_step "Step 3/3: Running analytics and generating visualizations..."
python3 src/analytics/dashboard.py
print_success "Analytics complete"

echo ""
echo "========================================="
echo "Pipeline execution completed successfully!"
echo "========================================="
echo ""
echo "Generated outputs:"
echo "  - Processed data: data/processed/"
echo "  - Analytics visualizations: data/analytics_output/"
echo ""
echo "To view Spark UI during execution, visit: http://localhost:4040"
echo ""
