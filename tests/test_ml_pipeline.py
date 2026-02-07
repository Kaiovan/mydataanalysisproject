"""
Unit tests for ML pipeline module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml.ml_pipeline import (
    prepare_conversion_features,
    prepare_churn_features,
    prepare_ltv_features,
    train_classifier,
    train_regressor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_session_data():
    """Create sample session data matching ETL output schema"""
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        "session_id": [f"session_{i}" for i in range(n)],
        "user_id": [f"user_{i % 50}" for i in range(n)],
        "num_events": np.random.randint(1, 50, n),
        "num_page_views": np.random.randint(0, 30, n),
        "num_product_clicks": np.random.randint(0, 20, n),
        "num_add_to_cart": np.random.randint(0, 10, n),
        "num_purchases": np.random.choice([0, 0, 0, 0, 1], n),
        "session_duration_seconds": np.random.randint(10, 3600, n),
        "avg_time_between_events": np.random.uniform(1, 120, n),
        "num_unique_event_types": np.random.randint(1, 6, n),
        "device": np.random.choice(["desktop", "mobile", "tablet"], n),
        "browser": np.random.choice(["Chrome", "Firefox", "Safari", "Edge"], n),
        "referrer": np.random.choice(["google.com", "direct", "facebook.com"], n),
        "converted": np.random.choice([0, 0, 0, 0, 1], n),
    })


@pytest.fixture
def sample_user_data():
    """Create sample user data matching ETL output schema"""
    np.random.seed(42)
    n = 100
    base_date = pd.Timestamp("2024-01-01")

    return pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(n)],
        "num_sessions": np.random.randint(1, 20, n),
        "total_events": np.random.randint(5, 200, n),
        "num_active_days": np.random.randint(1, 30, n),
        "first_seen": [base_date + pd.Timedelta(days=int(d)) for d in np.random.randint(0, 10, n)],
        "last_seen": [base_date + pd.Timedelta(days=int(d)) for d in np.random.randint(15, 30, n)],
        "total_purchases": np.random.choice([0, 0, 1, 2, 5], n),
        "total_revenue": np.random.uniform(0, 2000, n),
        "unique_products_viewed": np.random.randint(0, 50, n),
        "days_active": np.random.randint(1, 30, n),
        "avg_events_per_session": np.random.uniform(1, 20, n),
        "avg_revenue_per_purchase": np.random.uniform(0, 500, n),
        "user_segment": np.random.choice(
            ["High Value", "Converted", "Engaged", "New/Inactive"], n
        ),
    })


# =============================================================================
# Feature Engineering Tests
# =============================================================================


class TestConversionFeatures:
    """Tests for conversion feature preparation"""

    def test_output_shapes(self, sample_session_data):
        X, y, feature_names = prepare_conversion_features(sample_session_data)
        assert X.shape[0] == len(sample_session_data)
        assert len(feature_names) == X.shape[1]
        assert len(y) == len(sample_session_data)

    def test_no_nan_values(self, sample_session_data):
        X, y, _ = prepare_conversion_features(sample_session_data)
        assert not np.isnan(X).any(), "Features contain NaN values"
        assert not np.isnan(y).any(), "Target contains NaN values"

    def test_target_is_binary(self, sample_session_data):
        _, y, _ = prepare_conversion_features(sample_session_data)
        assert set(np.unique(y)).issubset({0, 1})

    def test_has_derived_features(self, sample_session_data):
        _, _, feature_names = prepare_conversion_features(sample_session_data)
        assert "events_per_minute" in feature_names
        assert "cart_to_click_ratio" in feature_names

    def test_has_encoded_categoricals(self, sample_session_data):
        _, _, feature_names = prepare_conversion_features(sample_session_data)
        device_features = [f for f in feature_names if f.startswith("device_")]
        assert len(device_features) > 0, "Device features not encoded"


class TestChurnFeatures:
    """Tests for churn feature preparation"""

    def test_output_shapes(self, sample_user_data):
        X, y, feature_names = prepare_churn_features(sample_user_data)
        assert X.shape[0] == len(sample_user_data)
        assert len(feature_names) == X.shape[1]
        assert len(y) == len(sample_user_data)

    def test_no_nan_values(self, sample_user_data):
        X, y, _ = prepare_churn_features(sample_user_data)
        assert not np.isnan(X).any(), "Features contain NaN values"
        assert not np.isnan(y).any(), "Target contains NaN values"

    def test_target_is_binary(self, sample_user_data):
        _, y, _ = prepare_churn_features(sample_user_data)
        assert set(np.unique(y)).issubset({0, 1})

    def test_has_derived_features(self, sample_user_data):
        _, _, feature_names = prepare_churn_features(sample_user_data)
        assert "days_since_last_seen" in feature_names
        assert "purchase_frequency" in feature_names
        assert "engagement_score" in feature_names


class TestLTVFeatures:
    """Tests for LTV feature preparation"""

    def test_output_shapes(self, sample_user_data):
        X, y, feature_names = prepare_ltv_features(sample_user_data)
        assert X.shape[0] == len(sample_user_data)
        assert len(feature_names) == X.shape[1]
        assert len(y) == len(sample_user_data)

    def test_no_nan_values(self, sample_user_data):
        X, y, _ = prepare_ltv_features(sample_user_data)
        assert not np.isnan(X).any(), "Features contain NaN values"

    def test_target_is_continuous(self, sample_user_data):
        _, y, _ = prepare_ltv_features(sample_user_data)
        assert y.dtype in [np.float64, np.float32]


# =============================================================================
# Model Training Tests
# =============================================================================


class TestClassifierTraining:
    """Tests for classifier training"""

    def test_train_returns_valid_results(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200)
        feature_names = [f"feature_{i}" for i in range(10)]

        results = train_classifier(X, y, feature_names, "Test Model")

        assert "model" in results
        assert "predictions" in results
        assert "metrics" in results
        assert len(results["predictions"]) == len(y)

    def test_metrics_are_valid(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200)
        feature_names = [f"feature_{i}" for i in range(10)]

        results = train_classifier(X, y, feature_names, "Test Model")
        metrics = results["metrics"]

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert "feature_importance" in metrics
        assert len(metrics["feature_importance"]) == 10

    def test_predictions_are_probabilities(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200)
        feature_names = [f"feature_{i}" for i in range(10)]

        results = train_classifier(X, y, feature_names, "Test Model")

        assert np.all(results["predictions"] >= 0)
        assert np.all(results["predictions"] <= 1)


class TestRegressorTraining:
    """Tests for regressor training"""

    def test_train_returns_valid_results(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200) * 100 + 500
        feature_names = [f"feature_{i}" for i in range(10)]

        results = train_regressor(X, y, feature_names, "Test LTV Model")

        assert "model" in results
        assert "predictions" in results
        assert "metrics" in results
        assert len(results["predictions"]) == len(y)

    def test_metrics_are_valid(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200) * 100 + 500
        feature_names = [f"feature_{i}" for i in range(10)]

        results = train_regressor(X, y, feature_names, "Test LTV Model")
        metrics = results["metrics"]

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert "r2" in metrics
        assert "feature_importance" in metrics
        assert len(metrics["feature_importance"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
