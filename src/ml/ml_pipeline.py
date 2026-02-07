"""
Machine Learning Pipeline for E-commerce Clickstream Analytics

Implements three predictive models:
1. Purchase Conversion Prediction (session-level classification)
2. Customer Churn Prediction (user-level classification)
3. Customer Lifetime Value Prediction (user-level regression)

Uses scikit-learn with RandomForest and GradientBoosting algorithms.
Reads processed Parquet data from the ETL pipeline output.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def prepare_conversion_features(
    session_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for purchase conversion prediction.

    Takes session-level data and extracts behavioral and categorical features
    that indicate whether a session will result in a purchase.

    Args:
        session_data: DataFrame with session-level metrics from ETL pipeline

    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (0 = no purchase, 1 = purchase)
        feature_names: List of feature column names
    """
    df = session_data.copy()

    # Numerical features
    df["events_per_minute"] = df["num_events"] / (
        df["session_duration_seconds"].clip(lower=1) / 60
    )
    df["cart_to_click_ratio"] = df["num_add_to_cart"] / df[
        "num_product_clicks"
    ].clip(lower=1)

    numerical_features = [
        "num_events",
        "num_page_views",
        "num_product_clicks",
        "num_add_to_cart",
        "session_duration_seconds",
        "avg_time_between_events",
        "num_unique_event_types",
        "events_per_minute",
        "cart_to_click_ratio",
    ]

    # Encode categorical features
    categorical_features = ["device", "browser", "referrer"]
    encoded_dfs = []

    for col_name in categorical_features:
        if col_name in df.columns:
            dummies = pd.get_dummies(df[col_name], prefix=col_name, dtype=int)
            encoded_dfs.append(dummies)

    # Combine all features
    feature_df = df[numerical_features].copy()
    for enc_df in encoded_dfs:
        feature_df = pd.concat([feature_df, enc_df], axis=1)

    # Fill any NaN values
    feature_df = feature_df.fillna(0)

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    y = df["converted"].values.astype(int)

    print(f"  Conversion features: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"  Class distribution: {np.sum(y == 1)} converted, {np.sum(y == 0)} not converted")

    return X, y, feature_names


def prepare_churn_features(
    user_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for customer churn prediction.

    Derives a churn label based on user recency (last_seen > 7 days ago
    relative to the max date in the dataset), then extracts engagement
    and purchase behavior features.

    Args:
        user_data: DataFrame with user-level metrics from ETL pipeline

    Returns:
        X: Feature matrix
        y: Target vector (0 = active, 1 = churned)
        feature_names: List of feature column names
    """
    df = user_data.copy()

    # Create churn target: user hasn't been seen in last 7 days of the dataset
    max_date = pd.to_datetime(df["last_seen"]).max()
    df["last_seen_dt"] = pd.to_datetime(df["last_seen"])
    df["days_since_last_seen"] = (max_date - df["last_seen_dt"]).dt.days
    df["is_churned"] = (df["days_since_last_seen"] > 7).astype(int)

    # Engagement features
    df["purchase_frequency"] = df["total_purchases"] / df["num_sessions"].clip(lower=1)
    df["engagement_score"] = (
        df["num_sessions"] * df["avg_events_per_session"]
    ) / df["days_active"].clip(lower=1)

    # Encode user_segment
    segment_dummies = pd.get_dummies(df["user_segment"], prefix="segment", dtype=int)

    numerical_features = [
        "num_sessions",
        "total_events",
        "num_active_days",
        "avg_events_per_session",
        "days_since_last_seen",
        "total_purchases",
        "total_revenue",
        "avg_revenue_per_purchase",
        "unique_products_viewed",
        "days_active",
        "purchase_frequency",
        "engagement_score",
    ]

    feature_df = pd.concat([df[numerical_features], segment_dummies], axis=1)
    feature_df = feature_df.fillna(0)

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    y = df["is_churned"].values.astype(int)

    print(f"  Churn features: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"  Class distribution: {np.sum(y == 1)} churned, {np.sum(y == 0)} active")

    return X, y, feature_names


def prepare_ltv_features(
    user_data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for Customer Lifetime Value (CLV) prediction.

    Uses user engagement and purchase history to predict total revenue.

    Args:
        user_data: DataFrame with user-level metrics from ETL pipeline

    Returns:
        X: Feature matrix
        y: Target vector (total_revenue)
        feature_names: List of feature column names
    """
    df = user_data.copy()

    # Derived features
    df["revenue_per_session"] = df["total_revenue"] / df["num_sessions"].clip(lower=1)
    df["purchase_rate"] = df["total_purchases"] / df["num_sessions"].clip(lower=1)
    df["first_seen_dt"] = pd.to_datetime(df["first_seen"])
    max_date = pd.to_datetime(df["last_seen"]).max()
    df["days_since_first_seen"] = (max_date - df["first_seen_dt"]).dt.days
    df["product_diversity"] = df["unique_products_viewed"] / df["total_events"].clip(
        lower=1
    )

    # Encode user_segment
    segment_dummies = pd.get_dummies(df["user_segment"], prefix="segment", dtype=int)

    numerical_features = [
        "num_sessions",
        "total_events",
        "num_active_days",
        "total_purchases",
        "unique_products_viewed",
        "days_active",
        "avg_events_per_session",
        "revenue_per_session",
        "purchase_rate",
        "days_since_first_seen",
        "product_diversity",
    ]

    feature_df = pd.concat([df[numerical_features], segment_dummies], axis=1)
    feature_df = feature_df.fillna(0)

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(np.float64)
    y = df["total_revenue"].values.astype(np.float64)

    print(f"  LTV features: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"  Revenue range: ${y.min():.2f} - ${y.max():.2f} (mean: ${y.mean():.2f})")

    return X, y, feature_names


# =============================================================================
# MODEL TRAINING
# =============================================================================


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Train a RandomForest classifier with cross-validation and evaluation.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Feature column names
        model_name: Name for logging
        n_estimators: Number of trees
        max_depth: Max tree depth
        test_size: Test split ratio
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with model, predictions, metrics, and feature importance
    """
    print(f"\n  Training {model_name}...")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc")
    print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Full dataset predictions (for saving)
    full_predictions = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return {
        "model": model,
        "predictions": full_predictions,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def train_regressor(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Train a GradientBoosting regressor with cross-validation and evaluation.

    Args:
        X: Feature matrix
        y: Target values (revenue)
        feature_names: Feature column names
        model_name: Name for logging
        n_estimators: Number of boosting iterations
        max_depth: Max tree depth
        learning_rate: Shrinkage rate
        test_size: Test split ratio
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with model, predictions, metrics, and feature importance
    """
    print(f"\n  Training {model_name}...")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        loss="huber",
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    print(f"  Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Predictions
    y_pred = model.predict(X_test)
    full_predictions = model.predict(X)

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE (avoid division by zero)
    non_zero_mask = y_test != 0
    if non_zero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        )
    else:
        mape = 0.0

    print(f"  RMSE:  ${rmse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "feature_importance": importance,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    return {
        "model": model,
        "predictions": full_predictions,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_classifier_results(
    results: Dict,
    model_name: str,
    feature_names: List[str],
    output_dir: Path,
):
    """
    Generate visualization plots for a classification model.

    Creates:
    - Confusion matrix heatmap
    - ROC curve
    - Top 15 feature importance bar chart

    Args:
        results: Dictionary from train_classifier
        model_name: Name for titles and filenames
        feature_names: Feature column names
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} - Model Evaluation", fontsize=14, fontweight="bold")

    # Confusion matrix
    ax1 = axes[0]
    cm = np.array(results["metrics"]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")

    # ROC curve
    ax2 = axes[1]
    fpr, tpr, _ = roc_curve(results["y_test"], results["y_pred_proba"])
    ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {results["metrics"]["roc_auc"]:.4f}')
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Feature importance
    ax3 = axes[2]
    importance = pd.Series(results["metrics"]["feature_importance"]).nlargest(15)
    ax3.barh(range(len(importance)), importance.values)
    ax3.set_yticks(range(len(importance)))
    ax3.set_yticklabels(importance.index, fontsize=8)
    ax3.set_xlabel("Importance")
    ax3.set_title("Top 15 Features")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(output_dir / f"{filename}_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}_evaluation.png")


def plot_regressor_results(
    results: Dict,
    model_name: str,
    feature_names: List[str],
    output_dir: Path,
):
    """
    Generate visualization plots for a regression model.

    Creates:
    - Actual vs Predicted scatter plot
    - Residual distribution histogram
    - Top 15 feature importance bar chart

    Args:
        results: Dictionary from train_regressor
        model_name: Name for titles and filenames
        feature_names: Feature column names
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} - Model Evaluation", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(results["y_test"], results["y_pred"], alpha=0.5, s=10)
    max_val = max(results["y_test"].max(), results["y_pred"].max())
    ax1.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    ax1.set_xlabel("Actual Revenue ($)")
    ax1.set_ylabel("Predicted Revenue ($)")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residual distribution
    ax2 = axes[1]
    residuals = results["y_test"] - results["y_pred"]
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    # Feature importance
    ax3 = axes[2]
    importance = pd.Series(results["metrics"]["feature_importance"]).nlargest(15)
    ax3.barh(range(len(importance)), importance.values, color="green")
    ax3.set_yticks(range(len(importance)))
    ax3.set_yticklabels(importance.index, fontsize=8)
    ax3.set_xlabel("Importance")
    ax3.set_title("Top 15 Features")
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(output_dir / f"{filename}_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}_evaluation.png")


# =============================================================================
# ML PIPELINE ORCHESTRATOR
# =============================================================================


class MLPipeline:
    """
    End-to-end ML pipeline for clickstream analytics.

    Orchestrates feature engineering, model training, prediction generation,
    and result visualization for all three models.
    """

    def __init__(self, data_path: str, output_path: str):
        """
        Initialize ML Pipeline.

        Args:
            data_path: Path to processed Parquet data (ETL output)
            output_path: Path to save ML outputs (models, predictions, metrics)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)

        # Create output subdirectories
        for subdir in ["models", "features", "predictions", "metrics", "visualizations"]:
            (self.output_path / subdir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed Parquet data from ETL pipeline output."""
        print("\nLoading processed data...")

        session_data = pd.read_parquet(self.data_path / "fact_sessions")
        user_data = pd.read_parquet(self.data_path / "dim_user_metrics")

        print(f"  Loaded {len(session_data):,} sessions")
        print(f"  Loaded {len(user_data):,} users")

        return session_data, user_data

    def run_conversion_prediction(self, session_data: pd.DataFrame) -> Dict:
        """
        Train and evaluate purchase conversion prediction model.

        Predicts whether a browsing session will result in a purchase,
        enabling real-time targeting and personalized promotions.

        Args:
            session_data: Session-level DataFrame from ETL

        Returns:
            Dictionary with model, predictions, and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 1: PURCHASE CONVERSION PREDICTION")
        print("=" * 80)

        # Feature engineering
        X, y, feature_names = prepare_conversion_features(session_data)

        # Save features
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df["target"] = y
        feature_df["session_id"] = session_data["session_id"].values
        feature_df.to_parquet(self.output_path / "features" / "conversion_features.parquet")

        # Train model
        results = train_classifier(X, y, feature_names, "Purchase Conversion")

        # Save model
        joblib.dump(results["model"], self.output_path / "models" / "conversion_model.joblib")

        # Save predictions
        predictions_df = pd.DataFrame({
            "session_id": session_data["session_id"].values,
            "actual_converted": y,
            "conversion_probability": results["predictions"],
            "predicted_converted": (results["predictions"] >= 0.5).astype(int),
        })
        predictions_df["conversion_risk"] = pd.cut(
            predictions_df["conversion_probability"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        predictions_df.to_parquet(
            self.output_path / "predictions" / "conversion_predictions.parquet"
        )

        # Save metrics
        self._save_metrics(results["metrics"], "conversion_metrics.json")

        # Plot results
        plot_classifier_results(
            results, "Purchase Conversion", feature_names,
            self.output_path / "visualizations",
        )

        return results

    def run_churn_prediction(self, user_data: pd.DataFrame) -> Dict:
        """
        Train and evaluate customer churn prediction model.

        Identifies users at risk of churning so retention campaigns
        can be targeted to the right customers.

        Args:
            user_data: User-level DataFrame from ETL

        Returns:
            Dictionary with model, predictions, and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 2: CUSTOMER CHURN PREDICTION")
        print("=" * 80)

        # Feature engineering
        X, y, feature_names = prepare_churn_features(user_data)

        # Save features
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df["target"] = y
        feature_df["user_id"] = user_data["user_id"].values
        feature_df.to_parquet(self.output_path / "features" / "churn_features.parquet")

        # Train model
        results = train_classifier(X, y, feature_names, "Customer Churn")

        # Save model
        joblib.dump(results["model"], self.output_path / "models" / "churn_model.joblib")

        # Save predictions with retention priority
        predictions_df = pd.DataFrame({
            "user_id": user_data["user_id"].values,
            "actual_churned": y,
            "churn_probability": results["predictions"],
            "predicted_churned": (results["predictions"] >= 0.5).astype(int),
        })
        predictions_df["churn_risk"] = pd.cut(
            predictions_df["churn_probability"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
        # Retention priority = churn probability * revenue (high-value at-risk users)
        predictions_df = predictions_df.merge(
            user_data[["user_id", "total_revenue"]], on="user_id"
        )
        predictions_df["retention_priority"] = (
            predictions_df["churn_probability"] * predictions_df["total_revenue"]
        )
        predictions_df.to_parquet(
            self.output_path / "predictions" / "churn_predictions.parquet"
        )

        # Save metrics
        self._save_metrics(results["metrics"], "churn_metrics.json")

        # Plot results
        plot_classifier_results(
            results, "Customer Churn", feature_names,
            self.output_path / "visualizations",
        )

        return results

    def run_ltv_prediction(self, user_data: pd.DataFrame) -> Dict:
        """
        Train and evaluate customer lifetime value prediction model.

        Forecasts expected customer revenue to optimize acquisition spend,
        personalization strategies, and VIP identification.

        Args:
            user_data: User-level DataFrame from ETL

        Returns:
            Dictionary with model, predictions, and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 3: CUSTOMER LIFETIME VALUE (CLV) PREDICTION")
        print("=" * 80)

        # Feature engineering
        X, y, feature_names = prepare_ltv_features(user_data)

        # Save features
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df["target"] = y
        feature_df["user_id"] = user_data["user_id"].values
        feature_df.to_parquet(self.output_path / "features" / "ltv_features.parquet")

        # Train model
        results = train_regressor(X, y, feature_names, "Customer LTV")

        # Save model
        joblib.dump(results["model"], self.output_path / "models" / "ltv_model.joblib")

        # Save predictions
        predictions_df = pd.DataFrame({
            "user_id": user_data["user_id"].values,
            "actual_revenue": y,
            "predicted_ltv_90days": results["predictions"],
        })
        # Categorize LTV
        ltv_bins = [0, 50, 200, 500, np.inf]
        ltv_labels = ["Low", "Medium", "High", "VIP"]
        predictions_df["ltv_category"] = pd.cut(
            predictions_df["predicted_ltv_90days"].clip(lower=0),
            bins=ltv_bins,
            labels=ltv_labels,
            include_lowest=True,
        )
        predictions_df.to_parquet(
            self.output_path / "predictions" / "ltv_predictions.parquet"
        )

        # Save metrics
        self._save_metrics(results["metrics"], "ltv_metrics.json")

        # Plot results
        plot_regressor_results(
            results, "Customer LTV", feature_names,
            self.output_path / "visualizations",
        )

        return results

    def _save_metrics(self, metrics: Dict, filename: str):
        """Save metrics dictionary to JSON file."""
        metrics_path = self.output_path / "metrics" / filename
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved metrics to {metrics_path}")

    def run_full_pipeline(self):
        """
        Execute the complete ML pipeline.

        Loads data, trains all three models, generates predictions,
        saves artifacts, and prints a summary report.
        """
        print("\n" + "=" * 80)
        print("STARTING ML PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Load data
        session_data, user_data = self.load_data()

        # Run all models
        conversion_results = self.run_conversion_prediction(session_data)
        churn_results = self.run_churn_prediction(user_data)
        ltv_results = self.run_ltv_prediction(user_data)

        # Summary
        print("\n" + "=" * 80)
        print("ML PIPELINE SUMMARY")
        print("=" * 80)

        print("\n--- Purchase Conversion Model ---")
        print(f"  Accuracy: {conversion_results['metrics']['accuracy']:.4f}")
        print(f"  ROC-AUC:  {conversion_results['metrics']['roc_auc']:.4f}")
        print(f"  F1-Score: {conversion_results['metrics']['f1']:.4f}")

        print("\n--- Customer Churn Model ---")
        print(f"  Accuracy: {churn_results['metrics']['accuracy']:.4f}")
        print(f"  ROC-AUC:  {churn_results['metrics']['roc_auc']:.4f}")
        print(f"  F1-Score: {churn_results['metrics']['f1']:.4f}")

        print("\n--- Customer LTV Model ---")
        print(f"  RMSE:     ${ltv_results['metrics']['rmse']:.2f}")
        print(f"  MAE:      ${ltv_results['metrics']['mae']:.2f}")
        print(f"  R² Score: {ltv_results['metrics']['r2']:.4f}")

        print("\n" + "=" * 80)
        print("ML PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Outputs saved to: {self.output_path}")
        print("=" * 80)


def main():
    """Main execution function."""
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "processed"
    output_path = base_path / "data" / "ml_output"

    pipeline = MLPipeline(str(data_path), str(output_path))
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
