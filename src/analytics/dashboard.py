"""
Analytics Dashboard for Clickstream Data

Reads processed data from Spark output and creates visualizations
Demonstrates data analysis and visualization skills
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ClickstreamAnalytics:
    """Analytics and visualization for clickstream data"""

    def __init__(self, data_path: str):
        """
        Initialize analytics class

        Args:
            data_path: Path to processed data directory
        """
        self.data_path = Path(data_path)
        self.daily_metrics = None
        self.session_metrics = None
        self.user_metrics = None
        self.product_metrics = None

    def load_data(self):
        """Load processed data from Parquet files"""
        print("Loading processed data...")

        try:
            # Load daily metrics
            daily_path = self.data_path / "fact_daily_metrics"
            if daily_path.exists():
                self.daily_metrics = pd.read_parquet(daily_path)
                print(f"Loaded {len(self.daily_metrics)} days of data")

            # Load session metrics
            session_path = self.data_path / "fact_sessions"
            if session_path.exists():
                self.session_metrics = pd.read_parquet(session_path)
                print(f"Loaded {len(self.session_metrics)} sessions")

            # Load user metrics
            user_path = self.data_path / "dim_user_metrics"
            if user_path.exists():
                self.user_metrics = pd.read_parquet(user_path)
                print(f"Loaded {len(self.user_metrics)} users")

            # Load product metrics
            product_path = self.data_path / "dim_product_metrics"
            if product_path.exists():
                self.product_metrics = pd.read_parquet(product_path)
                print(f"Loaded {len(self.product_metrics)} products")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def plot_daily_trends(self, output_path: Optional[str] = None):
        """Plot daily traffic and revenue trends"""
        if self.daily_metrics is None:
            print("Daily metrics not loaded")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Daily Performance Trends', fontsize=16, fontweight='bold')

        df = self.daily_metrics.sort_values('event_date')

        # Plot 1: Daily Users and Sessions
        ax1 = axes[0, 0]
        ax1.plot(df['event_date'], df['unique_users'], marker='o', label='Unique Users', linewidth=2)
        ax1.plot(df['event_date'], df['unique_sessions'], marker='s', label='Sessions', linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count')
        ax1.set_title('Daily Users and Sessions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Daily Revenue
        ax2 = axes[0, 1]
        ax2.bar(df['event_date'], df['total_revenue'], alpha=0.7, color='green')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Revenue ($)')
        ax2.set_title('Daily Revenue')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # Plot 3: Conversion Rate
        ax3 = axes[1, 0]
        ax3.plot(df['event_date'], df['conversion_rate'], marker='o', color='purple', linewidth=2)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Conversion Rate (%)')
        ax3.set_title('Daily Conversion Rate')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # Plot 4: Average Order Value
        ax4 = axes[1, 1]
        ax4.bar(df['event_date'], df['avg_order_value'], alpha=0.7, color='orange')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('AOV ($)')
        ax4.set_title('Average Order Value')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved daily trends plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_user_segments(self, output_path: Optional[str] = None):
        """Plot user segmentation analysis"""
        if self.user_metrics is None:
            print("User metrics not loaded")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('User Segmentation Analysis', fontsize=16, fontweight='bold')

        # Segment distribution
        segment_counts = self.user_metrics['user_segment'].value_counts()

        # Plot 1: Segment Distribution (Pie)
        ax1 = axes[0]
        colors = sns.color_palette('Set2', len(segment_counts))
        ax1.pie(segment_counts.values, labels=segment_counts.index,
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('User Distribution by Segment')

        # Plot 2: Revenue by Segment (Bar)
        ax2 = axes[1]
        segment_revenue = self.user_metrics.groupby('user_segment')['total_revenue'].sum().sort_values(ascending=False)
        ax2.bar(range(len(segment_revenue)), segment_revenue.values, color=colors)
        ax2.set_xticks(range(len(segment_revenue)))
        ax2.set_xticklabels(segment_revenue.index, rotation=45, ha='right')
        ax2.set_ylabel('Total Revenue ($)')
        ax2.set_title('Revenue by User Segment')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved user segments plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_session_analysis(self, output_path: Optional[str] = None):
        """Plot session-level analysis"""
        if self.session_metrics is None:
            print("Session metrics not loaded")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Session Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Session Duration Distribution
        ax1 = axes[0, 0]
        session_duration_minutes = self.session_metrics['session_duration_seconds'] / 60
        ax1.hist(session_duration_minutes[session_duration_minutes < 60], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Session Duration (minutes)')
        ax1.set_ylabel('Number of Sessions')
        ax1.set_title('Session Duration Distribution (< 60 min)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Events per Session
        ax2 = axes[0, 1]
        ax2.hist(self.session_metrics['num_events'][self.session_metrics['num_events'] < 50],
                 bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Number of Events')
        ax2.set_ylabel('Number of Sessions')
        ax2.set_title('Events per Session Distribution')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Device Distribution
        ax3 = axes[1, 0]
        device_counts = self.session_metrics['device'].value_counts()
        ax3.bar(device_counts.index, device_counts.values, alpha=0.7)
        ax3.set_xlabel('Device Type')
        ax3.set_ylabel('Number of Sessions')
        ax3.set_title('Sessions by Device Type')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Conversion Rate by Device
        ax4 = axes[1, 1]
        conversion_by_device = self.session_metrics.groupby('device')['converted'].mean() * 100
        ax4.bar(conversion_by_device.index, conversion_by_device.values, alpha=0.7, color='coral')
        ax4.set_xlabel('Device Type')
        ax4.set_ylabel('Conversion Rate (%)')
        ax4.set_title('Conversion Rate by Device')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved session analysis plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_product_performance(self, output_path: Optional[str] = None):
        """Plot product performance metrics"""
        if self.product_metrics is None:
            print("Product metrics not loaded")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Product Performance', fontsize=16, fontweight='bold')

        # Plot 1: Top 15 Products by Interactions
        ax1 = axes[0, 0]
        top_products = self.product_metrics.nlargest(15, 'total_interactions')
        ax1.barh(range(len(top_products)), top_products['total_interactions'])
        ax1.set_yticks(range(len(top_products)))
        ax1.set_yticklabels(top_products['product_id'], fontsize=8)
        ax1.set_xlabel('Total Interactions')
        ax1.set_title('Top 15 Products by Interactions')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # Plot 2: Category Performance
        ax2 = axes[0, 1]
        category_performance = self.product_metrics.groupby('category')['total_interactions'].sum().sort_values(ascending=False)
        ax2.bar(range(len(category_performance)), category_performance.values)
        ax2.set_xticks(range(len(category_performance)))
        ax2.set_xticklabels(category_performance.index, rotation=45, ha='right')
        ax2.set_ylabel('Total Interactions')
        ax2.set_title('Performance by Category')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Click-to-Cart Rate Distribution
        ax3 = axes[1, 0]
        valid_rates = self.product_metrics['click_to_cart_rate'][self.product_metrics['click_to_cart_rate'] > 0]
        ax3.hist(valid_rates, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Click-to-Cart Rate')
        ax3.set_ylabel('Number of Products')
        ax3.set_title('Click-to-Cart Rate Distribution')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Price vs Interactions Scatter
        ax4 = axes[1, 1]
        sample_products = self.product_metrics.sample(min(1000, len(self.product_metrics)))
        scatter = ax4.scatter(sample_products['avg_price'], sample_products['total_interactions'],
                             alpha=0.5, c=sample_products['unique_users'], cmap='viridis')
        ax4.set_xlabel('Average Price ($)')
        ax4.set_ylabel('Total Interactions')
        ax4.set_title('Price vs. Interactions')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Unique Users')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved product performance plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_summary_report(self, output_path: Optional[str] = None):
        """Generate a text summary report"""
        report = []
        report.append("=" * 80)
        report.append("E-COMMERCE CLICKSTREAM ANALYTICS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")

        if self.daily_metrics is not None:
            report.append("--- OVERALL METRICS ---")
            total_revenue = self.daily_metrics['total_revenue'].sum()
            total_purchases = self.daily_metrics['total_purchases'].sum()
            avg_conversion = self.daily_metrics['conversion_rate'].mean()
            report.append(f"Total Revenue: ${total_revenue:,.2f}")
            report.append(f"Total Purchases: {total_purchases:,}")
            report.append(f"Average Conversion Rate: {avg_conversion:.2f}%")
            report.append("")

        if self.user_metrics is not None:
            report.append("--- USER METRICS ---")
            report.append(f"Total Users: {len(self.user_metrics):,}")
            report.append(f"Average Sessions per User: {self.user_metrics['num_sessions'].mean():.2f}")
            report.append(f"Average Revenue per User: ${self.user_metrics['total_revenue'].mean():.2f}")
            report.append("")
            report.append("User Segments:")
            for segment, count in self.user_metrics['user_segment'].value_counts().items():
                pct = count / len(self.user_metrics) * 100
                report.append(f"  {segment}: {count:,} ({pct:.1f}%)")
            report.append("")

        if self.session_metrics is not None:
            report.append("--- SESSION METRICS ---")
            report.append(f"Total Sessions: {len(self.session_metrics):,}")
            report.append(f"Average Events per Session: {self.session_metrics['num_events'].mean():.2f}")
            report.append(f"Average Session Duration: {self.session_metrics['session_duration_seconds'].mean() / 60:.2f} minutes")
            conversion_rate = self.session_metrics['converted'].mean() * 100
            report.append(f"Overall Conversion Rate: {conversion_rate:.2f}%")
            report.append("")

        if self.product_metrics is not None:
            report.append("--- PRODUCT METRICS ---")
            report.append(f"Total Products: {len(self.product_metrics):,}")
            report.append(f"Average Interactions per Product: {self.product_metrics['total_interactions'].mean():.2f}")
            report.append("")
            report.append("Top 5 Products by Interactions:")
            top_5 = self.product_metrics.nlargest(5, 'total_interactions')
            for idx, row in top_5.iterrows():
                report.append(f"  {row['product_id']}: {row['total_interactions']:,} interactions")
            report.append("")

        report.append("=" * 80)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Saved summary report to {output_path}")
        else:
            print(report_text)

        return report_text

    def run_all_analytics(self, output_dir: Optional[str] = None):
        """Run all analytics and generate visualizations"""
        print("\n" + "=" * 80)
        print("RUNNING CLICKSTREAM ANALYTICS")
        print("=" * 80 + "\n")

        self.load_data()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.plot_daily_trends(str(output_path / "daily_trends.png"))
            self.plot_user_segments(str(output_path / "user_segments.png"))
            self.plot_session_analysis(str(output_path / "session_analysis.png"))
            self.plot_product_performance(str(output_path / "product_performance.png"))
            self.generate_summary_report(str(output_path / "summary_report.txt"))
        else:
            self.plot_daily_trends()
            self.plot_user_segments()
            self.plot_session_analysis()
            self.plot_product_performance()
            self.generate_summary_report()

        print("\n" + "=" * 80)
        print("ANALYTICS COMPLETE")
        print("=" * 80)


def main():
    """Main execution"""
    base_path = Path(__file__).parent.parent.parent
    data_path = base_path / "data" / "processed"
    output_path = base_path / "data" / "analytics_output"

    analytics = ClickstreamAnalytics(str(data_path))
    analytics.run_all_analytics(str(output_path))


if __name__ == "__main__":
    main()
