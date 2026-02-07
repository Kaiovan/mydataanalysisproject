-- E-commerce Clickstream Data Warehouse Schema
-- Star Schema Design with Fact and Dimension Tables

-- ============================================================================
-- DIMENSION TABLES
-- ============================================================================

-- Dimension: Users
CREATE TABLE IF NOT EXISTS dim_users (
    user_id VARCHAR(50) PRIMARY KEY,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    num_sessions INTEGER,
    total_events INTEGER,
    num_active_days INTEGER,
    total_purchases INTEGER,
    total_revenue DECIMAL(10, 2),
    unique_products_viewed INTEGER,
    days_active INTEGER,
    avg_events_per_session DECIMAL(10, 2),
    avg_revenue_per_purchase DECIMAL(10, 2),
    user_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dim_users_segment ON dim_users(user_segment);
CREATE INDEX idx_dim_users_first_seen ON dim_users(first_seen);

-- Dimension: Products
CREATE TABLE IF NOT EXISTS dim_products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    price DECIMAL(10, 2),
    stock_quantity INTEGER,
    rating DECIMAL(3, 2),
    num_reviews INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dim_products_category ON dim_products(category);
CREATE INDEX idx_dim_products_price ON dim_products(price);

-- Dimension: Product Metrics (aggregated metrics for each product)
CREATE TABLE IF NOT EXISTS dim_product_metrics (
    product_id VARCHAR(50) PRIMARY KEY,
    category VARCHAR(100),
    total_interactions INTEGER,
    unique_users INTEGER,
    unique_sessions INTEGER,
    num_clicks INTEGER,
    num_add_to_cart INTEGER,
    avg_price DECIMAL(10, 2),
    click_to_cart_rate DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id)
);

-- Dimension: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key DATE PRIMARY KEY,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    quarter INTEGER,
    day_of_week INTEGER,
    day_name VARCHAR(10),
    month_name VARCHAR(10),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dim_date_year_month ON dim_date(year, month);

-- Dimension: Device
CREATE TABLE IF NOT EXISTS dim_device (
    device_id SERIAL PRIMARY KEY,
    device_type VARCHAR(50),
    browser VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_dim_device_unique ON dim_device(device_type, browser);

-- ============================================================================
-- FACT TABLES
-- ============================================================================

-- Fact: Clickstream Events (Grain: One row per event)
CREATE TABLE IF NOT EXISTS fact_events (
    event_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    event_date DATE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    page_type VARCHAR(50),
    product_id VARCHAR(50),
    category VARCHAR(100),
    price DECIMAL(10, 2),
    cart_size INTEGER,
    cart_value DECIMAL(10, 2),
    num_items INTEGER,
    payment_method VARCHAR(50),
    device VARCHAR(50),
    browser VARCHAR(50),
    ip_address VARCHAR(50),
    referrer VARCHAR(100),
    search_query VARCHAR(255),
    event_hour INTEGER,
    event_day_of_week INTEGER,
    event_month INTEGER,
    event_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (event_date) REFERENCES dim_date(date_key)
);

-- Indexes for performance
CREATE INDEX idx_fact_events_session ON fact_events(session_id);
CREATE INDEX idx_fact_events_user ON fact_events(user_id);
CREATE INDEX idx_fact_events_timestamp ON fact_events(timestamp);
CREATE INDEX idx_fact_events_date ON fact_events(event_date);
CREATE INDEX idx_fact_events_type ON fact_events(event_type);
CREATE INDEX idx_fact_events_product ON fact_events(product_id);

-- Partition by event_date for better query performance
-- Note: Actual partitioning syntax varies by database
-- ALTER TABLE fact_events PARTITION BY RANGE (event_date);

-- Fact: Sessions (Grain: One row per session)
CREATE TABLE IF NOT EXISTS fact_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP NOT NULL,
    session_duration_seconds INTEGER,
    num_events INTEGER,
    num_unique_event_types INTEGER,
    device VARCHAR(50),
    browser VARCHAR(50),
    referrer VARCHAR(100),
    num_page_views INTEGER,
    num_product_clicks INTEGER,
    num_add_to_cart INTEGER,
    num_purchases INTEGER,
    total_cart_value DECIMAL(10, 2),
    items_purchased INTEGER,
    avg_time_between_events DECIMAL(10, 2),
    converted BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
    FOREIGN KEY (event_date) REFERENCES dim_date(date_key)
);

CREATE INDEX idx_fact_sessions_user ON fact_sessions(user_id);
CREATE INDEX idx_fact_sessions_date ON fact_sessions(event_date);
CREATE INDEX idx_fact_sessions_converted ON fact_sessions(converted);
CREATE INDEX idx_fact_sessions_start ON fact_sessions(session_start);

-- Fact: Daily Metrics (Grain: One row per day)
CREATE TABLE IF NOT EXISTS fact_daily_metrics (
    metric_date DATE PRIMARY KEY,
    total_events INTEGER,
    unique_users INTEGER,
    unique_sessions INTEGER,
    total_purchases INTEGER,
    total_revenue DECIMAL(10, 2),
    avg_order_value DECIMAL(10, 2),
    conversion_rate DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (metric_date) REFERENCES dim_date(date_key)
);

-- ============================================================================
-- AGGREGATE TABLES (Materialized Views / Summary Tables)
-- ============================================================================

-- Product performance by day
CREATE TABLE IF NOT EXISTS agg_product_daily (
    product_id VARCHAR(50),
    metric_date DATE,
    num_views INTEGER,
    num_clicks INTEGER,
    num_add_to_cart INTEGER,
    num_purchases INTEGER,
    revenue DECIMAL(10, 2),
    unique_users INTEGER,
    PRIMARY KEY (product_id, metric_date),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (metric_date) REFERENCES dim_date(date_key)
);

-- User activity by week
CREATE TABLE IF NOT EXISTS agg_user_weekly (
    user_id VARCHAR(50),
    week_start_date DATE,
    num_sessions INTEGER,
    num_events INTEGER,
    num_purchases INTEGER,
    total_spent DECIMAL(10, 2),
    PRIMARY KEY (user_id, week_start_date),
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id)
);

-- Category performance
CREATE TABLE IF NOT EXISTS agg_category_metrics (
    category VARCHAR(100) PRIMARY KEY,
    total_products INTEGER,
    total_views INTEGER,
    total_clicks INTEGER,
    total_add_to_cart INTEGER,
    total_purchases INTEGER,
    total_revenue DECIMAL(10, 2),
    avg_price DECIMAL(10, 2),
    conversion_rate DECIMAL(5, 2)
);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: User purchase history
CREATE OR REPLACE VIEW vw_user_purchase_history AS
SELECT
    u.user_id,
    u.user_segment,
    e.timestamp,
    e.session_id,
    e.product_id,
    p.product_name,
    p.category,
    e.cart_value,
    e.num_items,
    e.payment_method
FROM fact_events e
JOIN dim_users u ON e.user_id = u.user_id
LEFT JOIN dim_products p ON e.product_id = p.product_id
WHERE e.event_type = 'purchase';

-- View: Session conversion funnel
CREATE OR REPLACE VIEW vw_session_funnel AS
SELECT
    event_date,
    COUNT(DISTINCT session_id) as total_sessions,
    COUNT(DISTINCT CASE WHEN num_page_views > 0 THEN session_id END) as sessions_with_views,
    COUNT(DISTINCT CASE WHEN num_product_clicks > 0 THEN session_id END) as sessions_with_clicks,
    COUNT(DISTINCT CASE WHEN num_add_to_cart > 0 THEN session_id END) as sessions_with_cart,
    COUNT(DISTINCT CASE WHEN num_purchases > 0 THEN session_id END) as sessions_with_purchase,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN num_purchases > 0 THEN session_id END) /
          NULLIF(COUNT(DISTINCT session_id), 0), 2) as conversion_rate
FROM fact_sessions
GROUP BY event_date;

-- View: Product recommendations based on cart co-occurrence
CREATE OR REPLACE VIEW vw_product_affinity AS
SELECT
    e1.product_id as product_a,
    e2.product_id as product_b,
    COUNT(DISTINCT e1.session_id) as sessions_together,
    COUNT(DISTINCT e1.user_id) as users_together
FROM fact_events e1
JOIN fact_events e2 ON e1.session_id = e2.session_id AND e1.product_id < e2.product_id
WHERE e1.event_type = 'add_to_cart' AND e2.event_type = 'add_to_cart'
GROUP BY e1.product_id, e2.product_id
HAVING COUNT(DISTINCT e1.session_id) >= 5;

-- View: Daily cohort retention
CREATE OR REPLACE VIEW vw_user_retention AS
SELECT
    DATE(u.first_seen) as cohort_date,
    COUNT(DISTINCT u.user_id) as cohort_size,
    COUNT(DISTINCT CASE WHEN DATE(e.timestamp) = DATE(u.first_seen) THEN u.user_id END) as day_0,
    COUNT(DISTINCT CASE WHEN DATE(e.timestamp) = DATE(u.first_seen) + INTERVAL '1 day' THEN u.user_id END) as day_1,
    COUNT(DISTINCT CASE WHEN DATE(e.timestamp) = DATE(u.first_seen) + INTERVAL '7 days' THEN u.user_id END) as day_7,
    COUNT(DISTINCT CASE WHEN DATE(e.timestamp) = DATE(u.first_seen) + INTERVAL '30 days' THEN u.user_id END) as day_30
FROM dim_users u
LEFT JOIN fact_events e ON u.user_id = e.user_id
GROUP BY DATE(u.first_seen);

-- ============================================================================
-- COMMENTS (Documentation)
-- ============================================================================

COMMENT ON TABLE dim_users IS 'Dimension table storing user attributes and lifetime metrics';
COMMENT ON TABLE dim_products IS 'Dimension table storing product catalog information';
COMMENT ON TABLE dim_product_metrics IS 'Aggregated product performance metrics';
COMMENT ON TABLE dim_date IS 'Date dimension for time-based analysis';
COMMENT ON TABLE fact_events IS 'Fact table containing all clickstream events (grain: one event)';
COMMENT ON TABLE fact_sessions IS 'Fact table containing session-level aggregates';
COMMENT ON TABLE fact_daily_metrics IS 'Fact table containing daily aggregated metrics';

-- ============================================================================
-- TABLEAU-OPTIMIZED VIEWS FOR CUSTOMER JOURNEY ANALYTICS
-- Designed for Singtel-style customer experience and touchpoint analysis
-- ============================================================================

-- View 1: Customer Journey Funnel with Touchpoint Performance
-- Use Case: Visualize conversion funnel, identify drop-off points, analyze touchpoint effectiveness
CREATE OR REPLACE VIEW vw_tableau_customer_journey_funnel AS
WITH funnel_steps AS (
    SELECT
        event_date,
        device,
        browser,
        referrer,
        -- Touchpoint counts
        COUNT(DISTINCT CASE WHEN event_type = 'page_view' THEN session_id END) as step_1_page_view,
        COUNT(DISTINCT CASE WHEN event_type = 'product_click' THEN session_id END) as step_2_product_click,
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN session_id END) as step_3_add_to_cart,
        COUNT(DISTINCT CASE WHEN event_type = 'checkout' THEN session_id END) as step_4_checkout,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN session_id END) as step_5_purchase,

        -- Unique users at each step
        COUNT(DISTINCT CASE WHEN event_type = 'page_view' THEN user_id END) as users_step_1,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as users_step_5,

        -- Revenue metrics
        SUM(CASE WHEN event_type = 'purchase' THEN cart_value ELSE 0 END) as total_revenue,
        COUNT(DISTINCT session_id) as total_sessions
    FROM fact_events
    GROUP BY event_date, device, browser, referrer
)
SELECT
    event_date,
    device,
    browser,
    referrer,
    total_sessions,
    -- Funnel steps
    step_1_page_view,
    step_2_product_click,
    step_3_add_to_cart,
    step_4_checkout,
    step_5_purchase,
    -- Conversion rates between steps
    ROUND(100.0 * step_2_product_click / NULLIF(step_1_page_view, 0), 2) as pv_to_click_rate,
    ROUND(100.0 * step_3_add_to_cart / NULLIF(step_2_product_click, 0), 2) as click_to_cart_rate,
    ROUND(100.0 * step_4_checkout / NULLIF(step_3_add_to_cart, 0), 2) as cart_to_checkout_rate,
    ROUND(100.0 * step_5_purchase / NULLIF(step_4_checkout, 0), 2) as checkout_to_purchase_rate,
    -- Overall conversion
    ROUND(100.0 * step_5_purchase / NULLIF(total_sessions, 0), 2) as overall_conversion_rate,
    -- Drop-off analysis
    step_1_page_view - step_2_product_click as dropoff_step_1_to_2,
    step_2_product_click - step_3_add_to_cart as dropoff_step_2_to_3,
    step_3_add_to_cart - step_4_checkout as dropoff_step_3_to_4,
    step_4_checkout - step_5_purchase as dropoff_step_4_to_5,
    -- Revenue metrics
    total_revenue,
    ROUND(total_revenue / NULLIF(step_5_purchase, 0), 2) as avg_order_value,
    users_step_1,
    users_step_5
FROM funnel_steps;

-- View 2: Session Analytics with Customer Touchpoint Details
-- Use Case: Analyze session quality, device/browser performance, engagement metrics
CREATE OR REPLACE VIEW vw_tableau_session_analytics AS
SELECT
    s.session_id,
    s.user_id,
    u.user_segment,
    s.event_date,
    d.day_name,
    d.is_weekend,
    s.session_start,
    s.session_end,
    s.session_duration_seconds,
    ROUND(s.session_duration_seconds / 60.0, 2) as session_duration_minutes,
    s.device,
    s.browser,
    s.referrer,
    -- Engagement metrics
    s.num_events,
    s.num_unique_event_types,
    s.avg_time_between_events,
    CASE
        WHEN s.num_events = 1 THEN 'Bounce'
        WHEN s.num_events BETWEEN 2 AND 5 THEN 'Low Engagement'
        WHEN s.num_events BETWEEN 6 AND 15 THEN 'Medium Engagement'
        ELSE 'High Engagement'
    END as engagement_level,
    -- Conversion indicators
    s.num_page_views,
    s.num_product_clicks,
    s.num_add_to_cart,
    s.num_purchases,
    s.converted,
    CASE
        WHEN s.num_add_to_cart > 0 AND s.num_purchases = 0 THEN TRUE
        ELSE FALSE
    END as cart_abandoned,
    -- Financial metrics
    s.total_cart_value,
    s.items_purchased,
    ROUND(s.total_cart_value / NULLIF(s.items_purchased, 0), 2) as avg_item_price,
    -- Session quality score (0-100)
    LEAST(100,
        (s.num_events * 5) +
        (s.num_unique_event_types * 10) +
        (CASE WHEN s.converted THEN 50 ELSE 0 END)
    ) as session_quality_score,
    -- Time-based attributes for filtering
    EXTRACT(HOUR FROM s.session_start) as session_hour,
    EXTRACT(DOW FROM s.session_start) as session_day_of_week
FROM fact_sessions s
JOIN dim_users u ON s.user_id = u.user_id
JOIN dim_date d ON s.event_date = d.date_key;

-- View 3: User Segment Performance Analysis
-- Use Case: Compare customer segments, identify high-value users, track segment trends
CREATE OR REPLACE VIEW vw_tableau_user_segment_performance AS
SELECT
    u.user_segment,
    COUNT(DISTINCT u.user_id) as total_users,
    -- Activity metrics
    SUM(u.num_sessions) as total_sessions,
    SUM(u.total_events) as total_events,
    ROUND(AVG(u.avg_events_per_session), 2) as avg_events_per_session,
    ROUND(AVG(u.num_active_days), 2) as avg_active_days,
    -- Purchase behavior
    SUM(u.total_purchases) as total_purchases,
    SUM(u.total_revenue) as total_revenue,
    ROUND(AVG(u.total_revenue), 2) as avg_revenue_per_user,
    ROUND(AVG(u.avg_revenue_per_purchase), 2) as avg_order_value,
    -- Conversion metrics
    ROUND(100.0 * SUM(CASE WHEN u.total_purchases > 0 THEN 1 ELSE 0 END) /
          NULLIF(COUNT(DISTINCT u.user_id), 0), 2) as user_conversion_rate,
    -- Product engagement
    ROUND(AVG(u.unique_products_viewed), 2) as avg_products_viewed,
    -- Segment share
    ROUND(100.0 * COUNT(DISTINCT u.user_id) /
          SUM(COUNT(DISTINCT u.user_id)) OVER (), 2) as segment_share_pct,
    ROUND(100.0 * SUM(u.total_revenue) /
          NULLIF(SUM(SUM(u.total_revenue)) OVER (), 0), 2) as revenue_share_pct
FROM dim_users u
GROUP BY u.user_segment;

-- View 4: Cohort Retention Analysis
-- Use Case: Track user retention, identify successful cohorts, measure engagement over time
CREATE OR REPLACE VIEW vw_tableau_cohort_retention AS
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE(first_seen) as cohort_date,
        DATE_TRUNC('month', first_seen) as cohort_month
    FROM dim_users
),
user_activity AS (
    SELECT DISTINCT
        e.user_id,
        DATE(e.timestamp) as activity_date,
        DATE_TRUNC('month', e.timestamp) as activity_month
    FROM fact_events e
)
SELECT
    c.cohort_month,
    COUNT(DISTINCT c.user_id) as cohort_size,
    a.activity_month,
    EXTRACT(MONTH FROM AGE(a.activity_month, c.cohort_month)) as months_since_signup,
    COUNT(DISTINCT a.user_id) as active_users,
    ROUND(100.0 * COUNT(DISTINCT a.user_id) /
          NULLIF(COUNT(DISTINCT c.user_id), 0), 2) as retention_rate
FROM user_cohorts c
LEFT JOIN user_activity a ON c.user_id = a.user_id
WHERE a.activity_month IS NOT NULL
GROUP BY c.cohort_month, a.activity_month;

-- View 5: Touchpoint Performance by Channel
-- Use Case: Evaluate effectiveness of different customer touchpoints (device, browser, referrer)
CREATE OR REPLACE VIEW vw_tableau_touchpoint_performance AS
SELECT
    e.event_date,
    e.device as touchpoint_device,
    e.browser as touchpoint_browser,
    e.referrer as touchpoint_referrer,
    -- Volume metrics
    COUNT(*) as total_interactions,
    COUNT(DISTINCT e.session_id) as unique_sessions,
    COUNT(DISTINCT e.user_id) as unique_users,
    -- Event breakdown
    COUNT(CASE WHEN e.event_type = 'page_view' THEN 1 END) as page_views,
    COUNT(CASE WHEN e.event_type = 'product_click' THEN 1 END) as product_clicks,
    COUNT(CASE WHEN e.event_type = 'add_to_cart' THEN 1 END) as add_to_carts,
    COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchases,
    -- Conversion metrics
    ROUND(100.0 * COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) /
          NULLIF(COUNT(DISTINCT e.session_id), 0), 2) as session_conversion_rate,
    -- Revenue metrics
    SUM(CASE WHEN e.event_type = 'purchase' THEN e.cart_value ELSE 0 END) as total_revenue,
    ROUND(AVG(CASE WHEN e.event_type = 'purchase' THEN e.cart_value END), 2) as avg_order_value,
    -- Engagement quality
    ROUND(AVG(CASE WHEN e.event_type NOT IN ('page_view') THEN 1.0 ELSE 0.0 END) * 100, 2) as engagement_rate
FROM fact_events e
GROUP BY e.event_date, e.device, e.browser, e.referrer;

-- View 6: Executive KPI Dashboard
-- Use Case: High-level metrics for leadership, daily/weekly trends, business health indicators
CREATE OR REPLACE VIEW vw_tableau_executive_kpis AS
WITH daily_stats AS (
    SELECT
        dm.metric_date,
        d.day_name,
        d.is_weekend,
        dm.total_events,
        dm.unique_users,
        dm.unique_sessions,
        dm.total_purchases,
        dm.total_revenue,
        dm.avg_order_value,
        dm.conversion_rate,
        -- Week-over-week comparisons
        LAG(dm.unique_users, 7) OVER (ORDER BY dm.metric_date) as users_prev_week,
        LAG(dm.total_revenue, 7) OVER (ORDER BY dm.metric_date) as revenue_prev_week,
        LAG(dm.conversion_rate, 7) OVER (ORDER BY dm.metric_date) as conversion_prev_week
    FROM fact_daily_metrics dm
    JOIN dim_date d ON dm.metric_date = d.date_key
)
SELECT
    metric_date,
    day_name,
    is_weekend,
    -- Core metrics
    total_events,
    unique_users,
    unique_sessions,
    total_purchases,
    total_revenue,
    avg_order_value,
    conversion_rate,
    -- Derived KPIs
    ROUND(total_revenue / NULLIF(unique_users, 0), 2) as revenue_per_user,
    ROUND(total_events::NUMERIC / NULLIF(unique_sessions, 0), 2) as events_per_session,
    ROUND(total_purchases::NUMERIC / NULLIF(unique_sessions, 0) * 100, 2) as purchase_rate,
    -- Week-over-week change
    ROUND(100.0 * (unique_users - users_prev_week) / NULLIF(users_prev_week, 0), 2) as users_wow_change_pct,
    ROUND(100.0 * (total_revenue - revenue_prev_week) / NULLIF(revenue_prev_week, 0), 2) as revenue_wow_change_pct,
    conversion_rate - conversion_prev_week as conversion_wow_change_pp,
    -- 7-day moving averages
    ROUND(AVG(total_revenue) OVER (ORDER BY metric_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) as revenue_7day_ma,
    ROUND(AVG(unique_users) OVER (ORDER BY metric_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) as users_7day_ma
FROM daily_stats;

-- View 7: Product Analytics with Category Performance
-- Use Case: Product portfolio analysis, category trends, pricing insights
CREATE OR REPLACE VIEW vw_tableau_product_analytics AS
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.price,
    p.rating,
    pm.total_interactions,
    pm.unique_users,
    pm.unique_sessions,
    pm.num_clicks,
    pm.num_add_to_cart,
    pm.click_to_cart_rate,
    -- Calculate purchases from events
    (SELECT COUNT(*) FROM fact_events e
     WHERE e.product_id = p.product_id AND e.event_type = 'purchase') as num_purchases,
    -- Revenue
    (SELECT SUM(cart_value) FROM fact_events e
     WHERE e.product_id = p.product_id AND e.event_type = 'purchase') as total_revenue,
    -- Conversion metrics
    ROUND(100.0 * pm.num_clicks / NULLIF(pm.total_interactions, 0), 2) as click_through_rate,
    ROUND(100.0 * pm.num_add_to_cart / NULLIF(pm.num_clicks, 0), 2) as cart_conversion_rate,
    -- Engagement score
    ROUND((pm.total_interactions * 0.3) + (pm.unique_users * 0.4) + (pm.num_clicks * 0.3), 2) as engagement_score,
    -- Price tier
    CASE
        WHEN p.price < 50 THEN 'Budget'
        WHEN p.price BETWEEN 50 AND 150 THEN 'Mid-Range'
        WHEN p.price BETWEEN 150 AND 500 THEN 'Premium'
        ELSE 'Luxury'
    END as price_tier,
    -- Performance tier
    CASE
        WHEN pm.total_interactions > (SELECT AVG(total_interactions) * 1.5 FROM dim_product_metrics) THEN 'Top Performer'
        WHEN pm.total_interactions > (SELECT AVG(total_interactions) FROM dim_product_metrics) THEN 'Above Average'
        WHEN pm.total_interactions > (SELECT AVG(total_interactions) * 0.5 FROM dim_product_metrics) THEN 'Average'
        ELSE 'Underperforming'
    END as performance_tier
FROM dim_products p
JOIN dim_product_metrics pm ON p.product_id = pm.product_id;

-- View 8: Cart Abandonment Analysis
-- Use Case: Identify cart abandonment patterns, optimize checkout flow, reduce drop-offs
CREATE OR REPLACE VIEW vw_tableau_cart_abandonment AS
WITH cart_sessions AS (
    SELECT
        s.session_id,
        s.user_id,
        u.user_segment,
        s.event_date,
        s.device,
        s.browser,
        s.session_duration_seconds,
        s.num_add_to_cart,
        s.num_purchases,
        s.total_cart_value,
        CASE
            WHEN s.num_add_to_cart > 0 AND s.num_purchases = 0 THEN TRUE
            ELSE FALSE
        END as is_abandoned,
        -- Time to abandonment
        (SELECT MAX(timestamp) FROM fact_events e
         WHERE e.session_id = s.session_id AND e.event_type = 'add_to_cart') as last_cart_event
    FROM fact_sessions s
    JOIN dim_users u ON s.user_id = u.user_id
    WHERE s.num_add_to_cart > 0
)
SELECT
    event_date,
    device,
    browser,
    user_segment,
    COUNT(*) as total_cart_sessions,
    SUM(CASE WHEN is_abandoned THEN 1 ELSE 0 END) as abandoned_sessions,
    SUM(CASE WHEN NOT is_abandoned THEN 1 ELSE 0 END) as completed_sessions,
    ROUND(100.0 * SUM(CASE WHEN is_abandoned THEN 1 ELSE 0 END) /
          NULLIF(COUNT(*), 0), 2) as abandonment_rate,
    -- Lost revenue
    SUM(CASE WHEN is_abandoned THEN total_cart_value ELSE 0 END) as potential_lost_revenue,
    ROUND(AVG(CASE WHEN is_abandoned THEN total_cart_value END), 2) as avg_abandoned_cart_value,
    ROUND(AVG(CASE WHEN is_abandoned THEN session_duration_seconds END), 2) as avg_abandoned_session_duration,
    ROUND(AVG(CASE WHEN is_abandoned THEN num_add_to_cart END), 2) as avg_items_in_abandoned_cart
FROM cart_sessions
GROUP BY event_date, device, browser, user_segment;

-- ============================================================================
-- ML PREDICTION TABLES
-- ============================================================================

-- ML Predictions: Session Conversion Probability
CREATE TABLE IF NOT EXISTS ml_conversion_predictions (
    session_id VARCHAR(50) PRIMARY KEY,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversion_probability DECIMAL(5, 4),
    predicted_converted BOOLEAN,
    conversion_risk VARCHAR(20),
    actual_converted BOOLEAN,
    FOREIGN KEY (session_id) REFERENCES fact_sessions(session_id)
);

CREATE INDEX idx_ml_conversion_risk ON ml_conversion_predictions(conversion_risk);

-- ML Predictions: User Churn Probability
CREATE TABLE IF NOT EXISTS ml_churn_predictions (
    user_id VARCHAR(50) PRIMARY KEY,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    churn_probability DECIMAL(5, 4),
    predicted_churned BOOLEAN,
    churn_risk VARCHAR(20),
    retention_priority_score DECIMAL(10, 2),
    actual_churned BOOLEAN,
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id)
);

CREATE INDEX idx_ml_churn_risk ON ml_churn_predictions(churn_risk);
CREATE INDEX idx_ml_churn_priority ON ml_churn_predictions(retention_priority_score DESC);

-- ML Predictions: Customer Lifetime Value
CREATE TABLE IF NOT EXISTS ml_ltv_predictions (
    user_id VARCHAR(50) PRIMARY KEY,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_ltv_90days DECIMAL(10, 2),
    ltv_category VARCHAR(20),
    actual_revenue DECIMAL(10, 2),
    FOREIGN KEY (user_id) REFERENCES dim_users(user_id)
);

CREATE INDEX idx_ml_ltv_category ON ml_ltv_predictions(ltv_category);
CREATE INDEX idx_ml_ltv_value ON ml_ltv_predictions(predicted_ltv_90days DESC);

-- View: ML Performance Summary
CREATE OR REPLACE VIEW vw_ml_performance_summary AS
SELECT
    'Conversion' as model_name,
    COUNT(*) as total_predictions,
    AVG(conversion_probability) as avg_probability,
    SUM(CASE WHEN predicted_converted = actual_converted THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy
FROM ml_conversion_predictions
UNION ALL
SELECT
    'Churn' as model_name,
    COUNT(*) as total_predictions,
    AVG(churn_probability) as avg_probability,
    SUM(CASE WHEN predicted_churned = actual_churned THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy
FROM ml_churn_predictions
UNION ALL
SELECT
    'LTV' as model_name,
    COUNT(*) as total_predictions,
    AVG(predicted_ltv_90days) as avg_predicted_value,
    NULL as accuracy
FROM ml_ltv_predictions;

COMMENT ON TABLE ml_conversion_predictions IS 'ML predictions for session purchase conversion probability';
COMMENT ON TABLE ml_churn_predictions IS 'ML predictions for customer churn risk and retention priority';
COMMENT ON TABLE ml_ltv_predictions IS 'ML predictions for 90-day customer lifetime value';
