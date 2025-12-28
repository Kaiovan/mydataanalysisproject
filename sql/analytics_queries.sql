-- Analytics Queries for E-commerce Clickstream Data Warehouse
-- Demonstrates various SQL techniques and analytical patterns

-- ============================================================================
-- BASIC AGGREGATIONS
-- ============================================================================

-- Query 1: Daily revenue and conversion trends
SELECT
    event_date,
    total_events,
    unique_users,
    unique_sessions,
    total_purchases,
    total_revenue,
    ROUND(total_revenue / NULLIF(total_purchases, 0), 2) as avg_order_value,
    conversion_rate
FROM fact_daily_metrics
ORDER BY event_date DESC
LIMIT 30;

-- Query 2: User segmentation distribution
SELECT
    user_segment,
    COUNT(*) as num_users,
    SUM(total_purchases) as total_purchases,
    SUM(total_revenue) as total_revenue,
    ROUND(AVG(total_revenue), 2) as avg_revenue_per_user,
    ROUND(AVG(num_sessions), 2) as avg_sessions_per_user
FROM dim_users
GROUP BY user_segment
ORDER BY total_revenue DESC;

-- Query 3: Top 20 products by revenue
SELECT
    p.product_id,
    p.product_name,
    p.category,
    pm.total_interactions,
    pm.unique_users,
    pm.num_add_to_cart,
    pm.click_to_cart_rate,
    p.price
FROM dim_product_metrics pm
JOIN dim_products p ON pm.product_id = p.product_id
ORDER BY pm.total_interactions DESC
LIMIT 20;

-- ============================================================================
-- CONVERSION FUNNEL ANALYSIS
-- ============================================================================

-- Query 4: Session-level conversion funnel
SELECT
    event_date,
    total_sessions,
    sessions_with_views,
    sessions_with_clicks,
    sessions_with_cart,
    sessions_with_purchase,
    ROUND(100.0 * sessions_with_clicks / NULLIF(sessions_with_views, 0), 2) as view_to_click_rate,
    ROUND(100.0 * sessions_with_cart / NULLIF(sessions_with_clicks, 0), 2) as click_to_cart_rate,
    ROUND(100.0 * sessions_with_purchase / NULLIF(sessions_with_cart, 0), 2) as cart_to_purchase_rate,
    conversion_rate as overall_conversion_rate
FROM vw_session_funnel
ORDER BY event_date DESC;

-- Query 5: Event-level funnel aggregation
WITH funnel AS (
    SELECT
        COUNT(DISTINCT CASE WHEN event_type = 'page_view' THEN session_id END) as page_views,
        COUNT(DISTINCT CASE WHEN event_type = 'product_click' THEN session_id END) as product_clicks,
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN session_id END) as add_to_cart,
        COUNT(DISTINCT CASE WHEN event_type = 'checkout' THEN session_id END) as checkout,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN session_id END) as purchase
    FROM fact_events
)
SELECT
    page_views,
    product_clicks,
    ROUND(100.0 * product_clicks / NULLIF(page_views, 0), 2) as pv_to_click_rate,
    add_to_cart,
    ROUND(100.0 * add_to_cart / NULLIF(product_clicks, 0), 2) as click_to_cart_rate,
    checkout,
    ROUND(100.0 * checkout / NULLIF(add_to_cart, 0), 2) as cart_to_checkout_rate,
    purchase,
    ROUND(100.0 * purchase / NULLIF(checkout, 0), 2) as checkout_to_purchase_rate
FROM funnel;

-- ============================================================================
-- TIME-BASED ANALYSIS
-- ============================================================================

-- Query 6: Hourly traffic patterns
SELECT
    event_hour,
    COUNT(*) as num_events,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT session_id) as unique_sessions,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases
FROM fact_events
GROUP BY event_hour
ORDER BY event_hour;

-- Query 7: Day of week analysis
SELECT
    CASE event_day_of_week
        WHEN 1 THEN 'Sunday'
        WHEN 2 THEN 'Monday'
        WHEN 3 THEN 'Tuesday'
        WHEN 4 THEN 'Wednesday'
        WHEN 5 THEN 'Thursday'
        WHEN 6 THEN 'Friday'
        WHEN 7 THEN 'Saturday'
    END as day_name,
    COUNT(DISTINCT session_id) as sessions,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases,
    SUM(CASE WHEN event_type = 'purchase' THEN cart_value ELSE 0 END) as revenue
FROM fact_events
GROUP BY event_day_of_week
ORDER BY event_day_of_week;

-- Query 8: Month-over-month growth
WITH monthly_metrics AS (
    SELECT
        event_year,
        event_month,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(DISTINCT session_id) as sessions,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases,
        SUM(CASE WHEN event_type = 'purchase' THEN cart_value ELSE 0 END) as revenue
    FROM fact_events
    GROUP BY event_year, event_month
)
SELECT
    event_year,
    event_month,
    unique_users,
    sessions,
    purchases,
    ROUND(revenue, 2) as revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY event_year, event_month)) /
          NULLIF(LAG(revenue) OVER (ORDER BY event_year, event_month), 0), 2) as revenue_growth_pct
FROM monthly_metrics
ORDER BY event_year, event_month;

-- ============================================================================
-- COHORT ANALYSIS
-- ============================================================================

-- Query 9: User cohort retention (simplified)
SELECT
    DATE_TRUNC('week', first_seen) as cohort_week,
    COUNT(DISTINCT user_id) as cohort_size,
    ROUND(AVG(num_sessions), 2) as avg_sessions,
    ROUND(AVG(total_revenue), 2) as avg_revenue,
    SUM(CASE WHEN total_purchases > 0 THEN 1 ELSE 0 END) as converted_users
FROM dim_users
GROUP BY DATE_TRUNC('week', first_seen)
ORDER BY cohort_week DESC;

-- Query 10: Purchase frequency by cohort
SELECT
    DATE_TRUNC('month', first_seen) as cohort_month,
    COUNT(*) as total_users,
    SUM(CASE WHEN total_purchases = 0 THEN 1 ELSE 0 END) as zero_purchases,
    SUM(CASE WHEN total_purchases = 1 THEN 1 ELSE 0 END) as one_purchase,
    SUM(CASE WHEN total_purchases BETWEEN 2 AND 5 THEN 1 ELSE 0 END) as two_to_five,
    SUM(CASE WHEN total_purchases > 5 THEN 1 ELSE 0 END) as more_than_five
FROM dim_users
GROUP BY DATE_TRUNC('month', first_seen)
ORDER BY cohort_month DESC;

-- ============================================================================
-- PRODUCT ANALYSIS
-- ============================================================================

-- Query 11: Category performance comparison
SELECT
    category,
    COUNT(DISTINCT product_id) as num_products,
    SUM(total_interactions) as total_interactions,
    SUM(unique_users) as total_unique_users,
    SUM(num_add_to_cart) as total_add_to_cart,
    ROUND(AVG(click_to_cart_rate), 4) as avg_click_to_cart_rate,
    ROUND(AVG(avg_price), 2) as avg_product_price
FROM dim_product_metrics
GROUP BY category
ORDER BY total_interactions DESC;

-- Query 12: Product affinity (frequently bought together)
SELECT
    p1.product_name as product_a,
    p2.product_name as product_b,
    sessions_together,
    users_together
FROM vw_product_affinity pa
JOIN dim_products p1 ON pa.product_a = p1.product_id
JOIN dim_products p2 ON pa.product_b = p2.product_id
ORDER BY sessions_together DESC
LIMIT 20;

-- Query 13: Abandoned cart products
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(DISTINCT e.session_id) as times_added_to_cart,
    COUNT(DISTINCT CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM fact_events pe
            WHERE pe.session_id = e.session_id
            AND pe.event_type = 'purchase'
        )
        THEN e.session_id
    END) as times_abandoned,
    ROUND(100.0 * COUNT(DISTINCT CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM fact_events pe
            WHERE pe.session_id = e.session_id
            AND pe.event_type = 'purchase'
        )
        THEN e.session_id
    END) / NULLIF(COUNT(DISTINCT e.session_id), 0), 2) as abandonment_rate
FROM fact_events e
JOIN dim_products p ON e.product_id = p.product_id
WHERE e.event_type = 'add_to_cart'
GROUP BY p.product_id, p.product_name, p.category
HAVING COUNT(DISTINCT e.session_id) >= 10
ORDER BY times_abandoned DESC
LIMIT 20;

-- ============================================================================
-- USER BEHAVIOR ANALYSIS
-- ============================================================================

-- Query 14: Session duration distribution
SELECT
    CASE
        WHEN session_duration_seconds < 60 THEN '<1 min'
        WHEN session_duration_seconds < 300 THEN '1-5 min'
        WHEN session_duration_seconds < 900 THEN '5-15 min'
        WHEN session_duration_seconds < 1800 THEN '15-30 min'
        ELSE '>30 min'
    END as duration_bucket,
    COUNT(*) as num_sessions,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) as converted_sessions,
    ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*), 2) as conversion_rate
FROM fact_sessions
GROUP BY duration_bucket
ORDER BY
    CASE duration_bucket
        WHEN '<1 min' THEN 1
        WHEN '1-5 min' THEN 2
        WHEN '5-15 min' THEN 3
        WHEN '15-30 min' THEN 4
        ELSE 5
    END;

-- Query 15: Device and browser performance
SELECT
    device,
    browser,
    COUNT(DISTINCT session_id) as sessions,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) as conversions,
    ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*), 2) as conversion_rate,
    ROUND(AVG(session_duration_seconds), 2) as avg_session_duration
FROM fact_sessions
GROUP BY device, browser
ORDER BY sessions DESC;

-- Query 16: Referrer source analysis
SELECT
    referrer,
    COUNT(DISTINCT session_id) as sessions,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(CASE WHEN converted THEN 1 ELSE 0 END) as conversions,
    ROUND(100.0 * SUM(CASE WHEN converted THEN 1 ELSE 0 END) / COUNT(*), 2) as conversion_rate,
    SUM(total_cart_value) as total_revenue,
    ROUND(AVG(num_events), 2) as avg_events_per_session
FROM fact_sessions
GROUP BY referrer
ORDER BY sessions DESC;

-- ============================================================================
-- WINDOW FUNCTIONS AND RANKING
-- ============================================================================

-- Query 17: Top products per category
WITH ranked_products AS (
    SELECT
        p.category,
        p.product_id,
        p.product_name,
        pm.total_interactions,
        pm.unique_users,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY pm.total_interactions DESC) as rank
    FROM dim_product_metrics pm
    JOIN dim_products p ON pm.product_id = p.product_id
)
SELECT
    category,
    product_id,
    product_name,
    total_interactions,
    unique_users,
    rank
FROM ranked_products
WHERE rank <= 5
ORDER BY category, rank;

-- Query 18: Running total of daily revenue
SELECT
    event_date,
    total_revenue,
    SUM(total_revenue) OVER (ORDER BY event_date) as cumulative_revenue,
    AVG(total_revenue) OVER (ORDER BY event_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as revenue_7day_ma
FROM fact_daily_metrics
ORDER BY event_date;

-- Query 19: User lifetime value percentiles
SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_revenue) as p25_ltv,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_revenue) as median_ltv,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_revenue) as p75_ltv,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY total_revenue) as p90_ltv,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_revenue) as p95_ltv,
    MAX(total_revenue) as max_ltv
FROM dim_users
WHERE total_revenue > 0;

-- ============================================================================
-- ADVANCED ANALYTICS
-- ============================================================================

-- Query 20: Customer lifetime value by acquisition source
SELECT
    first_sessions.referrer as acquisition_source,
    COUNT(DISTINCT first_sessions.user_id) as acquired_users,
    SUM(u.total_revenue) as total_ltv,
    ROUND(AVG(u.total_revenue), 2) as avg_ltv,
    ROUND(AVG(u.num_sessions), 2) as avg_sessions,
    ROUND(AVG(u.total_purchases), 2) as avg_purchases
FROM (
    SELECT
        user_id,
        referrer,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY session_start) as rn
    FROM fact_sessions
) first_sessions
JOIN dim_users u ON first_sessions.user_id = u.user_id
WHERE first_sessions.rn = 1
GROUP BY first_sessions.referrer
ORDER BY total_ltv DESC;

-- Query 21: Time to first purchase
SELECT
    CASE
        WHEN days_to_first_purchase = 0 THEN 'Same day'
        WHEN days_to_first_purchase <= 1 THEN '1 day'
        WHEN days_to_first_purchase <= 7 THEN '2-7 days'
        WHEN days_to_first_purchase <= 30 THEN '8-30 days'
        ELSE '>30 days'
    END as time_to_purchase,
    COUNT(*) as num_users,
    ROUND(AVG(total_revenue), 2) as avg_revenue
FROM (
    SELECT
        u.user_id,
        u.total_revenue,
        MIN(DATE(e.timestamp)) - DATE(u.first_seen) as days_to_first_purchase
    FROM dim_users u
    JOIN fact_events e ON u.user_id = e.user_id
    WHERE e.event_type = 'purchase'
    GROUP BY u.user_id, u.total_revenue, u.first_seen
) purchase_timing
GROUP BY time_to_purchase
ORDER BY
    CASE time_to_purchase
        WHEN 'Same day' THEN 1
        WHEN '1 day' THEN 2
        WHEN '2-7 days' THEN 3
        WHEN '8-30 days' THEN 4
        ELSE 5
    END;

-- Query 22: Search query effectiveness
SELECT
    search_query,
    COUNT(*) as search_count,
    COUNT(DISTINCT session_id) as sessions_with_search,
    COUNT(DISTINCT CASE
        WHEN EXISTS (
            SELECT 1 FROM fact_events pe
            WHERE pe.session_id = e.session_id
            AND pe.event_type = 'purchase'
            AND pe.timestamp > e.timestamp
        )
        THEN session_id
    END) as sessions_converted_after_search,
    ROUND(100.0 * COUNT(DISTINCT CASE
        WHEN EXISTS (
            SELECT 1 FROM fact_events pe
            WHERE pe.session_id = e.session_id
            AND pe.event_type = 'purchase'
            AND pe.timestamp > e.timestamp
        )
        THEN session_id
    END) / NULLIF(COUNT(DISTINCT session_id), 0), 2) as conversion_rate
FROM fact_events e
WHERE page_type = 'search' AND search_query IS NOT NULL
GROUP BY search_query
HAVING COUNT(DISTINCT session_id) >= 10
ORDER BY search_count DESC
LIMIT 20;
