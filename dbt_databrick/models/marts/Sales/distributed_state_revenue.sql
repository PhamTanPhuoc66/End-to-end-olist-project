SELECT
    doc.customer_state,
    SUM(foi.price) AS total_revenue,
    COUNT(DISTINCT foi.order_key) AS total_orders,
    COUNT(DISTINCT doc.customer_key) AS total_customers
FROM {{source('star_schema','fact_order_items')}} AS foi
JOIN {{source('star_schema','dim_olist_customers')}} AS doc
    ON foi.customer_key = doc.customer_key AND doc.is_current = true
GROUP BY
    doc.customer_state
ORDER BY
    total_revenue DESC;