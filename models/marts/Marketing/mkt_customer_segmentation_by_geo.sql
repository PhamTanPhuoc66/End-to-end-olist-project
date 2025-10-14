SELECT
    doc.customer_state,
    COUNT(DISTINCT doc.customer_key) AS number_of_customers,
    SUM(foi.price) AS total_revenue,
    AVG(foi.price) AS average_spend_per_item
FROM {{source('star_schema','fact_order_items')}} AS foi
JOIN {{source('star_schema','dim_olist_customers')}} AS doc
    ON foi.customer_key = doc.customer_key AND doc.is_current = true
GROUP BY
    doc.customer_state
ORDER BY
    total_revenue DESC;