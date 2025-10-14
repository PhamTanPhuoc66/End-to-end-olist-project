SELECT
    doc.customer_state,
    SUM(foi.price) AS total_product_value,
    SUM(foi.freight_value) AS total_freight_value,
    (SUM(foi.freight_value) / SUM(foi.price)) * 100 AS freight_to_price_ratio_percent
FROM {{source('star_schema','fact_order_items')}} AS foi
JOIN {{source('star_schema','dim_olist_customers')}} AS doc
    ON foi.customer_key = doc.customer_key AND doc.is_current=true
GROUP BY
    doc.customer_state
ORDER BY
    freight_to_price_ratio_percent DESC;