SELECT
    dd.year,
    dd.month,
    SUM(foi.price) AS revenue
FROM {{source('star_schema','fact_order_items')}} AS foi
JOIN {{source('star_schema','dim_olist_orders')}} AS do
    ON foi.order_key = do.order_key
JOIN {{source('star_schema','dim_date')}} AS dd
    ON do.order_purchase_date_key = dd.date_key
GROUP BY
    dd.year,
    dd.month
ORDER BY
    dd.year,
    dd.month;