WITH ranked_sellers AS (
    SELECT
        dos.seller_id,
        dos.seller_city,
        dos.seller_state,
        dd.year,
        SUM(foi.price) AS total_revenue,
        RANK() OVER (PARTITION BY dd.year ORDER BY SUM(foi.price) DESC) AS rank_num
    FROM {{source('star_schema','fact_order_items')}} AS foi
    JOIN {{source('star_schema','dim_olist_sellers')}} AS dos
        ON foi.seller_key = dos.seller_key
    JOIN {{source('star_schema','dim_olist_orders')}} AS doo
        ON foi.order_key = doo.order_key
    JOIN {{source('star_schema','dim_date')}} AS dd
        ON doo.order_purchase_date_key = dd.date_key 
    GROUP BY
        dos.seller_id,
        dos.seller_city,
        dos.seller_state,
        dd.year
)
SELECT
    seller_id,
    seller_city,
    seller_state,
    year,
    total_revenue,
    rank_num
FROM ranked_sellers
WHERE rank_num <= 10
ORDER BY
    year,
    rank_num;