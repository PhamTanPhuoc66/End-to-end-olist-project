SELECT
    ds.seller_state,
    AVG(
        DATEDIFF(
            HOUR,
            -- Sử dụng TRY_TO_TIMESTAMP và DATE_FORMAT
            TRY_TO_TIMESTAMP(CONCAT(approved_date.full_date, ' ', DATE_FORMAT(approved_time.full_time, 'HH:mm:ss')), 'yyyy-MM-dd HH:mm:ss'),
            -- Sử dụng TRY_TO_TIMESTAMP và DATE_FORMAT
            TRY_TO_TIMESTAMP(CONCAT(carrier_date.full_date, ' ', DATE_FORMAT(carrier_time.full_time, 'HH:mm:ss')), 'yyyy-MM-dd HH:mm:ss')
        )
    ) AS avg_processing_hours
FROM {{ source('star_schema', 'dim_olist_orders') }} AS doo
JOIN {{ source('star_schema', 'fact_order_items') }} AS foi
    ON doo.order_key = foi.order_key
JOIN {{ source('star_schema', 'dim_olist_sellers') }} AS ds
    ON foi.seller_key = ds.seller_key
-- JOIN để lấy thời gian đơn hàng được duyệt
JOIN {{ source('star_schema', 'dim_date') }} AS approved_date
    ON doo.order_approved_at_date_key = approved_date.date_key
JOIN {{ source('star_schema', 'dim_time') }} AS approved_time
    ON doo.order_approved_at_time_key = approved_time.time_key
-- JOIN để lấy thời gian bàn giao cho nhà vận chuyển
JOIN {{ source('star_schema', 'dim_date') }} AS carrier_date
    ON doo.order_delivered_carrier_date_key = carrier_date.date_key
JOIN {{ source('star_schema', 'dim_time') }} AS carrier_time
    ON doo.order_delivered_carrier_time_key = carrier_time.time_key
WHERE
    doo.order_approved_at_date_key IS NOT NULL
    AND doo.order_delivered_carrier_date_key IS NOT NULL
GROUP BY
    ds.seller_state
ORDER BY
    avg_processing_hours ASC;