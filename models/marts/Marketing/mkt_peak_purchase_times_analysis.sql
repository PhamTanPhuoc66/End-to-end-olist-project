SELECT
    CASE dd.day_of_week
        WHEN 1 THEN 'Thứ Hai'
        WHEN 2 THEN 'Thứ Ba'
        WHEN 3 THEN 'Thứ Tư'
        WHEN 4 THEN 'Thứ Năm'
        WHEN 5 THEN 'Thứ Sáu'
        WHEN 6 THEN 'Thứ Bảy'
        WHEN 7 THEN 'Chủ Nhật'
        ELSE 'Không xác định'
    END AS day_of_week_name,
    dt.time_of_day_bucket,
    COUNT(DISTINCT doo.order_key) AS total_orders
FROM {{source('star_schema','dim_olist_orders')}} AS doo
JOIN {{source('star_schema','dim_date')}} AS dd
    ON doo.order_purchase_date_key = dd.date_key
JOIN {{source('star_schema','dim_time')}} AS dt
    ON doo.order_purchase_time_key = dt.time_key
GROUP BY
    1, 2 
ORDER BY
    total_orders DESC;