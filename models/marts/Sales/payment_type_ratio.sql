SELECT
    dopt.payment_type,
    COUNT(fp.order_key) AS number_of_transactions,
    SUM(fp.payment_value) as total_payment_value
FROM {{source('star_schema','fact_payments')}} as fp
JOIN {{source('star_schema','dim_olist_payment_type')}} as dopt
    ON fp.payment_type_key = dopt.payment_type_key
GROUP BY
    dopt.payment_type
ORDER BY
    number_of_transactions DESC;