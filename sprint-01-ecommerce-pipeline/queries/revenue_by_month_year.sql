WITH min_payments AS (
    SELECT order_id, MIN(payment_value) AS payment_value
    FROM olist_order_payments
    GROUP BY order_id
),
order_revenue AS (
    SELECT
        STRFTIME('%m', o.order_delivered_customer_date) AS month_no,
        CASE STRFTIME('%m', o.order_delivered_customer_date)
            WHEN '01' THEN 'Jan'
            WHEN '02' THEN 'Feb'
            WHEN '03' THEN 'Mar'
            WHEN '04' THEN 'Apr'
            WHEN '05' THEN 'May'
            WHEN '06' THEN 'Jun'
            WHEN '07' THEN 'Jul'
            WHEN '08' THEN 'Aug'
            WHEN '09' THEN 'Sep'
            WHEN '10' THEN 'Oct'
            WHEN '11' THEN 'Nov'
            WHEN '12' THEN 'Dec'
        END AS month,
        STRFTIME('%Y', o.order_delivered_customer_date) AS year,
        p.payment_value
    FROM olist_orders o
    JOIN min_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
)
SELECT
    month_no,
    month,
    COALESCE(SUM(CASE WHEN year = '2016' THEN payment_value END), 0.00) AS Year2016,
    COALESCE(SUM(CASE WHEN year = '2017' THEN payment_value END), 0.00) AS Year2017,
    COALESCE(SUM(CASE WHEN year = '2018' THEN payment_value END), 0.00) AS Year2018
FROM order_revenue
GROUP BY month_no, month
ORDER BY month_no ASC
