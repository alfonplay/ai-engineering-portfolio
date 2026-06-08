WITH base AS (
    SELECT DISTINCT
        order_id,
        STRFTIME('%m', order_purchase_timestamp) AS month_no,
        CASE STRFTIME('%m', order_purchase_timestamp)
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
        STRFTIME('%Y', order_purchase_timestamp) AS year,
        julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp) AS real_time,
        julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp) AS estimated_time
    FROM olist_orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
)
SELECT
    month_no,
    month,
    AVG(CASE WHEN year = '2016' THEN real_time END) AS Year2016_real_time,
    AVG(CASE WHEN year = '2017' THEN real_time END) AS Year2017_real_time,
    AVG(CASE WHEN year = '2018' THEN real_time END) AS Year2018_real_time,
    AVG(CASE WHEN year = '2016' THEN estimated_time END) AS Year2016_estimated_time,
    AVG(CASE WHEN year = '2017' THEN estimated_time END) AS Year2017_estimated_time,
    AVG(CASE WHEN year = '2018' THEN estimated_time END) AS Year2018_estimated_time
FROM base
GROUP BY month_no, month
ORDER BY month_no ASC
