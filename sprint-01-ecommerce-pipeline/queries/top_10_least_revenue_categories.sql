SELECT
    t.product_category_name_english AS Category,
    COUNT(DISTINCT o.order_id) AS Num_order,
    SUM(pay.payment_value) AS Revenue
FROM olist_orders o
JOIN olist_order_items i ON o.order_id = i.order_id
JOIN olist_products p ON i.product_id = p.product_id
JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
JOIN olist_order_payments pay ON o.order_id = pay.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND t.product_category_name_english IS NOT NULL
GROUP BY t.product_category_name_english
ORDER BY Revenue ASC
LIMIT 10
