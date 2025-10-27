import os
import mlflow
import pandas as pd
import numpy as np
from databricks import sql
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import mlflow
import time
from datetime import datetime 
from functools import wraps
from ultils import AdvancedModelTuner , _temporarily_disable_cuml_accel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold  
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import gc
load_dotenv()
database = "olist_dataset"
goldSche=database+ ".gold"
lookupGeo=database+".silver"+ ".silver_olist_geolocation"
lookupCateName=database+".silver"+ ".silver_product_category_name_translation"
connection = sql.connect(
                        server_hostname = "dbc-214ffec4-c1f2.cloud.databricks.com",
                        http_path = "/sql/1.0/warehouses/8753fa7395d762fe",
                        access_token = os.environ.get("DATABRICKS_TOKEN"))

query = f"""
WITH analysis_snapshot AS (
    SELECT 
        DATEADD(day, 1, MAX(d.full_date)) AS snapshot_date
    FROM {goldSche}.dim_olist_orders o
    JOIN {goldSche}.dim_date d ON o.order_purchase_date_key = d.date_key
),

-- Order-level metrics (granularity: order_key, customer_key)
order_dates AS ( 
    SELECT 
        o.order_key,
        o.order_id,
        oi.customer_key,
        d_purchase.full_date AS purchase_date,
        d_delivered.full_date AS delivered_date,
        -- 1. Số ngày giao trễ
        DATEDIFF(day, d_estimated.full_date, d_delivered.full_date) AS delivery_overdue_days,
        -- 2. Cờ giao trễ
        CASE WHEN d_delivered.full_date > d_estimated.full_date THEN 1 ELSE 0 END AS is_delivered_late,
        -- 3. Thời gian xử lý nội bộ
        DATEDIFF(day, d_approved.full_date, d_carrier.full_date) AS internal_processing_days,
        -- 4. Thời gian hứa hẹn
        DATEDIFF(day, d_purchase.full_date, d_estimated.full_date) AS promised_window_days,
        -- 5. Cờ ngày lễ
        d_purchase.is_holiday AS is_purchase_holiday
    FROM {goldSche}.dim_olist_orders o
    JOIN {goldSche}.fact_order_items oi ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_date d_purchase ON o.order_purchase_date_key = d_purchase.date_key
    LEFT JOIN {goldSche}.dim_date d_delivered ON o.order_delivered_customer_date_key = d_delivered.date_key
    LEFT JOIN {goldSche}.dim_date d_estimated ON o.order_estimated_delivery_date_key = d_estimated.date_key
    LEFT JOIN {goldSche}.dim_date d_approved ON o.order_approved_at_date_key = d_approved.date_key
    LEFT JOIN {goldSche}.dim_date d_carrier ON o.order_delivered_carrier_date_key = d_carrier.date_key
    WHERE o.order_status = 'delivered'
),

-- Order-level monetary values (granularity: order_key, customer_key)
order_values AS (
    SELECT  
        oi.order_key,
        oi.customer_key,
        SUM(oi.price) AS order_price,
        SUM(oi.freight_value) AS order_freight
    FROM {goldSche}.fact_order_items oi
    GROUP BY oi.order_key, oi.customer_key
),

-- Customer-level aggregates (granularity: customer_key, sẽ map lên customer_unique_id)
customer_key_agg AS (
    SELECT 
        od.customer_key,
        MAX(od.purchase_date) AS last_purchase_date,
        MIN(od.purchase_date) AS first_purchase_date,
        COUNT(DISTINCT od.order_id) AS frequency_orders,
        SUM(ov.order_price) AS monetary_value,
        AVG(ov.order_price) AS avg_order_price,
        SUM(ov.order_freight) AS sum_freight,
        AVG(ov.order_freight) AS avg_freight,
        AVG(DATEDIFF(day, od.purchase_date, od.delivered_date)) AS avg_delivery_delay,
        AVG(CAST(od.is_delivered_late AS FLOAT)) AS late_delivery_rate,
        AVG(od.delivery_overdue_days) AS avg_delivery_overdue_days,
        AVG(od.internal_processing_days) AS avg_internal_processing_days,
        AVG(od.promised_window_days) AS avg_promised_window_days,
        AVG(CASE WHEN od.is_purchase_holiday = true THEN 1.0 ELSE 0.0 END) AS holiday_purchase_rate,
        AVG(CASE WHEN od.is_purchase_holiday = true THEN od.delivery_overdue_days END) AS avg_overdue_days_on_holidays,
        AVG(CASE WHEN od.is_purchase_holiday = false THEN od.delivery_overdue_days END) AS avg_overdue_days_on_non_holidays
    FROM order_dates od
    JOIN order_values ov ON od.order_key = ov.order_key
    GROUP BY od.customer_key
),

-- Review aggregates by customer_key (map sau)
review_by_key AS (
    SELECT 
        r.customer_key,
        AVG(r.score) AS avg_review_score
    FROM {goldSche}.fact_reviews r
    GROUP BY r.customer_key
),

-- Payment aggregates by customer_key (map sau)
payment_by_key AS (
    SELECT 
        p.customer_key,
        COUNT(DISTINCT pt.payment_type) AS distinct_payment_methods,
        AVG(p.payment_installments) AS avg_installments
    FROM {goldSche}.fact_payments p
    JOIN {goldSche}.dim_olist_payment_type pt ON p.payment_type_key = pt.payment_type_key
    GROUP BY p.customer_key
),

-- Product diversity & Herfindahl index (tính trực tiếp ở granularity customer_unique_id)
product_diversity_unique AS (
    SELECT 
        customer_unique_id,
        COUNT(DISTINCT product_category_name) AS distinct_categories,
        COUNT(DISTINCT product_key) AS distinct_products,
        COUNT(DISTINCT seller_key) AS distinct_sellers,
        SUM(POWER(category_ratio, 2)) AS category_concentration
    FROM (
        SELECT 
            c.customer_unique_id,
            p.product_category_name,
            oi.product_key,
            oi.seller_key,
            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY c.customer_unique_id) AS category_ratio
        FROM {goldSche}.fact_order_items oi
        JOIN {goldSche}.dim_olist_products p ON oi.product_key = p.product_key
        JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
        GROUP BY c.customer_unique_id, p.product_category_name, oi.product_key, oi.seller_key
    ) t
    GROUP BY customer_unique_id
),


-- Top product per customer_unique_id
top_product_unique AS (
    SELECT customer_unique_id, product_key AS most_purchased_product
    FROM (
        SELECT 
            c.customer_unique_id,
            oi.product_key,
            ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id ORDER BY COUNT(*) DESC) AS rn
        FROM {goldSche}.fact_order_items oi
        JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
        GROUP BY c.customer_unique_id, oi.product_key
    ) t
    WHERE rn = 1
),

-- Top payment type per customer_unique_id
top_payment_unique AS (
    SELECT customer_unique_id, payment_type AS most_used_payment_type
    FROM (
        SELECT 
            c.customer_unique_id,
            pt.payment_type,
            ROW_NUMBER() OVER (PARTITION BY c.customer_unique_id ORDER BY COUNT(*) DESC) AS rn
        FROM {goldSche}.fact_payments p
        JOIN {goldSche}.dim_olist_customers c ON p.customer_key = c.customer_key
        JOIN {goldSche}.dim_olist_payment_type pt ON p.payment_type_key = pt.payment_type_key
        GROUP BY c.customer_unique_id, pt.payment_type
    ) t
    WHERE rn = 1
),

-- Review details (direct at customer_unique_id)
review_detail_unique AS (
    SELECT 
        c.customer_unique_id,
        SUM(CASE 
                WHEN LENGTH(rc.review_comment_title) > 0 OR LENGTH(rc.review_comment_message) > 0
                THEN 1 ELSE 0 
            END) AS review_with_comment,
        AVG(DATEDIFF(day, d2.full_date, d.full_date)) AS avg_review_response_time
    FROM {goldSche}.fact_reviews r
    JOIN {goldSche}.dim_olist_customers c ON r.customer_key = c.customer_key
    JOIN {goldSche}.dim_olist_review_content rc ON r.review_content_key = rc.review_content_key
    JOIN {goldSche}.dim_date d ON r.review_answer_date_key = d.date_key
    JOIN {goldSche}.dim_date d2 ON r.review_creation_date_key = d2.date_key
    GROUP BY c.customer_unique_id
),

-- Map customer_key -> customer_unique_id (for is_current customers)
customer_key_to_unique AS (
    SELECT customer_key, customer_unique_id
    FROM {goldSche}.dim_olist_customers
    WHERE is_current = true
)

-- FINAL: aggregate everything at customer_unique_id
SELECT 
    map.customer_unique_id AS customer_id,
    -- recency & tenure
    DATEDIFF(day, MAX(cka.last_purchase_date), (SELECT snapshot_date FROM analysis_snapshot)) AS recency_days,
    DATEDIFF(day, MIN(cka.first_purchase_date), MAX(cka.last_purchase_date)) AS customer_tenure,
    -- frequency, monetary
    SUM(cka.frequency_orders) AS frequency_orders,
    SUM(cka.monetary_value) AS monetary_value,
    SUM(cka.sum_freight) AS sum_freight,
    -- averages
    AVG(cka.avg_order_price) AS avg_order_price,
    AVG(cka.avg_freight) AS avg_freight,
    -- weighted delivery metrics
    AVG(cka.avg_delivery_delay) AS avg_delivery_delay,
    AVG(cka.late_delivery_rate) AS late_delivery_rate,
    AVG(cka.avg_delivery_overdue_days) AS avg_delivery_overdue_days,
    AVG(cka.avg_internal_processing_days) AS avg_internal_processing_days,
    AVG(cka.avg_promised_window_days) AS avg_promised_window_days,
    AVG(cka.holiday_purchase_rate) AS holiday_purchase_rate,
    AVG(cka.avg_overdue_days_on_holidays) AS avg_overdue_days_on_holidays,
    AVG(cka.avg_overdue_days_on_non_holidays) AS avg_overdue_days_on_non_holidays,

    -- join với các bảng unique-level
    ra.avg_review_score,
    pa.distinct_payment_methods,
    pa.avg_installments,
    pd.distinct_categories,
    pd.distinct_products,
    pd.distinct_sellers,
    pd.category_concentration,
    tp.most_purchased_product,
    tpay.most_used_payment_type,
    rd.review_with_comment,
    rd.avg_review_response_time,

    -- churn label
    CASE WHEN DATEDIFF(day, MAX(cka.last_purchase_date), (SELECT snapshot_date FROM analysis_snapshot)) > 180 THEN 1 ELSE 0 END AS churn_label

FROM customer_key_agg cka
JOIN customer_key_to_unique map ON cka.customer_key = map.customer_key
LEFT JOIN review_by_key ra ON cka.customer_key = ra.customer_key
LEFT JOIN payment_by_key pa ON cka.customer_key = pa.customer_key
LEFT JOIN product_diversity_unique pd ON map.customer_unique_id = pd.customer_unique_id
LEFT JOIN top_product_unique tp ON map.customer_unique_id = tp.customer_unique_id
LEFT JOIN top_payment_unique tpay ON map.customer_unique_id = tpay.customer_unique_id
LEFT JOIN review_detail_unique rd ON map.customer_unique_id = rd.customer_unique_id
GROUP BY map.customer_unique_id,
         ra.avg_review_score, pa.distinct_payment_methods, pa.avg_installments,
         pd.distinct_categories, pd.distinct_products, pd.distinct_sellers, pd.category_concentration,
         tp.most_purchased_product, tpay.most_used_payment_type, rd.review_with_comment, rd.avg_review_response_time;

"""
df = pd.read_sql(query, connection)
connection.close()
df.describe()
df.info()
def prepare_X_y(df):
    """
    Feature engineering and create X and y
    :param df: pandas dataframe
    :return: (X, y) output feature matrix (dataframe), target (series)
    """
    X = df.drop(columns=['customer_id', 'recency_days','churn_label'])
    y = df['churn_label'] 
    return X, y
X, y = prepare_X_y(df)

##########
RANDOM_STATE = 42
TRAIN_SIZE = 0.9
mlflow.set_tracking_uri("http://172.17.0.1:5000")
EXPERIMENT_NAME = "churn_prediction"

#########
X_train, X_test ,y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")


mlflow.set_experiment(EXPERIMENT_NAME)



# ======================
# Định nghĩa hàm & nhóm cột 
# ======================
def to_str(X):
    return X.astype(str)

special_zero_col = [
    "avg_overdue_days_on_holidays", "review_with_comment", "avg_review_response_time"
]
median_cols = [
    "avg_delivery_delay", "avg_delivery_overdue_days", "avg_internal_processing_days",
    "avg_overdue_days_on_non_holidays", "avg_review_score", "distinct_payment_methods",
    "avg_installments", "distinct_categories", "distinct_products", "distinct_sellers",
    "category_concentration"
]
normal_numeric_cols = [
    "customer_tenure", "frequency_orders", "monetary_value", "avg_order_price",
    "sum_freight", "avg_freight", "late_delivery_rate", "avg_promised_window_days",
    "holiday_purchase_rate"
]

# ======================
# Transformers 
# ======================
special_transformer = ImPipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("scaler", StandardScaler())
])
median_transformer = ImPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
normal_transformer = ImPipeline(steps=[("scaler", StandardScaler())])

categorical_string_cols = ["most_used_payment_type"]
categorical_string_transformer = ImPipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self): 
        self.freq_map = {}
    def fit(self, X, y=None):
        col = X.columns[0] if isinstance(X, pd.DataFrame) else 0
        series = X[col] if isinstance(X, pd.DataFrame) else pd.Series(X[:, 0])
        self.freq_map = series.value_counts(normalize=True).to_dict()
        return self
    def transform(self, X):
        col = X.columns[0] if isinstance(X, pd.DataFrame) else 0
        series = X[col] if isinstance(X, pd.DataFrame) else pd.Series(X[:, 0])
        return series.map(self.freq_map).fillna(0).to_frame()

categorical_id_cols = ["most_purchased_product"]
categorical_id_transformer = ImPipeline(steps=[
    ("freq_enc", FrequencyEncoder()),
    ("scaler", StandardScaler())
])

# ======================
# ColumnTransformer (KHÔNG ĐỔI)
# ======================
preprocessor = ColumnTransformer(
    transformers=[
        ("special", special_transformer, special_zero_col),
        ("median", median_transformer, median_cols),
        ("normal", normal_transformer, normal_numeric_cols),
        ("categorical_str", categorical_string_transformer, categorical_string_cols),
        ("categorical_id", categorical_id_transformer, categorical_id_cols)
    ],
    remainder='drop'
)
# ======================
# Chọn sampler an toàn 
# ======================
min_count = int(y_train.value_counts().min())
if min_count <= 1:
    sampler = RandomOverSampler(random_state=42)
else:
    k_neighbors = min(5, max(1, min_count - 1))
    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)

# ======================
# Chạy thử nghiệm với 
# ======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with mlflow.start_run(run_name=f"Model Training (Churn prediction)-{timestamp}") as parent_run:


    # ==============================================================================
    #                          NAIVE BAYES (GAUSSIANNB)
    # ==============================================================================

    # Base model cho Gaussian Naive Bayes
    base_model_nb = GaussianNB()

    # Parameter grid cho Gaussian Naive Bayes
    # Tham số quan trọng nhất là 'var_smoothing', giúp ổn định tính toán.
    param_grid_nb = {
        'var_smoothing': np.logspace(0, -9, num=10)
    }

    # Cấu hình HalvingGridSearchCV cho Naive Bayes
    halving_search_nb = HalvingGridSearchCV(
        estimator=base_model_nb,
        param_grid=param_grid_nb,
        scoring="f1_weighted",
        cv=5,
        factor=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Chạy tuner cho Naive Bayes
    with AdvancedModelTuner(
        search_strategy=halving_search_nb,
        preprocessor=preprocessor,
        sampler=sampler,
        experiment_name=EXPERIMENT_NAME
    ) as tuner:

        print("\n--- Bắt đầu Huấn luyện với HalvingGridSearchCV + Naive Bayes ---")
        tuner.fit(X_train, y_train)

        print("\n--- Bắt đầu Đánh giá ---")
        tuner.evaluate_train_test(X_train, y_train, X_test, y_test)

        print("\n--- Vẽ Confusion Matrix ---")
        tuner.plot_confusion_matrix(X_test, y_test, "Test")



