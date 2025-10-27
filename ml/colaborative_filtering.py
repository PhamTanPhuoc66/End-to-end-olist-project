import os
import mlflow
import pandas as pd
import numpy as np
from databricks import sql
from dotenv import load_dotenv
import xgboost
import sklearn
import imblearn
import IPython
import scipy
import joblib
import cloudpickle
import surprise 
conda_env_dict = {
    'name': 'mlflow-env-with-gcc', # Tên môi trường
    'channels': ['conda-forge', 'defaults'], # Kênh
    'dependencies': [
        'python=3.12', # Phiên bản Python

        # ----- Thêm GCC -----
        'gcc_linux-64', # Gói GCC cho Linux x64 từ conda-forge

        # ----- Các gói chính -----
        # Sử dụng f-string để lấy phiên bản từ các thư viện đã import
        'mlflow',
        f'cloudpickle={cloudpickle.__version__}',
        f'joblib={joblib.__version__}',
        f'pandas={pd.__version__}',
        f'numpy={np.__version__}',
        f'scikit-learn={sklearn.__version__}',
        f'imbalanced-learn={imblearn.__version__}',
        'ipython',
        f'scikit-surprise={surprise.__version__}',
        f'scipy={scipy.__version__}',
        'threadpoolctl', # Có thể không cần ghi tường minh

        # ----- Luôn bao gồm pip -----
        'pip'
    ],
    # ----- Phần pip (thường trống nếu các gói trên có trên conda) -----
    # 'pip': [
    #    'some-package-only-on-pip'
    # ]
}
SURPRISE_MODEL_TAGS = {"build_env": "needs_gcc"}

load_dotenv()
database = "olist_dataset"
goldSche=database+ ".gold"
lookupGeo=database+".silver"+ ".silver_olist_geolocation"
lookupCateName=database+".silver"+ ".silver_product_category_name_translation"
connection = sql.connect(
                        server_hostname = "dbc-214ffec4-c1f2.cloud.databricks.com",
                        http_path = "/sql/1.0/warehouses/8753fa7395d762fe",
                        access_token = os.environ.get("DATABRICKS_TOKEN"))


query=f"""
WITH 
-- 1. Lấy dữ liệu tương tác cốt lõi giữa User và Item
user_item_interactions AS (
    SELECT
        c.customer_unique_id,
        oi.product_key,
        COUNT(oi.order_id) AS purchase_count, -- Số lần user mua sản phẩm này
        SUM(oi.price) AS total_spent_on_product, -- Tổng tiền user chi cho sản phẩm này
        AVG(oi.price) AS avg_price_for_product,
        SUM(oi.freight_value) AS total_freight_on_product,
        MIN(d.full_date) AS first_purchase_date,
        MAX(d.full_date) AS last_purchase_date
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    JOIN {goldSche}.dim_olist_orders o ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_date d ON o.order_purchase_date_key = d.date_key
    GROUP BY c.customer_unique_id, oi.product_key
),

-- 2. Lấy điểm review (explicit feedback) cho từng cặp (user, item)
-- Giả định rằng review của một đơn hàng áp dụng cho tất cả sản phẩm trong đơn hàng đó.
product_reviews AS (
    SELECT 
        c.customer_unique_id,
        oi.product_key,
        AVG(r.score) AS avg_review_score_by_user -- Điểm review trung bình user dành cho sản phẩm
    FROM {goldSche}.fact_reviews r
    JOIN {goldSche}.fact_order_items oi ON r.order_key = oi.order_key
    JOIN {goldSche}.dim_olist_customers c ON r.customer_key = c.customer_key
    GROUP BY c.customer_unique_id, oi.product_key
),

-- 3. Xây dựng các feature tổng hợp cho mỗi User
user_features AS (
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(oi.price) AS total_monetary,
        AVG(r.score) AS avg_overall_review_score, -- Điểm review trung bình của user trên mọi sản phẩm
        COUNT(DISTINCT p.product_category_name) AS distinct_categories_purchased,
        c.customer_city,
        c.customer_state
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    JOIN {goldSche}.dim_olist_orders o ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_olist_products p ON oi.product_key = p.product_key
    LEFT JOIN {goldSche}.fact_reviews r ON o.order_key = r.order_key
    GROUP BY c.customer_unique_id, c.customer_city, c.customer_state
),

-- 4. Xây dựng các feature tổng hợp cho mỗi Product (Item)
product_features AS (
    SELECT
        oi.product_key,
        AVG(oi.price) AS avg_product_price, -- Giá bán trung bình của sản phẩm
        COUNT(DISTINCT c.customer_unique_id) AS num_unique_purchasers, -- Số lượng người mua khác nhau
        COUNT(oi.order_id) AS total_sales_count, -- Tổng số lần được bán
        AVG(r.score) AS avg_product_review_score -- Điểm review trung bình của sản phẩm trên mọi user
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    LEFT JOIN {goldSche}.fact_reviews r ON oi.order_key = r.order_key
    GROUP BY oi.product_key
)

-- 5. Kết hợp tất cả lại để tạo ra bảng phân tích cuối cùng
SELECT
    -- Interaction Keys
    uii.customer_unique_id,
    uii.product_key,
    p.product_category_name,

    -- Interaction Features (Implicit Feedback)
    uii.purchase_count,
    uii.total_spent_on_product,
    uii.avg_price_for_product,
    uii.total_freight_on_product,
    uii.first_purchase_date,
    uii.last_purchase_date,
    
    -- Explicit Feedback
    pr.avg_review_score_by_user,

    -- User Features
    uf.total_orders AS user_total_orders,
    uf.total_monetary AS user_total_monetary,
    uf.avg_overall_review_score AS user_avg_overall_review_score,
    uf.distinct_categories_purchased AS user_distinct_categories_purchased,
    uf.customer_city,
    uf.customer_state,
    
    -- Product (Item) Features
    pf.avg_product_price,
    pf.num_unique_purchasers AS product_num_unique_purchasers,
    pf.total_sales_count AS product_total_sales_count,
    pf.avg_product_review_score AS product_avg_review_score,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm

FROM user_item_interactions uii
LEFT JOIN product_reviews pr ON uii.customer_unique_id = pr.customer_unique_id AND uii.product_key = pr.product_key
LEFT JOIN user_features uf ON uii.customer_unique_id = uf.customer_unique_id
LEFT JOIN product_features pf ON uii.product_key = pf.product_key
JOIN {goldSche}.dim_olist_products p ON uii.product_key = p.product_key
ORDER BY uii.customer_unique_id, uii.purchase_count DESC;
"""
df = pd.read_sql(query, connection)
df.describe()
df.info()
connection.close()


# Xử lý các cột review bị thiếu
review_cols = ['avg_review_score_by_user', 'user_avg_overall_review_score', 'product_avg_review_score']
for col in review_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
    print(f"Đã điền các giá trị thiếu trong cột '{col}' bằng median: {median_value:.2f}")

# Xử lý các cột thuộc tính sản phẩm bị thiếu (chỉ có 1 dòng)
# Ta sẽ xóa các dòng có bất kỳ giá trị nào bị thiếu trong các cột này
product_physic_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
initial_rows = len(df)
df.dropna(subset=product_physic_cols, inplace=True)
rows_dropped = initial_rows - len(df)

print(f"\nĐã xóa {rows_dropped} dòng có giá trị thiếu ở thuộc tính vật lý của sản phẩm.")

# Chuyển đổi các cột ngày tháng sang kiểu datetime
date_cols = ['first_purchase_date', 'last_purchase_date']

for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    print(f"Đã chuyển đổi cột '{col}' sang kiểu datetime.")

# Kiểm tra lại kiểu dữ liệu của các cột này
print("\nKiểu dữ liệu sau khi chuyển đổi:")
print(df[date_cols].dtypes)

# Đếm số lượng dòng bị trùng lặp
duplicate_rows = df.duplicated().sum()
print(f"Tìm thấy {duplicate_rows} dòng dữ liệu bị trùng lặp.")

# Xóa các dòng bị trùng lặp nếu có
if duplicate_rows > 0:
    df.drop_duplicates(inplace=True)
    print("Đã xóa các dòng bị trùng lặp.")
    
    
# Cell Code Hoàn Chỉnh - Đóng gói thành Class Tuner Linh hoạt

# ===================================================================
# 1. IMPORTS VÀ CÀI ĐẶT
# ===================================================================
import pandas as pd
import numpy as np
import os
import tempfile
import mlflow
import surprise
import gc
from surprise import SVD, SVDpp, NMF, SlopeOne,KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from joblib import Parallel, delayed
from typing import Dict, Optional, Any
from ultils import RecSysModelTuner
# Cấu hình MLflow
mlflow.set_tracking_uri("http://172.17.0.1:5000")

# Giả định bạn đã có một DataFrame tên là 'df' đã được làm sạch và tải sẵn

# ===================================================================
# 2. CHUẨN BỊ DỮ LIỆU (Thực hiện một lần ở ngoài)
# ===================================================================
print("Chuẩn bị dữ liệu cho Surprise...")
cf_df = df[['customer_unique_id', 'product_key', 'avg_review_score_by_user']].copy()
cf_df.columns = ['userID', 'itemID', 'rating']
cf_df.dropna(subset=['rating'], inplace=True)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(cf_df, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
full_trainset = data.build_full_trainset()
input_example = cf_df[['userID', 'itemID']].head(5)
print("✅ Dữ liệu đã sẵn sàng!")




# ===================================================================
# 4. FLOW THỬ NGHIỆM CHÍNH (SỬ DỤNG CLASS MỚI)
# ===================================================================
experiment_name = "Collaborative_Filtering_Class_Based_Flow"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="Main_CF_Experiment") as parent_run:
    mlflow.set_tag("project", "Flexible Tuning of CF Models")
    print(f"🚀 Bắt đầu Parent Run: {parent_run.info.run_id}")

    # --- Thử nghiệm SVD ---
    param_grid_svd = {'n_factors': [50, 100], 'n_epochs': [20], 'lr_all': [0.005], 'reg_all': [0.02], 'random_state': [42]}
    svd_tuner = RecSysModelTuner(SVD, "SVD_Tuning", "Matrix Factorization", "svd-recommender-tuned", param_grid=param_grid_svd, conda_env=conda_env_dict,tags=SURPRISE_MODEL_TAGS)
    svd_tuner.run(data, full_trainset, trainset, testset, input_example)
    
    # --- Thử nghiệm SVD++ ---
    param_grid_svdpp = {'n_factors': [20, 50], 'n_epochs': [20], 'lr_all': [0.007], 'reg_all': [0.02], 'random_state': [42]}
    svdpp_tuner = RecSysModelTuner(SVDpp, "SVDpp_Tuning", "Matrix Factorization", "svdpp-recommender-tuned", param_grid=param_grid_svdpp,conda_env=conda_env_dict,tags=SURPRISE_MODEL_TAGS)
    svdpp_tuner.run(data, full_trainset, trainset, testset, input_example)
    
    # --- Thử nghiệm NMF ---
    param_grid_nmf = {'n_factors': [15, 30], 'n_epochs': [50], 'reg_pu': [0.06], 'reg_qi': [0.06], 'random_state': [42]}
    nmf_tuner = RecSysModelTuner(NMF, "NMF_Tuning", "Matrix Factorization", "nmf-recommender-tuned", param_grid=param_grid_nmf,conda_env=conda_env_dict,tags=SURPRISE_MODEL_TAGS)
    nmf_tuner.run(data, full_trainset, trainset, testset, input_example)

    # # --- Thử nghiệm SlopeOne (không tuning) ---
    SAMPLE_FRAC = 0.15 # Lấy 15% dữ liệu
    sampled_cf_df = cf_df.sample(frac=SAMPLE_FRAC, random_state=42)
    
    # Tạo lại các đối tượng surprise từ dữ liệu mẫu
    data_sampled = Dataset.load_from_df(sampled_cf_df, reader)
    trainset_sampled, testset_sampled = train_test_split(data_sampled, test_size=0.2, random_state=42)
    full_trainset_sampled = data_sampled.build_full_trainset()
    print(f"✅ Dữ liệu mẫu đã sẵn sàng với {len(sampled_cf_df)} tương tác.")
    
    slopeone_tuner = RecSysModelTuner(SlopeOne, "SlopeOne_Run_Sampled", "Item-Based", "slopeone-recommender-fixed",conda_env=conda_env_dict,tags=SURPRISE_MODEL_TAGS)
    # Chạy tuner trên bộ dữ liệu đã được lấy mẫu
    slopeone_tuner.run(
        data_sampled, 
        full_trainset_sampled, 
        trainset_sampled, 
        testset_sampled, 
        input_example,
        # Log thêm tham số về việc lấy mẫu
        sample_fraction=SAMPLE_FRAC,
    )
    # --- Thử nghiệm k-NN trên dữ liệu mẫu ---
    knn_tuner = RecSysModelTuner(KNNBasic, "KNN_ItemBased_Sampled_Run", "Neighborhood-Based", "knn-item-recommender-sampled",conda_env=conda_env_dict,tags=SURPRISE_MODEL_TAGS)
    knn_tuner.run(data_sampled, full_trainset_sampled, trainset_sampled, testset_sampled, input_example, sample_fraction=SAMPLE_FRAC)


del df
gc.collect()
print(f"\n🏁 Hoàn thành Parent Run.")