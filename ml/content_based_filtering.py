import os
import mlflow
import pandas as pd
import numpy as np
from databricks import sql
from dotenv import load_dotenv
import json
load_dotenv()
database = "olist_dataset"
goldSche=database+ ".gold"
lookupGeo=database+".silver"+ ".silver_olist_geolocation"
lookupCateName=database+".silver"+ ".silver_product_category_name_translation"
connection = sql.connect(
                        server_hostname = "dbc-214ffec4-c1f2.cloud.databricks.com",
                        http_path = "/sql/1.0/warehouses/8753fa7395d762fe",
                        access_token = os.environ.get("DATABRICKS_TOKEN"))

mlflow.set_tracking_uri("http://172.17.0.1:5000")
query_pro=f"""

WITH product_sales_aggregates AS (
    -- CTE 1: Tổng hợp các đặc tính từ dữ liệu bán hàng và tương tác cơ bản
    SELECT
        oi.product_key,
        COUNT(oi.order_id) AS total_sales_count, -- Tổng số lượt bán
        COUNT(DISTINCT c.customer_unique_id) AS num_unique_purchasers, -- Lượng khách hàng riêng biệt đã mua
        COUNT(DISTINCT oi.seller_key) AS num_sellers, -- Số lượng người bán khác nhau cho sản phẩm này
        
        -- Đặc tính về giá
        AVG(oi.price) AS avg_price, -- Giá bán trung bình
        STDDEV(oi.price) AS stddev_price, -- Độ lệch chuẩn của giá (cho thấy sự biến động giá)
        MIN(oi.price) AS min_price, -- Giá thấp nhất từng được bán
        MAX(oi.price) AS max_price, -- Giá cao nhất từng được bán

        -- Đặc tính về vận chuyển
        AVG(oi.freight_value) AS avg_freight_value, -- Phí vận chuyển trung bình
        AVG(oi.freight_value / NULLIF(oi.price, 0)) AS freight_to_price_ratio, -- Tỷ lệ phí vận chuyển trên giá

        -- Đặc tính về thời gian
        MIN(o.order_purchase_date_key) AS first_sale_date_key, -- Ngày bán đầu tiên
        MAX(o.order_purchase_date_key) AS last_sale_date_key -- Ngày bán gần nhất
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_orders o ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    GROUP BY oi.product_key
),

product_review_aggregates AS (
    -- CTE 2: Tổng hợp các đặc tính từ dữ liệu review của khách hàng
    SELECT
        oi.product_key,
        AVG(r.score) AS avg_review_score, -- Điểm review trung bình
        STDDEV(r.score) AS stddev_review_score, -- Độ lệch chuẩn của điểm review
        COUNT(r.review_id) AS total_reviews_count, -- Tổng số lượng review
        
        -- Tỷ lệ review tích cực/tiêu cực (thường hữu ích hơn điểm trung bình)
        AVG(CASE WHEN r.score >= 4 THEN 1.0 ELSE 0.0 END) AS positive_review_ratio,
        AVG(CASE WHEN r.score <= 2 THEN 1.0 ELSE 0.0 END) AS negative_review_ratio
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.fact_reviews r ON oi.order_key = r.order_key
    GROUP BY oi.product_key
),

product_payment_aggregates AS (
    -- CTE 3: Tổng hợp các đặc tính từ phương thức thanh toán
    SELECT
        oi.product_key,
        AVG(fp.payment_installments) AS avg_payment_installments -- Số kỳ trả góp trung bình
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.fact_payments fp ON oi.order_key = fp.order_key
    GROUP BY oi.product_key
)

-- FINAL SELECT: Kết hợp tất cả các đặc tính lại
SELECT
    -- === A. Đặc tính nội dung tĩnh (Core Content Features) ===
    p.product_key,
    p.product_category_name,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_photos_qty,
    
    -- Đặc tính vật lý
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    -- Feature tự tạo: Thể tích sản phẩm
    (p.product_length_cm * p.product_height_cm * p.product_width_cm) AS product_volume_cm3,

    -- === B. Đặc tính về sự phổ biến & Tương tác (Popularity & Interaction Features) ===
    COALESCE(psa.total_sales_count, 0) AS total_sales_count,
    COALESCE(psa.num_unique_purchasers, 0) AS num_unique_purchasers,
    -- Feature tự tạo: Tỷ lệ chuyển đổi (mua lại)
    (psa.total_sales_count / NULLIF(psa.num_unique_purchasers, 0)) AS repurchase_ratio,
    
    -- === C. Đặc tính về giá & Vận chuyển (Price & Shipping Features) ===
    COALESCE(psa.avg_price, 0) AS avg_price,
    COALESCE(psa.stddev_price, 0) AS stddev_price,
    COALESCE(psa.avg_freight_value, 0) AS avg_freight_value,
    COALESCE(psa.freight_to_price_ratio, 0) AS freight_to_price_ratio,

    -- === D. Đặc tính về Review & Sự hài lòng (Review & Satisfaction Features) ===
    COALESCE(pra.total_reviews_count, 0) AS total_reviews_count,
    COALESCE(pra.avg_review_score, 0) AS avg_review_score, -- Điểm trung bình có thể bị ảnh hưởng bởi vài review, nên dùng tỷ lệ bên dưới
    COALESCE(pra.stddev_review_score, 0) AS stddev_review_score, -- Độ lệch chuẩn cao cho thấy ý kiến trái chiều
    COALESCE(pra.positive_review_ratio, 0) AS positive_review_ratio,
    COALESCE(pra.negative_review_ratio, 0) AS negative_review_ratio,

    -- === E. Đặc tính về Bán hàng & Thanh toán (Seller & Payment Features) ===
    COALESCE(psa.num_sellers, 0) AS num_sellers,
    COALESCE(ppa.avg_payment_installments, 0) AS avg_payment_installments,

    -- === F. Đặc tính về Vòng đời sản phẩm (Lifecycle Features) ===
    psa.first_sale_date_key,
    psa.last_sale_date_key
    -- Bạn có thể tính toán 'tuổi' của sản phẩm ở bước sau bằng cách lấy ngày hiện tại trừ đi first_sale_date_key
    
FROM {goldSche}.dim_olist_products p
LEFT JOIN product_sales_aggregates psa ON p.product_key = psa.product_key
LEFT JOIN product_review_aggregates pra ON p.product_key = pra.product_key
LEFT JOIN product_payment_aggregates ppa ON p.product_key = ppa.product_key
WHERE p.product_category_name IS NOT NULL; -- Chỉ lấy các sản phẩm có danh mục rõ ràng
"""
df_pro = pd.read_sql(query_pro, connection)
df_pro.describe()
df_pro.info()

query_cus=f"""

WITH unique_customers AS (
    -- CTE 0: Tạo bảng khách hàng duy nhất để làm gốc
    -- Chọn ra thông tin city, state từ bản ghi mới nhất của mỗi khách hàng
    SELECT
        customer_unique_id,
        customer_city,
        customer_state
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER(PARTITION BY customer_unique_id ORDER BY customer_key DESC) as rn
        FROM {goldSche}.dim_olist_customers
    )
    WHERE rn = 1
),

customer_transactional_aggregates AS (
    -- CTE 1: Tổng hợp các đặc tính Giao dịch cốt lõi (RFM)
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.price) AS monetary,
        MAX(d.full_date) AS last_purchase_date,
        MIN(d.full_date) AS first_purchase_date,
        COUNT(oi.product_key) AS total_items_purchased,
        SUM(oi.price) / COUNT(DISTINCT o.order_id) AS avg_value_per_order,
        COUNT(oi.product_key) / COUNT(DISTINCT o.order_id) AS avg_items_per_order,
        COUNT(DISTINCT oi.seller_key) AS distinct_sellers_count
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_orders o ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    JOIN {goldSche}.dim_date d ON o.order_purchase_date_key = d.date_key
    GROUP BY c.customer_unique_id
),

customer_product_preferences AS (
    -- CTE 2: Phân tích sở thích về sản phẩm của khách hàng
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT p.product_category_name) AS distinct_categories_purchased,
        ARRAY_JOIN(
            TRANSFORM(
                SORT_ARRAY(COLLECT_LIST(STRUCT(o.order_purchase_date_key, p.product_category_name))),
                x -> x.product_category_name
            ), ' | '
        ) AS purchase_category_history
    FROM {goldSche}.fact_order_items oi
    JOIN {goldSche}.dim_olist_orders o ON oi.order_key = o.order_key
    JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
    JOIN {goldSche}.dim_olist_products p ON oi.product_key = p.product_key
    GROUP BY c.customer_unique_id
),

customer_favorite_category AS (
    -- CTE 3: Tìm danh mục sản phẩm yêu thích nhất
    SELECT DISTINCT customer_unique_id,
           FIRST_VALUE(product_category_name) OVER (PARTITION BY customer_unique_id ORDER BY category_count DESC) as favorite_category
    FROM (
        SELECT c.customer_unique_id, p.product_category_name, COUNT(*) as category_count
        FROM {goldSche}.fact_order_items oi
        JOIN {goldSche}.dim_olist_customers c ON oi.customer_key = c.customer_key
        JOIN {goldSche}.dim_olist_products p ON oi.product_key = p.product_key
        GROUP BY c.customer_unique_id, p.product_category_name
    )
),

customer_review_behavior AS (
    -- CTE 4: Phân tích hành vi đánh giá sản phẩm
    SELECT
        c.customer_unique_id,
        AVG(r.score) AS avg_review_score,
        COUNT(r.review_id) AS total_reviews_given
    FROM {goldSche}.fact_reviews r
    JOIN {goldSche}.dim_olist_customers c ON r.customer_key = c.customer_key
    GROUP BY c.customer_unique_id
),

customer_payment_behavior AS (
    -- CTE 5: Phân tích thói quen thanh toán
    SELECT
        c.customer_unique_id,
        AVG(fp.payment_installments) AS avg_payment_installments,
        MAX(fp.payment_installments) AS max_payment_installments,
        COUNT(DISTINCT fp.payment_type_key) AS distinct_payment_types_used
    FROM {goldSche}.fact_payments fp
    JOIN {goldSche}.dim_olist_orders o ON fp.order_key = o.order_key
    JOIN {goldSche}.dim_olist_customers c ON fp.customer_key = c.customer_key
    GROUP BY c.customer_unique_id
)

-- FINAL SELECT: Kết hợp tất cả lại, bắt đầu từ bảng khách hàng DUY NHẤT
SELECT
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    COALESCE(cta.frequency, 0) AS total_orders,
    COALESCE(cta.monetary, 0) AS total_spent,
    COALESCE(cta.avg_value_per_order, 0) AS avg_value_per_order,
    DATEDIFF(CURRENT_DATE(), cta.last_purchase_date) AS days_since_last_purchase,
    DATEDIFF(cta.last_purchase_date, cta.first_purchase_date) AS customer_lifetime_days,
    DATEDIFF(cta.last_purchase_date, cta.first_purchase_date) / NULLIF(cta.frequency - 1, 0) AS avg_days_between_orders,
    COALESCE(cpp.distinct_categories_purchased, 0) AS distinct_categories_purchased,
    cfc.favorite_category,
    cpp.purchase_category_history,
    COALESCE(crb.avg_review_score, 0) AS avg_review_score_given,
    COALESCE(crb.total_reviews_given, 0) AS total_reviews_given,
    COALESCE(crb.total_reviews_given, 0) / NULLIF(cta.frequency, 0) AS review_propensity,
    COALESCE(cpb.avg_payment_installments, 1) AS avg_payment_installments,
    COALESCE(cpb.max_payment_installments, 1) AS max_payment_installments,
    COALESCE(cpb.distinct_payment_types_used, 0) AS distinct_payment_types_used
FROM unique_customers c 
LEFT JOIN customer_transactional_aggregates cta ON c.customer_unique_id = cta.customer_unique_id
LEFT JOIN customer_product_preferences cpp ON c.customer_unique_id = cpp.customer_unique_id
LEFT JOIN customer_favorite_category cfc ON c.customer_unique_id = cfc.customer_unique_id
LEFT JOIN customer_review_behavior crb ON c.customer_unique_id = crb.customer_unique_id
LEFT JOIN customer_payment_behavior cpb ON c.customer_unique_id = cpb.customer_unique_id;
"""
df_cus = pd.read_sql(query_cus, connection)
df_cus.describe()
df_cus.info()
query_lab=f"""
-- Bảng tương tác để huấn luyện mô hình dự đoán sự hài lòng
-- Label: avg_review_score_by_user
-- Sample Weight: purchase_count

SELECT
    c.customer_unique_id,
    oi.product_key,
    
    -- Dùng cột này làm LABEL (y) cho mô hình.
    -- Đây là biến mục tiêu mà mô hình sẽ cố gắng dự đoán.
    AVG(r.score) AS avg_review_score_by_user,
    
    -- Dùng cột này làm SAMPLE_WEIGHT để cho mô hình biết độ quan trọng của từng mẫu.
    COUNT(oi.order_id) as purchase_count
    
FROM {goldSche}.fact_reviews r
-- Join với fact_order_items để lấy được product_key
JOIN {goldSche}.fact_order_items oi ON r.order_key = oi.order_key
-- Join với dim_customers để lấy customer_unique_id
JOIN {goldSche}.dim_olist_customers c ON r.customer_key = c.customer_key
-- Chỉ lấy các tương tác có review để làm target
GROUP BY c.customer_unique_id, oi.product_key
HAVING AVG(r.score) IS NOT NULL;
"""
df_lab = pd.read_sql(query_lab, connection)
df_lab.describe()
df_lab.info()

connection.close()
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import warnings
import os
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')
# DÁN KHỐI NÀY VÀO ĐẦU FILE CỦA BẠN
import mlflow
import mlflow.pyfunc
import pandas as pd

class RecommenderPipelineWrapper(mlflow.pyfunc.PythonModel):
    """Lớp Wrapper để MLflow hiểu mô hình pipeline của bạn."""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input):
        """
        Hàm này được MLflow gọi khi cần dự đoán.
        model_input là một Pandas DataFrame.
        """
        predictions = []
        for _, row in model_input.iterrows():
            customer_id = row['customer_id']
            
            candidate_ids_str = row['candidate_product_ids'] 

            try:
                # 2. Dùng json.loads để biến string trở lại thành list
                candidate_ids = json.loads(candidate_ids_str)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Lỗi khi giải mã (decode) candidate_product_ids: {e}. Input: {candidate_ids_str}")
                predictions.append(np.array([])) 
                continue
            recommendations = self.pipeline.predict(
                customer_id=customer_id,
                candidate_product_ids=candidate_ids
            )
            predictions.append(recommendations)
            
        return np.array(predictions)
class ContentBasedRecommender(nn.Module):
    """
    Phiên bản cuối cùng, linh hoạt, xây dựng các khối MLP từ config.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        embedding_dim = self.config['embedding_dim']

        # --- Helper function để xây dựng MLP ---
        def _create_mlp(input_dim, output_dim, hidden_dims):
            layers = []
            current_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config['dropout_rate']))
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, output_dim))
            return nn.Sequential(*layers)

        # --- User Tower Layers ---
        self.cust_embeddings = nn.ModuleDict({
            feat: nn.Embedding(config['vocab_sizes'][feat], embedding_dim)
            for feat in config['cust_cat_feats']
        })
        self.cust_seq_embedding = nn.Embedding(config['seq_vocab_size'], embedding_dim, padding_idx=0)
        self.positional_encoding = nn.Embedding(config['max_seq_len'], embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=config['transformer_nhead'], dim_feedforward=config['transformer_dim_feedforward'], dropout=config['dropout_rate'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['transformer_nlayers'])
        
        # === THAY ĐỔI: Xây dựng MLP từ config ===
        self.cust_numerical_mlp = _create_mlp(
            config['num_cust_numerical_feats'], 
            embedding_dim, 
            config['cust_numerical_mlp_dims']
        )
        
        user_tower_input_size = (len(config['cust_cat_feats']) + 2) * embedding_dim
        self.user_tower_mlp = _create_mlp(
            user_tower_input_size,
            embedding_dim,
            config['user_tower_mlp_dims']
        )

        # --- Product Tower Layers ---
        self.prod_embeddings = nn.ModuleDict({
            feat: nn.Embedding(config['vocab_sizes'][feat], embedding_dim)
            for feat in config['prod_cat_feats']
        })
        
        self.prod_numerical_mlp = _create_mlp(
            config['num_prod_numerical_feats'],
            embedding_dim,
            config['prod_numerical_mlp_dims']
        )
        
        prod_tower_input_size = (len(config['prod_cat_feats']) + 1) * embedding_dim
        self.product_tower_mlp = _create_mlp(
            prod_tower_input_size,
            embedding_dim,
            config['product_tower_mlp_dims']
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # (Giữ nguyên)
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.01)

    def forward(self, batch):
        # (Hàm forward giữ nguyên, không thay đổi)
        cust_cat_embeds = [self.cust_embeddings[feat](batch['customer_cat'][:, i]) for i, feat in enumerate(self.config['cust_cat_feats'])]
        seq_embeds = self.cust_seq_embedding(batch['customer_seq'])
        positions = torch.arange(0, seq_embeds.size(1), device=seq_embeds.device).unsqueeze(0)
        transformer_input = seq_embeds + self.positional_encoding(positions)
        transformer_mask = (batch['customer_seq'] == 0)
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=transformer_mask)
        mask = (batch['customer_seq'] != 0).float().unsqueeze(-1)
        seq_embed_agg = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        cust_num_embed = self.cust_numerical_mlp(batch['customer_num'])
        user_combined = torch.cat(cust_cat_embeds + [seq_embed_agg, cust_num_embed], dim=1)
        user_embedding = self.user_tower_mlp(user_combined)
        prod_cat_embeds = [self.prod_embeddings[feat](batch['product_cat'][:, i]) for i, feat in enumerate(self.config['prod_cat_feats'])]
        prod_num_embed = self.prod_numerical_mlp(batch['product_num'])
        product_combined = torch.cat(prod_cat_embeds + [prod_num_embed], dim=1)
        product_embedding = self.product_tower_mlp(product_combined)
        prediction = torch.sum(user_embedding * product_embedding, dim=1)
        return prediction.squeeze()

class RecommenderDataset(Dataset):
    def __init__(self, interactions_df, customers_df_indexed, products_df_indexed, config):
        self.interactions = interactions_df
        self.customers = customers_df_indexed
        self.products = products_df_indexed
        self.config = config
    def __len__(self): return len(self.interactions)
    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        customer_id, product_id = interaction['customer_unique_id'], int(interaction['product_key'])
        label = torch.tensor(interaction['avg_review_score_by_user'], dtype=torch.float32)
        sample_weight = torch.tensor(interaction['purchase_count'], dtype=torch.float32)
        customer_row = self.customers.loc[customer_id]
        cust_cat = torch.tensor([customer_row[col] for col in self.config['cust_cat_feats']], dtype=torch.long)
        cust_num = torch.tensor(customer_row[self.config['cust_num_feats']].values.astype(np.float32), dtype=torch.float32)
        cust_seq = torch.tensor(customer_row[self.config['cust_seq_feat']], dtype=torch.long)
        product_row = self.products.loc[product_id]
        prod_cat = torch.tensor([product_row[col] for col in self.config['prod_cat_feats']], dtype=torch.long)
        prod_num = torch.tensor(product_row[self.config['prod_num_feats']].values.astype(np.float32), dtype=torch.float32)
        return {'customer_cat': cust_cat, 'customer_num': cust_num, 'customer_seq': cust_seq, 'product_cat': prod_cat, 'product_num': prod_num, 'label': label, 'sample_weight': sample_weight}

class RecommenderPipeline:
    def __init__(self, model_config, df_customers, df_products, df_interactions):
        self.config = model_config
        self.df_customers_raw = df_customers
        self.df_products_raw = df_products
        self.df_interactions = df_interactions
        self.device = self._check_device()
        self.model, self.optimizer, self.criterion, self.scheduler = None, None, None, None
        self.encoders, self.scalers = {}, {}
        self.history = {'train_loss': [], 'test_loss': []}
        self.customers_processed, self.products_processed = None, None
    def _check_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Sử dụng thiết bị: {device.upper()}")
        return device
    @staticmethod
    def collate_fn(batch):
        batch_data = {key: [] for key in batch[0].keys()}
        for sample in batch:
            for key, value in sample.items(): batch_data[key].append(value)
        fixed_size_keys = ['customer_cat', 'customer_num', 'product_cat', 'product_num', 'label', 'sample_weight']
        final_batch = {key: torch.stack(batch_data[key]) for key in fixed_size_keys}
        final_batch['customer_seq'] = rnn_utils.pad_sequence(batch_data['customer_seq'], batch_first=True, padding_value=0)
        return final_batch
    @staticmethod
    def _get_feature_lists(df, id_col, exclude_cols=None):
        if exclude_cols is None: exclude_cols = []
        num_feats, cat_feats = [], []
        for col in df.columns:
            if col == id_col or col in exclude_cols: continue
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]): cat_feats.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 30 and df[col].dtype != 'float64': cat_feats.append(col)
                else: num_feats.append(col)
        return num_feats, cat_feats
    def preprocess_data(self):
        print("⚙️  Bắt đầu tiền xử lý dữ liệu...")
        customers_df, products_df = self.df_customers_raw.copy(), self.df_products_raw.copy()
        cust_num_feats, cust_cat_feats = self._get_feature_lists(customers_df, 'customer_unique_id', ['purchase_category_history'])
        prod_num_feats, prod_cat_feats = self._get_feature_lists(products_df, 'product_key')
        self.config.update({'cust_num_feats': cust_num_feats, 'cust_cat_feats': cust_cat_feats, 'cust_seq_feat': 'purchase_category_history', 'prod_num_feats': prod_num_feats, 'prod_cat_feats': prod_cat_feats})
        for col in cust_num_feats: customers_df[col].fillna(customers_df[col].median(), inplace=True)
        for col in prod_num_feats: products_df[col].fillna(products_df[col].median(), inplace=True)
        for col in cust_cat_feats + ['purchase_category_history']: customers_df[col].fillna('<MISSING>', inplace=True)
        for col in prod_cat_feats: products_df[col].fillna('<MISSING>', inplace=True)
        self.df_interactions.dropna(inplace=True)
        all_cat_feats = cust_cat_feats + prod_cat_feats
        for col in all_cat_feats:
            all_values = pd.concat([customers_df[col] if col in customers_df else pd.Series(dtype='object'), products_df[col] if col in products_df else pd.Series(dtype='object')]).astype(str).unique()
            le = LabelEncoder().fit(all_values)
            self.encoders[col] = le
            self.config['vocab_sizes'][col] = len(le.classes_)
            if col in customers_df: customers_df[col] = le.transform(customers_df[col].astype(str))
            if col in products_df: products_df[col] = le.transform(products_df[col].astype(str))
        self.scalers['customer'] = StandardScaler().fit(customers_df[cust_num_feats])
        customers_df[cust_num_feats] = self.scalers['customer'].transform(customers_df[cust_num_feats])
        self.scalers['product'] = StandardScaler().fit(products_df[prod_num_feats])
        products_df[prod_num_feats] = self.scalers['product'].transform(products_df[prod_num_feats])
        cat_encoder = self.encoders.get('product_category_name', self.encoders.get('favorite_category'))
        vocab = {cat: i + 1 for i, cat in enumerate(cat_encoder.classes_)}
        vocab['<MISSING>'] = len(vocab) + 1
        self.config['seq_vocab_size'] = len(vocab) + 1
        sequences = customers_df['purchase_category_history'].apply(lambda x: x.split(' | ') if x != '<MISSING>' else [])
        tokenized = [[vocab.get(token, 0) for token in seq] for seq in sequences]
        customers_df['purchase_category_history'] = tokenized
        self.customers_processed = customers_df.set_index('customer_unique_id')
        self.products_processed = products_df.set_index('product_key')
        print("✅ Tiền xử lý hoàn tất.")
    def create_dataloaders(self, test_size=0.2, batch_size=64, num_workers=0):
        print("📦 Tạo DataLoaders...")
        train_df, test_df = train_test_split(self.df_interactions, test_size=test_size, random_state=42)
        train_dataset = RecommenderDataset(train_df, self.customers_processed, self.products_processed, self.config)
        test_dataset = RecommenderDataset(test_df, self.customers_processed, self.products_processed, self.config)
        pin_memory = True if self.device == 'cuda' else False
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        print(f"✅ DataLoaders đã sẵn sàng (num_workers={num_workers}, pin_memory={pin_memory}).")

    def build_model(self, learning_rate=1e-3):
        print("🏗️  Xây dựng mô hình...")
        self.config['num_cust_numerical_feats'] = len(self.config['cust_num_feats'])
        self.config['num_prod_numerical_feats'] = len(self.config['prod_num_feats'])
        
        # === THAY ĐỔI 2: Tính max_seq_len an toàn và hiệu quả hơn ===
        max_len = max(len(x) for x in self.customers_processed[self.config['cust_seq_feat']]) if not self.customers_processed.empty else 0
        self.config['max_seq_len'] = max_len if max_len > 0 else 1 # Tránh max_len = 0
        
        self.model = ContentBasedRecommender(self.config).to(self.device)
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
        print("✅ Mô hình đã được xây dựng.")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in loop:
            for key, value in batch.items(): batch[key] = value.to(self.device)
            predictions = self.model(batch)
            unweighted_loss = self.criterion(predictions, batch['label'])
            weighted_loss = (unweighted_loss * batch['sample_weight']).mean()
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()
            total_loss += weighted_loss.item()
            loop.set_postfix(loss=weighted_loss.item())
        return total_loss / len(self.train_loader)
    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                for key, value in batch.items(): batch[key] = value.to(self.device)
                predictions = self.model(batch)
                loss = torch.sqrt(self.criterion(predictions, batch['label']).mean())
                total_loss += loss.item()
        return total_loss / len(data_loader)
    def train(self, epochs=10, learning_rate=1e-3, test_size=0.2, batch_size=64, num_workers=0):
        self.preprocess_data()
        self.create_dataloaders(test_size=test_size, batch_size=batch_size, num_workers=num_workers)
        self.build_model(learning_rate=learning_rate)
        print("\n" + "="*50 + "\n🚀 BẮT ĐẦU HUẤN LUYỆN\n" + "="*50)
        for epoch in range(epochs):
            train_loss = self._train_epoch()
            test_rmse = self._evaluate(self.test_loader)
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_rmse)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss (Weighted MSE): {train_loss:.4f} | Test RMSE: {test_rmse:.4f} | Next LR: {current_lr:.10f}")
        print("\n🎉 Huấn luyện hoàn tất!")
        return self.history
    def predict(self, customer_id: str, candidate_product_ids: list):
        if not self.model: raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi hàm .train() trước.")
        self.model.eval()
        try:
            customer_row = self.customers_processed.loc[customer_id]
            candidate_product_ids = [int(pid) for pid in candidate_product_ids]
            product_rows = self.products_processed.loc[candidate_product_ids]
        except KeyError as e:
            print(f"Lỗi: Không tìm thấy customer_id hoặc một trong các product_id. {e}")
            return None
        num_candidates = len(candidate_product_ids)
        with torch.no_grad():
            cust_cat = torch.tensor([customer_row[col] for col in self.config['cust_cat_feats']], dtype=torch.long)
            cust_num = torch.tensor(customer_row[self.config['cust_num_feats']].values.astype(np.float32), dtype=torch.float32)
            cust_seq = torch.tensor(customer_row[self.config['cust_seq_feat']], dtype=torch.long)
            cust_seq_padded = torch.nn.functional.pad(cust_seq, (0, self.config['max_seq_len'] - len(cust_seq)), 'constant', 0)
            
            # === THAY ĐỔI 3: Sửa lỗi tham chiếu sai tên biến ===
            cust_cat_batch = cust_cat.unsqueeze(0).repeat(num_candidates, 1)
            cust_num_batch = cust_num.unsqueeze(0).repeat(num_candidates, 1)
            cust_seq_batch = cust_seq_padded.unsqueeze(0).repeat(num_candidates, 1)
            
            prod_cat_batch = torch.tensor(product_rows[self.config['prod_cat_feats']].values, dtype=torch.long)
            prod_num_batch = torch.tensor(product_rows[self.config['prod_num_feats']].values.astype(np.float32), dtype=torch.float32)
            batch = {'customer_cat': cust_cat_batch.to(self.device), 'customer_num': cust_num_batch.to(self.device), 'customer_seq': cust_seq_batch.to(self.device), 'product_cat': prod_cat_batch.to(self.device), 'product_num': prod_num_batch.to(self.device)}
            scores = self.model(batch)
        results_df = pd.DataFrame({'product_key': candidate_product_ids, 'predicted_score': scores.cpu().numpy()})
        return results_df.sort_values('predicted_score', ascending=False)



import datetime
import cloudpickle
import sklearn
# THAY THẾ TOÀN BỘ KHỐI __main__ CŨ BẰNG KHỐI NÀY
if __name__ == '__main__':
    mlflow.set_tracking_uri("http://172.17.0.1:5000")
    # --- Bước 2: Định nghĩa Config ---
    model_config = {
        'embedding_dim': 32, 'dropout_rate': 0.1, 'transformer_nhead': 8,
        'transformer_nlayers': 16, 'transformer_dim_feedforward': 32, 'vocab_sizes': {},
        'cust_numerical_mlp_dims': [128,64,32], 'user_tower_mlp_dims': [128,64,32],
        'prod_numerical_mlp_dims': [128,128,64], 'product_tower_mlp_dims': [128,64],
    }
    
    # Các tham số huấn luyện
    training_params = {
        'epochs': 20, 'learning_rate': 0.01,
        'test_size': 0.2, 'batch_size': 256,
    }

    # --- Tích hợp MLflow ---
    mlflow.set_experiment("Deep_Content_Based_Filtering") # Đặt tên cho thí nghiệm

    with mlflow.start_run(run_name=f"Deep_CB_Model_Training- {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"): # Bắt đầu một lần chạy
        # 1. Log các tham số
        mlflow.log_params(model_config)
        mlflow.log_params(training_params)

        # 2. Khởi tạo và Huấn luyện Pipeline
        pipeline = RecommenderPipeline(
            model_config=model_config, df_customers=df_cus,
            df_products=df_pro, df_interactions=df_lab
        )
        num_workers = os.cpu_count()
        
        # Gọi train và nhận lại history
        history = pipeline.train(
            epochs=training_params['epochs'], learning_rate=training_params['learning_rate'],
            test_size=training_params['test_size'], batch_size=training_params['batch_size'],
            num_workers=num_workers
        )
        
        # 3. Log các chỉ số (metrics)
        for metric_name, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=epoch)
        INFERENCE_DL_PIP_REQUIREMENTS = [
            # Core MLflow & Serialization
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            f"mlflow=={mlflow.__version__}",
            f"cloudpickle=={cloudpickle.__version__}", # MLflow dùng để load pyfunc model

            # Data Handling
            f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}",

            "torch==2.8.0 ",
            "tqdm",

            # Scikit-learn (Cần cho class definition khi unpickle encoder/scaler)
            f"scikit-learn=={sklearn.__version__}",
        ]
        print("✅ Huấn luyện xong. Di chuyển mô hình về CPU để chuẩn bị lưu...")
        pipeline.model = pipeline.model.to('cpu') # Lệnh pipeline.model.to('cpu') không thay đổi pipeline.model. Thay vào đó, nó trả về một BẢN SAO (copy) của model đã được chuyển sang CPU.
        pipeline.device = 'cpu'
        print("  -> Dọn dẹp optimizer và scheduler (state của GPU)...")
        pipeline.optimizer = None
        pipeline.scheduler = None
        pipeline.criterion = None # Xóa luôn criterion cho chắc
        # 4. Log model
        wrapper = RecommenderPipelineWrapper(pipeline)
        input_example_df = pd.DataFrame({
            "customer_id": [df_lab['customer_unique_id'].iloc[0]],
            "candidate_product_ids": [json.dumps([1001, 1002, 1003])]
        })
        mlflow.pyfunc.log_model(
            artifact_path="dl_recommender_model",
            python_model=wrapper,
            input_example=input_example_df,
            registered_model_name="Deep Content-Based Recommender",
            pip_requirements=INFERENCE_DL_PIP_REQUIREMENTS
            
        )
        print("✅ Đã log model và các thông số vào MLflow.")

    print("\n--- 4. Ví dụ về Dự đoán ---")
    target_customer_id = df_lab['customer_unique_id'].iloc[0]
    recommendations = pipeline.predict(
        customer_id=target_customer_id,
        candidate_product_ids=[1001, 1002, 1003] 
    )
    print(recommendations)