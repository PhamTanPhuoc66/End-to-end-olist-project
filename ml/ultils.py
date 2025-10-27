import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Optional
import mlflow
import pandas as pd
import numpy as np
import gc
from mlflow.tracking import MlflowClient

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import cloudpickle
from IPython import get_ipython
from IPython.display import display
import tempfile
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
# Vá tạm: nếu thiếu _parameter_constraints thì gán một dict trống
if not hasattr(RandomForestClassifier, "_parameter_constraints"):
    RandomForestClassifier._parameter_constraints = {}
import imblearn, sys
from imblearn.pipeline import Pipeline as ImPipeline
from mlflow.models.signature import infer_signature

# Patch để tránh PicklingError khi dump/load
sys.modules['imblearn.pipeline'] = imblearn.pipeline

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
# Top-level wrapper class (module-level giúp pickle ổn định hơn)
class _JoblibPipelineWrapper(mlflow.pyfunc.PythonModel):
    """
    Lớp wrapper tuân thủ theo chuẩn `mlflow.pyfunc.PythonModel`.
    
    Nó đóng gói một pipeline đã được huấn luyện (lưu bằng joblib) để đảm bảo
    việc load và predict hoạt động một cách nhất quán khi triển khai mô hình
    thông qua MLflow, đặc biệt hữu ích khi pipeline chứa các thành phần phức tạp
    hoặc các thành phần từ thư viện cuML.
    """
    def load_context(self, context):
        from joblib import load
        pipeline_path = context.artifacts["pipeline"]
        self.pipeline = load(pipeline_path)

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        Phương thức predict với type hints để đáp ứng yêu cầu của MLflow.
        """
        return self.pipeline.predict(model_input)

@contextmanager
def _temporarily_disable_cuml_accel():
    """
    Context manager để tạm thời vô hiệu hóa extension `cuml.accel` trong môi trường IPython.
    
    Việc này cần thiết để tránh lỗi `PicklingError` khi lưu (serializing) các đối tượng
    từ thư viện cuML bằng `joblib` hoặc `cloudpickle`. Extension sẽ được tự động
    tải lại sau khi thoát khỏi khối `with`.
    """
    """
    Unload cuml.accel (if running in IPython) then yield.
    After yield, reload it if we unloaded it.
    Works in Jupyter/Colab. If not in IPython, does nothing.
    """
    ip = get_ipython()
    unloaded = False
    if ip is not None:
        try:
            # try to unload (no error if not loaded)
            ip.run_line_magic("unload_ext", "cuml.accel")
            unloaded = True
            print("[info] cuml.accel unloaded for safe pickling...")
        except Exception:
            # either not loaded or unload failed — ignore
            unloaded = False
    
    try:
        yield
    finally:
        if ip is not None and unloaded:
            try:
                ip.run_line_magic("load_ext", "cuml.accel")
                print("[info] cuml.accel reloaded.")
            except Exception:
                pass
class AdvancedModelTuner:
    """
    Pipeline chuẩn cho ML experiment:
    - ColumnTransformer input sẵn
    - Tùy chọn sampler (SMOTE/ADASYN)
    - GridSearchCV tuning
    - In data đã xử lý, metrics, confusion matrix
    - Vẽ feature importance nếu có
    Note: dùng gpu thì đừng truyền n_jobs.
    """
    """
    Một class đa năng để tự động hóa quy trình thử nghiệm Machine Learning.

    Class này đóng gói các bước phổ biến như xây dựng pipeline, tinh chỉnh
    siêu tham số, đánh giá mô hình và ghi lại kết quả bằng MLflow. Nó được
    thiết kế để hoạt động linh hoạt: có thể tự quản lý một MLflow run hoàn chỉnh
    hoặc tạo một run con (nested run) bên trong một run đang hoạt động.

    Thuộc tính (Attributes):
        search_strategy (Any): Đối tượng chiến lược tìm kiếm (vd: GridSearchCV).
        preprocessor (Any): Bộ tiền xử lý dữ liệu (vd: ColumnTransformer).
        sampler (Optional[Any]): Đối tượng lấy mẫu để xử lý mất cân bằng (vd: SMOTE).
        experiment_name (Optional[str]): Tên của MLflow experiment.
        manage_run (bool): Cờ để bật/tắt việc quản lý run tự động.
        pipeline (Optional[Pipeline]): Pipeline hoàn chỉnh được xây dựng trong `fit`.
        best_estimator_ (Optional[Pipeline]): Pipeline tốt nhất được tìm thấy sau khi tuning.
        train_results_ (Optional[Dict]): Từ điển chứa các metrics trên tập huấn luyện.
        test_results_ (Optional[Dict]): Từ điển chứa các metrics trên tập kiểm tra.
    """
    def __init__(self, 
                 preprocessor: Any, 
                 model: Optional[Any] = None,
                 search_strategy: Optional[Any] = None,
                 
                 sampler: Optional[Any] = None, 
                 experiment_name=None,
                 manage_run: bool = True,
                 auto_cleanup: bool = True,
                 pip_requirements: Optional[list] = None,  
                 conda_env: Optional[str] = None
                 ):
        

        """
        Khởi tạo AdvancedModelTuner.

        Args:
            model (Optional[Any]): Model cơ sở cần huấn luyện (vd: XGBClassifier(), LogisticRegression()). (model or search)
            preprocessor (Any): Bộ tiền xử lý dữ liệu (vd: ColumnTransformer).
            search_strategy (Optional[Any]): Một instance đã được cấu hình của một lớp tìm kiếm
                                           (vd: GridSearchCV). Nếu là None, model sẽ được fit
                                           mà không cần tuning.
            sampler (Optional[Any], optional): Một instance của một lớp sampler từ
                                               `imblearn` (vd: SMOTE). Mặc định là None.
            experiment_name (Optional[str], optional): Tên của MLflow experiment. Nếu không
                                                       cung cấp, sẽ dùng "default_experiment".
            manage_run (bool, optional): Nếu là True (mặc định), class sẽ tự động quản lý
                                         việc bắt đầu và kết thúc MLflow run. Nếu False,
                                         người dùng phải tự quản lý run từ bên ngoài.
                                         
                                         
                        -->chỉ false nếu lồng thôi và chỉ có 1 run , còn nếu lồng mà true thì là run lồng run
                        FALSE VÀ KO LỒNG THÌ KO LOG.
        """
        self.search_strategy = search_strategy
        self.model=model
        self.experiment_name=experiment_name
        self.sampler = sampler
        self.preprocessor = preprocessor
        
        
        self.manage_run = manage_run
        self.auto_cleanup = auto_cleanup
        
        self.pipeline: Optional[Pipeline] = None
        self.best_estimator_: Optional[Pipeline] = None
        self.train_results_: Optional[Dict] = None
        self.test_results_: Optional[Dict] = None
        
        
        if pip_requirements and conda_env:
            raise ValueError("Chỉ có thể cung cấp 'pip_requirements' hoặc 'conda_env', không phải cả hai.")
        self.pip_requirements = pip_requirements
        self.conda_env = conda_env
        
        
        self.run_ = None# Thêm thuộc tính để giữ run context
        
    def __enter__(self):
        """Bắt đầu MLflow run khi vào khối lệnh `with`.
            Tự động phát hiện và tạo run lồng nhau nếu cần
        """

        if not self.manage_run:
            return self

        # Kiểm tra xem có run nào đang hoạt động không
        if mlflow.active_run():
            self._is_nested = True
            print("--- Active run detected. Creating a nested run... ---")
            # Nếu có, tạo một run con (nested)
            self.run_ = mlflow.start_run(run_name=self.get_run_name(), nested=True)
        else:
            self._is_nested = False
            print("--- No active run. Creating a new parent run... ---")
            # Nếu không, tạo một run cha mới
            mlflow.set_experiment(self.experiment_name or "default_experiment")
            self.run_ = mlflow.start_run(run_name=self.get_run_name())
        return self
    def cleanup(self):
        """
        Xóa các thuộc tính chiếm nhiều bộ nhớ và gọi garbage collector.
        Hữu ích để giải phóng RAM và VRAM (với cuML).
        """
        print("\n" + "="*30)
        print("🧹 Bắt đầu tự động dọn dẹp bộ nhớ...")
        
        # Danh sách các thuộc tính cần xóa
        attrs_to_delete = [
            'best_estimator_',
            'search_strategy',
            'pipeline',
            'train_results_',
            'test_results_',
            'model' # Xóa cả model gốc nếu có
        ]
        
        for attr in attrs_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)
                print(f"   -> Đã xóa self.{attr}")
        
        # Gọi garbage collector
        gc.collect()
        print("👍 Dọn dẹp hoàn tất. Bộ nhớ đã được giải phóng.")
        print("="*30)
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Kết thúc MLflow run khi ra khỏi khối lệnh `with`."""
        if self.manage_run and self.run_ is not None:
            mlflow.end_run()
            if self._is_nested:
                print("--- Nested run finished. ---")
            else:
                print("--- Parent run finished. ---")
        if self.auto_cleanup:
            self.cleanup()
    
        

    def get_run_name(self):
        """
        Lấy tên run một cách linh hoạt dựa trên model hoặc search_strategy.
        """
        model_obj = None
        suffix = "run"

        if self.search_strategy:
            # Ưu tiên lấy model từ search_strategy nếu có
            model_obj = self.search_strategy.estimator
            suffix = "tuning"
        elif self.model:
            # Nếu không, lấy từ model cơ sở
            model_obj = self.model
            suffix = "training"
        
        if model_obj is None:
            return "Untitled_Run" # Trả về tên mặc định nếu không có model

        # Lấy tên class từ model object
        if hasattr(model_obj, 'steps'):
             # Nếu model là một pipeline, lấy tên của bước cuối cùng
             model_name = model_obj.steps[-1][1].__class__.__name__
        else:
             model_name = model_obj.__class__.__name__
             
        return f"{model_name}_{suffix}"
    def _prepare_param_grid(self):
        """
        Hàm nội bộ để tự động thêm tiền tố 'clf__' vào các tham số 
        chưa có tiền tố trong lưới tìm kiếm.
        """
        if hasattr(self.search_strategy, 'param_grid'):
            param_key = 'param_grid'
        elif hasattr(self.search_strategy, 'param_distributions'):
            param_key = 'param_distributions'
        elif hasattr(self.search_strategy, 'search_spaces'):
            param_key = 'search_spaces'
        else:
            raise AttributeError(
                "Không tìm thấy thuộc tính lưới tham số (param_grid, "
                "param_distributions, or search_spaces) trong search_strategy."
            )
        original_params = getattr(self.search_strategy, param_key)

        new_params = {}
        for k, v in original_params.items():
            if '__' not in k:
                new_params['clf__' + k] = v
            else:
                new_params[k] = v
        
        setattr(self.search_strategy, param_key, new_params)
        print("[info] Parameter grid has been prepared with prefixes.")
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,**fit_params):
        """
        Xây dựng pipeline, thực hiện tìm kiếm siêu tham số và lưu lại kết quả.

        Các bước thực hiện:
        1. Lắp ráp pipeline hoàn chỉnh từ `preprocessor`, `sampler` (nếu có), và model.
        2. Cấu hình lại `search_strategy` để hoạt động với pipeline này.
        3. Chạy `fit` của `search_strategy` trên dữ liệu huấn luyện.
        4. Lưu lại estimator tốt nhất (`best_estimator_`).
        5. Log các tham số và score tốt nhất vào MLflow (nếu `manage_run`=True).
        6. Log model tốt nhất vào MLflow.
        7. In kết quả tóm tắt và hiển thị một vài dòng dữ liệu đã qua xử lý.

        Args:
            X_train (pd.DataFrame): DataFrame chứa các đặc trưng của tập huấn luyện.
            y_train (pd.Series): Series chứa nhãn của tập huấn luyện.

        Returns:
            AdvancedModelTuner: Trả về chính instance của class để có thể gọi chaining.
        """
        if self.search_strategy:
            base_model=self.search_strategy.estimator
            
            steps = [('preprocessor', self.preprocessor)]
            if self.sampler:
                steps.append(('sampler', self.sampler))
            steps.append(('clf', base_model))

            self.pipeline = ImPipeline(steps)
            self.search_strategy.estimator = self.pipeline
            
            
            self._prepare_param_grid()

            print("Bắt đầu quá trình tuning...")
            

            self.search_strategy.fit(X_train, y_train, **fit_params)
            self.best_estimator_ = self.search_strategy.best_estimator_
            
            ######### LOGGING TRỰC TIẾP
            best_params = {k.replace('clf__',''):v for k,v in self.search_strategy.best_params_.items()}
            # Log vào MLflow (chỉ log nếu run đang được quản lý)
            if self.manage_run:
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", self.search_strategy.best_score_)
                
                
                # Log model
                self.log_best_model(X_train.dropna())
        
        ###########     
        else:
            # --- PHẦN 2: LOGIC FIT BÌNH THƯỜNG (KHI KHÔNG CÓ SEARCH) ---
            print("Không có search strategy, tiến hành fit model bình thường...")
            
            # Giả định bạn có self.model khi search_strategy=None
            if not hasattr(self, 'model'):
                raise AttributeError("Khi 'search_strategy' là None, instance phải có thuộc tính 'model'.")
            
            base_model = self.model
            
            steps = [('preprocessor', self.preprocessor)]
            if self.sampler:
                steps.append(('sampler', self.sampler))
            steps.append(('clf', base_model))
            
            self.pipeline = ImPipeline(steps)
            
            # Truyền fit_params với prefix 'clf__' cho bước model
            prefixed_fit_params = {f'clf__{k}': v for k, v in fit_params.items()}
            self.pipeline.fit(X_train, y_train, **prefixed_fit_params)
            
            # Pipeline đã được fit chính là estimator tốt nhất
            self.best_estimator_ = self.pipeline
            
            print("\n=== Model đã được huấn luyện với tham số mặc định ===")
            best_params = base_model.get_params()
            # Log các tham số đã dùng
            if self.manage_run:
                mlflow.log_params(base_model.get_params())
            self.log_best_model(X_train.dropna())
        if self.search_strategy:
            # Hiển thị thông tin
            print("\n=== Best Parameters ===")
            print(best_params)
            scoring_metric = self.search_strategy.scoring or 'score'
            print(f"Best CV {scoring_metric}: {self.search_strategy.best_score_:.4f}")
            
        
        # Show processed data (1-2 batch)
        fitted_preprocessor = self.best_estimator_.named_steps['preprocessor']
        X_processed = fitted_preprocessor.transform(X_train)
        if isinstance(X_processed, pd.DataFrame):
            display(X_processed.head())
            print(X_processed.describe())
        else:
            # Nếu output là np.array
            print(pd.DataFrame(X_processed).head())
            print(pd.DataFrame(X_processed).describe())

        return self


    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'Test'):
        """
        Metrics + classification report.
        """

        if self.best_estimator_ is not None:
            y_pred = self.best_estimator_.predict(X)
        elif self.model is not None:
            y_pred = self.model.predict(X)
        else:
            # Ném ra lỗi nếu không có model nào để dùng
            raise RuntimeError("Không có model nào để đánh giá. Vui lòng chạy fit() trước.")
        print(f"\n=== Evaluation on {dataset_name} ===")
        print(classification_report(y, y_pred, digits=4))
        results = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1": f1_score(y, y_pred, average='weighted')
        }
        ####### LOGGING TRỰC TIẾP VỚI PREFIX
        mlflow.log_metrics({f"{dataset_name.lower()}_{k}": v for k, v in results.items()})
        #######
        return results

    def evaluate_train_test(self, X_train, y_train, X_test, y_test):
        self.train_results_ = self.evaluate(X_train, y_train, 'Train')
        self.test_results_ = self.evaluate(X_test, y_test, 'Test')
        print(f"\n--- Summary ---")
        print(f"Train F1: {self.train_results_['f1']:.4f}")
        print(f"Test F1:  {self.test_results_['f1']:.4f}")

    def plot_confusion_matrix(self, X, y, dataset_name='Test'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        if self.best_estimator_ is not None:
            y_pred = self.best_estimator_.predict(X)
        elif self.model is not None:
            y_pred = self.model.predict(X)
        else:
            # Ném ra lỗi nếu không có model nào để dùng
            raise RuntimeError("Không có model nào để đánh giá. Vui lòng chạy fit() trước.")
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {dataset_name}')
        # plt.show()

    def plot_feature_importance(self):
        """
        Vẽ feature importance nếu model có .feature_importances_ hoặc .coef_
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        model = self.best_estimator_.named_steps['clf'] or self.model
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = model.coef_.ravel()
            else:
                print("Model không có feature_importances_ hoặc coef_")
                return
            # Lấy tên feature từ preprocessor
            feature_names = []
            preprocessor = self.best_estimator_.named_steps['preprocessor']

            for name, trans, cols in preprocessor.transformers_:
                if hasattr(trans, 'named_steps'):
                    # Pipeline trong ColumnTransformer
                    last_step = list(trans.named_steps.values())[-1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        feature_names.extend(last_step.get_feature_names_out(cols))
                    else:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
            df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df = df.sort_values('importance', ascending=False)
            # ===================

            # 1. Tạo figure và axes một cách tường minh
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 2. Vẽ biểu đồ trên axes đã tạo
            sns.barplot(data=df, x='importance', y='feature', ax=ax)
            ax.set_title('Top 20 Feature Importances')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            fig.tight_layout() # Căn chỉnh cho đẹp

            # 3. LOG BIỂU ĐỒ VÀO MLFLOW
            mlflow.log_figure(fig, "feature_importance.png")
            
            # 4. Hiển thị và dọn dẹp
            # plt.show()
            plt.close(fig) 
            # ===================

        except Exception as e:
            print("Error plotting feature importance:", e)

    def get_cv_results_df(self):
        if not hasattr(self.search_strategy, 'cv_results_') :
            print("The search strategy has not been fitted or does not have cv_results_.")
            return None
        return pd.DataFrame(self.search_strategy.cv_results_)

    def save_pipeline(self, filepath: str):
        if self.best_estimator_ is None and self.model is None:
            raise RuntimeError("Fit trước khi lưu pipeline.")
        
        dump(self.best_estimator_, filepath) or dump(self.model, filepath)
        print(f"Pipeline saved to {filepath}")

    @staticmethod
    def load_pipeline(filepath: str):
        pipeline = load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline



    def log_best_model(self, X_sample: 'pd.DataFrame'):
        """
        Logs the model reliably using the mlflow.pyfunc flavor, which offers
        more control and avoids sklearn-specific serialization issues with cuML.
        """
        if self.best_estimator_ is None and self.model is None:
            raise RuntimeError("Fit the model before logging.")

        from mlflow.models import infer_signature
        import mlflow
        import tempfile
        import os
        import joblib

        artifact_name = self.best_estimator_.named_steps['clf'].__class__.__name__ or self.model.__class__.__name__
        registered_model_name = f"{self.experiment_name}_{artifact_name}"
        
        signature = None
        input_example = X_sample.dropna().head()

        if not input_example.empty:
            try:
                # 1. Chạy predict để lấy output
                if self.best_estimator_ is not None:
                    predictions = self.best_estimator_.predict(input_example)
                elif self.model is not None:
                    predictions = self.model.predict(input_example)
                else:
                    raise RuntimeError("No model found for signature inference.")

                
                # Để tương thích, chuyển đổi output từ cuDF sang numpy nếu cần
                if 'cudf' in str(type(predictions)):
                    print("[info] Converting cuDF predictions to numpy for signature inference.")
                    predictions = predictions.to_numpy()

                # 2. Tự suy luận signature
                signature = infer_signature(input_example, predictions)
                
                print("\n--- Verifying Inferred Signature ---")
                print(signature)
                print("------------------------------------\n")
                print("✔️ Signature inferred successfully.")

            except Exception as e:
                print(f"[warning] Could not manually infer signature: {e}")
        
        print("[info] Using robust mlflow.pyfunc logging strategy...")
        with _temporarily_disable_cuml_accel():
            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline_file = os.path.join(tmpdir, "pipeline.joblib")
                joblib.dump(self.best_estimator_, pipeline_file) or joblib.dump(self.model, pipeline_file)
                utils_abs_path = os.path.abspath(__file__)
                print(f"--- DEBUG: Absolute path for code_paths: {utils_abs_path}") # In ra để kiểm tra
                # Log model sử dụng pyfunc flavor, cách này rõ ràng và tường minh hơn
                mlflow.pyfunc.log_model(
                    name=artifact_name,
                    code_paths=[utils_abs_path],
                    python_model=_JoblibPipelineWrapper(),
                    artifacts={"pipeline": pipeline_file},
                    registered_model_name=registered_model_name,
                    signature=signature,
                    # **ĐIỂM SỬA LỖI QUAN TRỌNG NHẤT**
                    # Đặt input_example=None để ngăn MLflow chạy predict tự động,
                    # việc này sẽ tránh được lỗi `check_is_fitted` và `AttributeError`.
                    input_example=None,
                    pip_requirements=self.pip_requirements, 
                    conda_env=self.conda_env                
                )
                print(f"✔️ Logged pipeline via pyfunc + joblib as '{artifact_name}'")

    def fit_discrete(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    X_val: pd.DataFrame = None, 
                    y_val: pd.Series = None,
                    **fit_params):
        """
        Thực hiện quy trình tuning theo các bước rời rạc, linh hoạt hơn.

        Quy trình:
        1. Fit bộ tiền xử lý (preprocessor) trên dữ liệu train và transform cả train/val set.
        2. Chỉ tune siêu tham số cho model (classifier) trên dữ liệu đã được xử lý.
        - Nếu validation set được cung cấp, nó sẽ được truyền vào `fit_params` để
            model có thể tự sử dụng (ví dụ cho early stopping).
        3. Lắp ráp preprocessor đã fit và model tốt nhất thành một pipeline cuối cùng
        mà KHÔNG cần huấn luyện lại.
        """
        print("--- Bắt đầu quy trình Fit Rời Rạc (Discrete Fit) ---")
        
        # === BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ===
        print("🔄 Bước 1: Tiền xử lý dữ liệu...")
        # Fit và transform trên tập train
        X_train_proc = self.preprocessor.fit_transform(X_train, y_train)
        
        # Chuẩn bị validation set nếu được cung cấp
        if X_val is not None and y_val is not None:
            print("    -> Đang transform dữ liệu validation...")
            X_val_proc = self.preprocessor.transform(X_val)
            # Thêm vào fit_params để search_strategy có thể truyền vào model
            fit_params['eval_set'] = [(X_val_proc, y_val)]

        # === BƯỚC 2: TINH CHỈNH SIÊU THAM SỐ CHO MODEL ===
        print(f"🚀 Bước 2: Tinh chỉnh siêu tham số cho model...")
        
        # Lấy ra model gốc và đảm bảo search_strategy đang tune trên nó
        base_model = self.search_strategy.estimator
        self.search_strategy.estimator = base_model
        
        # Chạy tuning trên dữ liệu đã được tiền xử lý
        self.search_strategy.fit(X_train_proc, y_train, **fit_params)
        
        # Lấy ra model tốt nhất đã được huấn luyện
        best_model_only = self.search_strategy.best_estimator_

        # # In và log kết quả tuning
        best_params = self.search_strategy.best_params_
        print("\n=== Các tham số tốt nhất tìm được ===")
        print(best_params)

        scoring_metric = self.search_strategy.scoring or 'score'
        print(f"Best CV {scoring_metric}: {self.search_strategy.best_score_:.4f}")

        if self.manage_run:
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", self.search_strategy.best_score_)

        # 🔍 Kiểm tra số cây thực tế khi dùng early stopping
        best_model_only = self.search_strategy.best_estimator_
        if hasattr(best_model_only, "best_iteration") and best_model_only.best_iteration is not None:
            actual_estimators = best_model_only.best_iteration
            print(f"✅ Early Stopping đã kích hoạt! Số cây tối ưu: {actual_estimators}")
            if self.manage_run:
                mlflow.log_metric("actual_n_estimators", actual_estimators)
        else:
            print("ℹ️ Early Stopping không được kích hoạt. Model đã chạy hết số cây `n_estimators`.")

            
        # === BƯỚC 3: LẮP RÁP PIPELINE CUỐI CÙNG ===
        print("\n🔧 Bước 3: Lắp ráp pipeline hoàn chỉnh cho production...")
        
        # Ghép preprocessor đã được fit và model tốt nhất đã được fit
        steps = [('preprocessor', self.preprocessor)]
        if self.sampler:
            steps.append(('sampler', self.sampler))
        steps.append(('clf', best_model_only))
        
        # Tạo pipeline cuối cùng mà không cần .fit() lại
        final_pipeline = ImPipeline(steps)
        
        # Gán vào thuộc tính của class để có thể sử dụng sau này
        self.best_estimator_ = final_pipeline
        self.pipeline = final_pipeline

        # Log pipeline hoàn chỉnh vào MLflow
        if self.manage_run:
            print("📦 Logging pipeline cuối cùng vào MLflow...")
            self.log_best_model(X_train.dropna())

        print("\n✅ Quy trình fit rời rạc hoàn tất. Pipeline đã sẵn sàng để sử dụng.")
        return self
    
try:
    import surprise   
    from surprise import SVD, SVDpp, NMF, SlopeOne, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split, GridSearchCV
    from joblib import Parallel, delayed
    SURPRISE_INSTALLED = True
    print("✅ Thư viện 'surprise' đã được tìm thấy.")
except ImportError:
    SURPRISE_INSTALLED = False
    print("⚠️ Cảnh báo: Thư viện 'surprise' không được cài đặt. Các chức năng RecSys sẽ không khả dụng.")
if SURPRISE_INSTALLED:
    class SurpriseWrapper(mlflow.pyfunc.PythonModel):
        """Lớp wrapper để MLflow có thể làm việc với model Surprise."""
        def load_context(self, context):
            self.n_jobs = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
            self.model = surprise.dump.load(context.artifacts["surprise_model"])[1]
        def predict(self, context, model_input):
            user_ids = model_input['userID'].tolist()
            item_ids = model_input['itemID'].tolist()
            tasks = (delayed(self.model.predict)(uid, iid) for uid, iid in zip(user_ids, item_ids))
            predictions_obj = Parallel(n_jobs=self.n_jobs)(tasks)
            return np.array([pred.est for pred in predictions_obj])

    class RecSysModelTuner:
        """
        Class tổng quát để chạy một thử nghiệm hoàn chỉnh cho một model recommender system,
        bao gồm tuning, training, evaluation và logging vào MLflow.
        """
        def __init__(self, model_class: Any, run_name: str, model_family: str, reg_model_name: str, param_grid: Optional[Dict] = None,pip_requirements: Optional[list] = None,  # <-- THÊM DÒNG NÀY
                    conda_env: Optional[str] = None,
                    tags: Optional[Dict[str, str]] = None):
            self.model_class = model_class
            self.run_name = run_name
            self.model_family = model_family
            self.reg_model_name = reg_model_name
            self.param_grid = param_grid
            if pip_requirements and conda_env:
                raise ValueError("Chỉ có thể cung cấp 'pip_requirements' hoặc 'conda_env', không phải cả hai.")
            self.pip_requirements = pip_requirements
            self.conda_env = conda_env
            self.tags = tags if tags is not None else {}

        def _cleanup(self):
            """Ép buộc garbage collection để giải phóng RAM."""
            print("\n🧹 Running garbage collection...")
            gc.collect()
            print("👍 Memory cleanup complete.")

        def run(self, data, full_trainset, trainset, testset, input_example,**extra_params):
            """
            Thực thi toàn bộ flow cho model đã được cấu hình.
            """
            client = MlflowClient()
            with mlflow.start_run(run_name=self.run_name, nested=True):
                try:
                    print(f"\n--- Bắt đầu: {self.run_name} ---")
                    mlflow.set_tag("algorithm_family", self.model_family)
                    if extra_params:
                        mlflow.log_params(extra_params)
                    best_params = {}
                    if self.param_grid:
                        mlflow.set_tag("tuning", "GridSearchCV")
                        gs = GridSearchCV(self.model_class, self.param_grid, measures=['rmse'], cv=3, n_jobs=-1)
                        gs.fit(data)
                        best_params = gs.best_params['rmse']
                        mlflow.log_params(best_params)
                        mlflow.log_metric("best_cv_rmse", gs.best_score['rmse'])
                        print(f"  Tuning complete. Best CV RMSE: {gs.best_score['rmse']:.4f}")
                    else:
                        mlflow.set_tag("tuning", "skipped")
                        print("  No param_grid provided. Skipping tuning.")

                    final_algo = self.model_class(**best_params)
                    final_algo.fit(full_trainset)
                    
                    predictions_test = final_algo.test(testset)
                    test_rmse = accuracy.rmse(predictions_test, verbose=False)
                    test_mae = accuracy.mae(predictions_test, verbose=False)
                    
                    trainset_for_eval = trainset.build_testset()
                    predictions_train = final_algo.test(trainset_for_eval)
                    train_rmse = accuracy.rmse(predictions_train, verbose=False)
                    train_mae = accuracy.mae(predictions_train, verbose=False)

                    mlflow.log_metrics({"test_rmse": test_rmse, "test_mae": test_mae, "train_rmse": train_rmse, "train_mae": train_mae})
                    
                    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                        surprise.dump.dump(tmp.name, algo=final_algo)
                        utils_abs_path = os.path.abspath(__file__)
                        print(f"--- DEBUG: Absolute path for code_paths: {utils_abs_path}") # In ra để kiểm tra
                        
                        from mlflow.types.schema import Schema, ColSpec, TensorSpec
                        import numpy as np

                        # 1. Định nghĩa Input Schema (Khớp với file MLmodel)
                        input_schema = Schema([
                            ColSpec("string", "userID"),
                            ColSpec("long", "itemID")
                        ])

                        # 2. Định nghĩa Output Schema (Ép buộc nó phải là TENSOR)
                        # Đây là dòng vá lỗi 'float' object
                        # Nó nói: "Output là một array 1 chiều [-1] chứa các số np.float64"
                        output_schema = Schema([
                            TensorSpec(np.dtype(np.float64), [-1])
                        ])

                        # 3. Tạo Signature thủ công
                        manual_signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)
                        print("  Successfully created manual signature (forcing output as Tensor).")
                        mlflow.pyfunc.log_model(
                            name=f"{self.run_name}_model",
                            code_paths=[utils_abs_path],
                            python_model=SurpriseWrapper(),
                            artifacts={"surprise_model": tmp.name},
                            input_example=input_example,
                            signature=manual_signature,
                            registered_model_name=self.reg_model_name,
                            pip_requirements=self.pip_requirements, # <-- THÊM DÒNG NÀY
                            conda_env=self.conda_env
                        )
                    print(f"  ✅ {self.run_name} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
                    if self.tags:
                            print(f"  Setting tags for Registered Model '{self.reg_model_name}': {self.tags}")
                            for key, value in self.tags.items():
                                try:
                                    client.set_registered_model_tag(self.reg_model_name, key, value)
                                    print(f"    Set tag: {key} = {value}")
                                except Exception as tag_error:
                                    print(f"    ⚠️ Failed to set tag '{key}': {tag_error}")
                except Exception as e:
                    print(f"  ❌ An error occurred in {self.run_name}: {e}")
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                
                finally:
                    self._cleanup()

    