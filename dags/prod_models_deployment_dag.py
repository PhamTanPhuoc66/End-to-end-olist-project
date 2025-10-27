import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.providers.docker.hooks.docker import DockerHook
from airflow.exceptions import AirflowSkipException
import json

# --- Cấu hình ---
# Tên repo trên Docker Hub (format: ten_dang_nhap/ten_repo)
DOCKER_HUB_REPO = "phamtanphuoc/champion-models"
# Connection ID cho MLflow
MLFLOW_CONN_ID = "mlflow_default"
# THAY ĐỔI: Airflow Variable để lưu trạng thái deploy của TẤT CẢ các model
LAST_DEPLOYED_VAR = "last_deployed_production_models_versions"


@dag(
    dag_id="mlflow_production_model_to_dockerhub_auto_discovery",
    start_date=pendulum.datetime(2025, 10, 17, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    doc_md="""
    ### MLflow Production Model Deployment DAG (Multi-Model Parallel)

    DAG này tự động quét **tất cả** các Registered Models trong MLflow.
    Nó so sánh phiên bản 'Production' của **từng model** với trạng thái đã deploy lần cuối.
    Với mỗi model có phiên bản mới, nó sẽ **chạy song song** các bước sau:
    1. Build một Docker image từ model URI.
    2. Push image đó lên Docker Hub.
    3. Cập nhật Airflow Variable với định danh của các model vừa được deploy thành công.
    """,
    tags=["mlops", "mlflow", "docker", "production-stage", "auto-discovery", "parallel"],
)
def mlflow_production_model_to_dockerhub_auto_discovery():
    """
    DAG để tự động deploy song song các model mới nhất có stage 'Production'.
    """

    @task
    def check_for_new_production_models() -> list[dict]:
        """
        Quét tất cả model, so sánh từng model với trạng thái đã lưu,
        và trả về một danh sách các model cần deploy.
        """
        from mlflow.tracking import MlflowClient
        from airflow.hooks.base import BaseHook

        print("Đang kết nối tới MLflow...")
        mlflow_conn = BaseHook.get_connection(MLFLOW_CONN_ID)
        tracking_uri = f"http://{mlflow_conn.host}:{mlflow_conn.port}"
        print(f"Connecting to MLflow tracking URI: {tracking_uri}")
        client = MlflowClient(tracking_uri=tracking_uri)
        
        try:
            last_deployed_str = Variable.get(LAST_DEPLOYED_VAR, default_var='{}')
            last_deployed_versions = json.loads(last_deployed_str)
            if not isinstance(last_deployed_versions, dict):
                print(f"Cảnh báo: Variable '{LAST_DEPLOYED_VAR}' không phải là JSON hợp lệ. Sử dụng dictionary rỗng.")
                last_deployed_versions = {}
        except (json.JSONDecodeError, TypeError):
            print(f"Cảnh báo: Không thể parse JSON từ Variable '{LAST_DEPLOYED_VAR}'. Sử dụng dictionary rỗng.")
            last_deployed_versions = {}
        
        print(f"Trạng thái các model đã deploy lần cuối: {last_deployed_versions}")

        models_to_deploy = []
        print("Đang quét TẤT CẢ registered models trong MLflow...")
        all_registered_models = client.search_registered_models()

        # === THÊM LOGGING CHI TIẾT ===
        print(f"Tìm thấy tổng cộng {len(all_registered_models)} registered models.")

        for registered_model in all_registered_models:
            model_name = registered_model.name
            print(f"--- Đang kiểm tra model: '{model_name}' ---")
            try:
                latest_prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
                
                if latest_prod_versions:
                    print(f"  + Tìm thấy phiên bản Production cho '{model_name}'.")
                    latest_prod_model = latest_prod_versions[0]
                    current_prod_version = int(latest_prod_model.version)
                    
                    last_deployed_version = int(last_deployed_versions.get(model_name, 0))

                    if current_prod_version > last_deployed_version:
                        print(f"  -> PHÁT HIỆN MỚI cho model '{model_name}': Production v{current_prod_version} > Deployed v{last_deployed_version}")
                        # Xác định env_manager dựa trên tag
                        env_manager = "local" # Mặc định
                        model_tags = registered_model.tags
                        if model_tags.get("build_env") == "needs_gcc":
                            env_manager = "conda"
                            print(f"  -> Model '{model_name}' có tag 'needs_gcc'. Dùng env_manager=conda.")
                        else:
                            print(f"  -> Model '{model_name}' không có tag 'needs_gcc'. Dùng env_manager=local.")
                        # -----------------
                        models_to_deploy.append({
                            "model_uri": latest_prod_model.source,
                            "version": current_prod_version,
                            "name": model_name,
                            "env_manager": env_manager,
                           
                        })
                    else:
                        print(f"  -> Model '{model_name}' đã được deploy phiên bản mới nhất (v{current_prod_version}).")
                else:
                    print(f"  - Không tìm thấy phiên bản Production nào cho model '{model_name}'.")

            except Exception as e:
                print(f"Lỗi khi truy vấn model '{model_name}': {e}. Bỏ qua model này.")
                continue

        if not models_to_deploy:
            print("Không có model nào cần deploy trong lần chạy này. Bỏ qua.")
            raise AirflowSkipException
        
        print(f"Tổng cộng {len(models_to_deploy)} model sẽ được deploy.")
        return models_to_deploy

    @task
    def build_push_and_cleanup_image(model_info: dict) -> dict:
        import subprocess
        import os
        from airflow.hooks.base import BaseHook
        # === THÊM CÁC IMPORT CẦN THIẾT ===
        import mlflow
        import tempfile
        import shutil
        # =================================
        env_manager = model_info.get("env_manager", "local")
        
        version = model_info["version"]
        name = model_info["name"]
        model_uri = model_info["model_uri"]
        safe_name_tag = name.replace(" ", "-").lower()
        image_name = f"{DOCKER_HUB_REPO}:{safe_name_tag}-v{version}"
        latest_tag = f"{DOCKER_HUB_REPO}:{safe_name_tag}-latest"
        
        print(f"Bắt đầu xử lý model: '{name}-v{version}'")
        print(f"Sử dụng DockerHook để đăng nhập vào Docker Hub...")
        
        docker_hook = DockerHook(
            docker_conn_id='docker_hub_default', 
            base_url='unix://var/run/docker.sock'
        )
        docker_conn = docker_hook.get_conn()
        
        print("Đang lấy thông tin MLflow server từ Airflow Connection...")
        mlflow_conn = BaseHook.get_connection(MLFLOW_CONN_ID)
        mlflow_tracking_uri = f"http://{mlflow_conn.host}:{mlflow_conn.port}"
        print(f"MLflow Tracking URI được sử dụng: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Tạo một thư mục tạm để chứa model tải về
        tmp_dir = tempfile.mkdtemp()
        print(f"Tạo thư mục tạm tại: {tmp_dir}")

        def run_command(command, env=None):
            print(f"  Đang chạy lệnh: {' '.join(command)}")
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=process_env
            )
            for line in iter(process.stdout.readline, ''):
                print(f"    {line.strip()}")
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)

        try:
            # === BƯỚC 1: TẢI MODEL VỀ THƯ MỤC TẠM ===
            print(f"Đang tải artifacts từ URI: {model_uri}...")
            local_model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=tmp_dir
            )
            print(f"Tải model thành công về đường dẫn cục bộ: {local_model_path}")
            # ========================================

            print(f"Model '{name}' sẽ được build với --env-manager {env_manager}")
            # Dòng lệnh build mới, đã sửa lỗi
            build_command = [
                "mlflow", "models", "build-docker",
                "-m", local_model_path,
                "-n", image_name,
                "--enable-mlserver",
                "--env-manager", env_manager
            ]
                
            # --------------------------------------------------------
            mlflow_env = {"MLFLOW_TRACKING_URI": mlflow_tracking_uri}
            run_command(build_command, env=mlflow_env)
            # =============================================

            tag_command = ["docker", "tag", image_name, latest_tag]
            run_command(tag_command)
            
            print(f"  Đang push image: {image_name}")
            docker_conn.push(image_name, decode=True)
            print(f"  Đang push image: {latest_tag}")
            docker_conn.push(latest_tag, decode=True)
        finally:
            print("  Dọn dẹp các image trên local worker...")
            try:
                run_command(["docker", "rmi", image_name])
                run_command(["docker", "rmi", latest_tag])
                print("  Dọn dẹp image thành công.")
            except subprocess.CalledProcessError as e:
                print(f"  Lỗi khi dọn dẹp image, có thể nó không tồn tại: {e}")
            
            # === DỌN DẸP THƯ MỤC TẠM ===
            print(f"  Dọn dẹp thư mục tạm: {tmp_dir}")
            shutil.rmtree(tmp_dir)
            print("  Dọn dẹp thư mục tạm thành công.")
            # ============================

        print(f"Hoàn tất xử lý model '{name}-v{version}'!")
        return {"name": name, "version": version}

    @task
    def update_last_deployed_versions(newly_deployed_models: list):
        # ... (Nội dung task này không thay đổi)
        if not newly_deployed_models:
            print("Không có model nào được deploy thành công để cập nhật.")
            return

        print(f"Bắt đầu cập nhật trạng thái cho {len(newly_deployed_models)} model đã deploy...")
        try:
            last_deployed_str = Variable.get(LAST_DEPLOYED_VAR, default_var='{}')
            last_deployed_versions = json.loads(last_deployed_str)
            if not isinstance(last_deployed_versions, dict):
                last_deployed_versions = {}
        except (json.JSONDecodeError, TypeError):
            last_deployed_versions = {}

        for model in newly_deployed_models:
            if model and 'name' in model and 'version' in model:
                model_name = model['name']
                new_version = model['version']
                print(f"  -> Cập nhật model '{model_name}' thành phiên bản {new_version}")
                last_deployed_versions[model_name] = new_version

        Variable.set(LAST_DEPLOYED_VAR, last_deployed_versions, serialize_json=True)
        print("Cập nhật trạng thái deploy thành công.")

    # Luồng thực thi của DAG
    models_to_deploy = check_for_new_production_models()
    deployed_models_info = build_push_and_cleanup_image.expand(model_info=models_to_deploy)
    update_last_deployed_versions(newly_deployed_models=deployed_models_info)

# Khởi tạo DAG
mlflow_production_model_to_dockerhub_auto_discovery()

