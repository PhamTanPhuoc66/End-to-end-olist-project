from __future__ import annotations
import pendulum
from airflow.decorators import dag
from airflow.models.param import Param
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule 
# --- CONFIG: ĐỊNH NGHĨA CÁC MODEL VÀ FILE SCRIPT TƯƠNG ỨNG ---
# Đây là "thực đơn" các model bạn muốn chạy.
# Key: Tên định danh (sẽ dùng cho Task ID).
# Value: Tên file .py chính xác cần thực thi.
MODELS_TO_RUN = {
    'churn_multi_models': 'churn_pred_multi_models.py',
    'churn_models': 'churn_pred_models.py',
    'collaborative_filtering_models': 'colaborative_filtering.py',
    'content_based_filtering_models': 'content_based_filtering.py',

}


@dag(
    dag_id='trigger_multi_databricks_and_docker',
    start_date=pendulum.datetime(2025, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    tags=['databricks', 'docker', 'ml-pipeline'],
    params={
        "dev_container_name": Param(
            "interesting_ride",
            type="string", 
            title="Standalone Dev Container Name"
        ),
        # Đường dẫn đến thư mục chứa tất cả các script train model
        "scripts_directory_path": Param(
            "/workspaces/End_to_end_ML_DL_project/ml", # <-- THAY BẰNG ĐƯỜNG DẪN THỰC TẾ TRONG CONTAINER
            type="string",
            title="Path to Scripts Directory inside Container"
        )
    },
    doc_md="""
    ### ML Pipeline: Trigger Databricks và Chạy Nhiều Model từ các file riêng biệt
    1. Kích hoạt 2 Job Databricks.
    2. Khởi động dev container.
    3. **Kích hoạt nhiều task huấn luyện model song song, mỗi task chạy một file Python riêng.**
    """
)
def multi_platform_trigger_dag():
    
    # --- Task 1 & 2: Kích hoạt jobs Databricks ---
    trigger_databricks_job_1 = DatabricksRunNowOperator(
        task_id='trigger_databricks_pipeline',
        databricks_conn_id='databricks_default',
        job_id=362052632397048,
    )
    
    trigger_databricks_job_2 = DatabricksRunNowOperator(
        task_id='trigger_databricks_dbt_marts',
        databricks_conn_id='databricks_default',
        job_id=11892938834751,
    )

    # --- Task 3: Khởi động container ---
    start_dev_container = BashOperator(
        task_id='start_dev_container',
        bash_command='docker start {{ params.dev_container_name }}'
    )

    # Task rỗng để làm điểm bắt đầu cho các nhánh chạy model
    training_kickoff = EmptyOperator(task_id="training_kickoff")

    # --- Task 4 (ĐỘNG): Vòng lặp tạo các task chạy model từ dictionary ---
    # for model_id, script_name in MODELS_TO_RUN.items():
    #     run_model_task = BashOperator(
    #         # Task ID sẽ là: 'run_model_churn_multi_models', 'run_model_collaborative_filtering', ...
    #         task_id=f'train_{model_id}',
            
    #         # Xây dựng câu lệnh docker exec để chạy file script tương ứng
    #         bash_command=(
    #             "docker exec {{ params.dev_container_name }} "
    #             "python {{ params.scripts_directory_path }}/"
    #             f"{script_name}"
    #         )
    #     )
        
    #     training_kickoff >> run_model_task
    previous_task = training_kickoff 
    model_tasks = list(MODELS_TO_RUN.items())

    for i, (model_id, script_name) in enumerate(model_tasks):
        # Dùng if/else để tạo command chính xác cho từng trường hợp
        if model_id == 'churn_models':
            # # Lệnh ĐÚNG cho 'churn_models' dùng -m và -w
            # bash_command = (
            #     "docker exec -w {{ params.scripts_directory_path }} "
            #     "{{ params.dev_container_name }} "
            #     f"python -m cuml.accel {script_name}"
            # )
            bash_command = (
                "docker exec {{ params.dev_container_name }} "
                "python {{ params.scripts_directory_path }}/"
                f"{script_name}"
            )
        else:
            # Lệnh ĐÚNG cho các trường hợp còn lại
            bash_command = (
                "docker exec {{ params.dev_container_name }} "
                "python {{ params.scripts_directory_path }}/"
                f"{script_name}"
            )

        run_model_task = BashOperator(
            task_id=f'train_{model_id}',
            bash_command=bash_command, # Sử dụng command đã được tạo đúng
            trigger_rule=TriggerRule.ALL_DONE 
        )
        
        # Nối task hiện tại vào task trước đó
        previous_task >> run_model_task
        
        # Cập nhật task hiện tại thành "task trước đó" cho vòng lặp tiếp theo
        previous_task = run_model_task

    # --- Định nghĩa luồng chính của DAG ---
    trigger_databricks_job_1 >> [trigger_databricks_job_2 , start_dev_container >> training_kickoff]

multi_platform_trigger_dag()