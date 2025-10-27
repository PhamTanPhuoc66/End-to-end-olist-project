# Bắt đầu từ image Airflow 2.9.1 chính thức
FROM apache/airflow:2.9.1

# --- PHẦN 1: CÀI ĐẶT DOCKER CLI (BẮT BUỘC) ---
# Phải chuyển sang user root để có quyền cài đặt các gói hệ thống
USER root

# Cài các gói cần thiết để thêm repository của Docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Thêm GPG key chính thức của Docker
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Thêm repository của Docker
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Cài đặt Docker CLI (chỉ cần client, không cần engine)
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*
# --- KẾT THÚC PHẦN 1 ---


# --- PHẦN 2: CẤP QUYỀN TRUY CẬP DOCKER.SOCK (BẮT BUỘC) ---
# Thêm user 'airflow' vào group 'docker' (với GID là 125).
# Group này sẽ có quyền truy cập vào file docker.sock được mount từ máy host.
RUN groupadd --force -g 125 docker && usermod -aG docker airflow
# --- KẾT THÚC PHẦN 2 ---


# --- PHẦN 3: CÀI ĐẶT CÁC GÓI PYTHON CỦA BẠN ---
# Bây giờ mới chuyển về user airflow để thực hiện các thao tác không cần quyền root
USER airflow

# Nâng cấp pip và các công cụ build
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy file constraints vào image
COPY --chown=airflow:airflow constraints-3.12.txt /constraints.txt

# Cài đặt các gói Python cần thiết cho project
RUN pip install --no-cache-dir \
    --constraint /constraints.txt \
    boto3 \
    psycopg2-binary \
    mlflow\
    apache-airflow-providers-databricks\
    apache-airflow-providers-docker
# --- KẾT THÚC PHẦN 3 ---