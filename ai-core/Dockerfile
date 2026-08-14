FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ĐỔI TÊN THƯ MỤC, TUYỆT ĐỐI KHÔNG DÙNG /var/azureml-app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/inference ./

ENV AZUREML_ENTRY_SCRIPT=online_score.py

EXPOSE 31311

CMD ["azmlinfsrv", "--entry_script", "online_score.py", "--port", "31311"]