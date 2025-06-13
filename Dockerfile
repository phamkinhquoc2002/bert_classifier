FROM python:3.11-slim

COPY ./ app

WORKDIR /app

ARG DVC_REMOTE_URI
ENV DVC_REMOTE_URI=${DVC_REMOTE_URI}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir dvc[gdrive]

RUN dvc init 
RUN dvc remote add -d storage gdrive://${DVC_REMOTE_URI}
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path /run/secrets/gdrive_creds.json

RUN cat .dvc/config
RUN dvc pull models/last.ckpt.dvc

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
