FROM python:3.11-slim
WORKDIR /app
COPY accelera_deployment/requirements.txt requirements.txt 
RUN python -m pip install --no-cache-dir --prefer-binary --timeout 120 --retries 10 -r requirements.txt
COPY accelera_deployment/server.py server.py
COPY accelera_deployment/modelservice.py modelservice.py
COPY accelera_deployment/schema_validation.py schema_validation.py
COPY accelera_deployment/tracking.py tracking.py
COPY config.json config.json
COPY models/preprocessing_pipeline.pkl /app/models/preprocessing_pipeline.pkl
COPY models/feature_selector.pkl /app/models/feature_selector.pkl
COPY models/final_model.pkl /app/models/final_model.pkl
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
