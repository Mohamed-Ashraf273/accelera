FROM python:3.14-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libllvm18 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY accelera_deployment/requirements.txt requirements.txt 
RUN python -m pip install --no-cache-dir --prefer-binary --ignore-requires-python --root-user-action=ignore --timeout 120 --retries 10 -r requirements.txt
COPY accelera_deployment/server.py server.py
COPY accelera_deployment/modelservice.py modelservice.py
COPY accelera_deployment/schema_validation.py schema_validation.py
COPY accelera_deployment/tracking.py tracking.py
COPY accelera_deployment/accelera/ /app/accelera/
COPY accelera_deployment/*.so /app/
ENV LD_LIBRARY_PATH=/app
COPY config.json config.json
COPY models/pipeline.pkl /app/models/pipeline.pkl
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
