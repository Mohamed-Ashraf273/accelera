FROM python:3.11-slim
WORKDIR /app
COPY accelera_deployment/requirements.txt requirements.txt 
COPY accelera_deployment/server.py server.py
COPY accelera_deployment/modelservice.py modelservice.py
COPY config.json config.json
COPY models/scaler.pkl /app/models/scaler.pkl
COPY models/model.pkl /app/models/model.pkl
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
