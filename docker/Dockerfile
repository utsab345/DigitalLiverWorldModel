FROM python:3.11-slim
WORKDIR /app
COPY digital_liver_world_model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn celery redis prometheus-client
COPY digital_liver_world_model /app/digital_liver_world_model
COPY api /app/api
COPY worker /app/worker
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
