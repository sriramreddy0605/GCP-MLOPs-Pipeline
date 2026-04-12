FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas scikit-learn joblib flask
COPY experiment.py .
# We will create an app.py next
COPY app.py . 
CMD ["python", "app.py"]
