FROM python:3.10.6-slim

WORKDIR /ml-rest-api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_driver.py .
COPY serialized_files/model_heuristic.pkl.pkl .
COPY serialized_files/model_RFLR.pkl.pkl .
COPY serialized_files/model_NN.pkl .

CMD ["python", "api_driver.py"]