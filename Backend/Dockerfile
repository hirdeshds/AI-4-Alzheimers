FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir fastapi uvicorn pillow python-dotenv requests

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
