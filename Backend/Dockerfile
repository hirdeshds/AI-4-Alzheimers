FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pillow \
    python-dotenv \
    requests \
    python-multipart

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
