FROM ghcr.io/berriai/litellm:main-latest

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY custom_callbacks ./custom_callbacks
COPY config.yaml ./config.yaml

EXPOSE 4000

CMD ["--config", "/app/config.yaml", "--port", "4000", "--host", "0.0.0.0"]
