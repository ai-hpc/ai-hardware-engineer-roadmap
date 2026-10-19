FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY service.py model.json /app/
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=1s --timeout=2s --start-period=1s --retries=2 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=1).read()"]
ENTRYPOINT ["python", "service.py", "--host", "0.0.0.0"]
