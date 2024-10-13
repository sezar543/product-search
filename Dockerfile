FROM python:3.10.7

COPY . .

RUN pip3 install -r requirements.txt


CMD ["fastapi", "run", "api.py"]