FROM FROM python:3.7
RUN pip install --upgrade pip
RUN mkdir app
WORKDIR app
COPY . .
RUN python -m pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]
