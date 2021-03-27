FROM python:3.7
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN mkdir app
WORKDIR app
COPY . .
RUN python -m pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]
