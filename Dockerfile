# Dockerfile for api_sklearn
# FROM python:3.7

# EXPOSE 8000

# WORKDIR /app
# COPY ./requirements.txt /app
# RUN pip install -r requirements.txt

# CMD ["hug", "-f", "api_temp.py"]

FROM tensorflow/tensorflow:2.6.0

EXPOSE 8000

WORKDIR /app
COPY ./requirements.txt /app
RUN apt update
RUN apt install -y libsm6 libxrender1 libxext-dev
RUN pip install -r requirements.txt

CMD ["hug", "-f", "api_image1.py"]