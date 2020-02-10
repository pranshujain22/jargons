# set the base image 
FROM python:3.6
MAINTAINER pranshu.jain

RUN mkdir /usr/src/app
ADD Jargons/ /usr/src/app

WORKDIR /usr/src/app
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD exec gunicorn Jargons.wsgi:application --bind 0.0.0.0:8000 --workers 3
