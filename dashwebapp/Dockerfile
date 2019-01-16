FROM ubuntu:latest
MAINTAINER anyuzhang2018
RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt


WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install --no-deps pandas==0.23.0

COPY . /app

ENV APP_ENV development
EXPOSE 5035
#wget https://s3-us-west-2.amazonaws.com/jenkinsiamtest1/model.py
#wget https://s3-us-west-2.amazonaws.com/housepricetest/properties_2016.csv
#wget https://s3-us-west-2.amazonaws.com/housepricetest/train_2016_v2.csv
#wget https://s3-us-west-2.amazonaws.com/housepricetest/sample_submission.csv


CMD [ "python" ,"model.py" ]
CMD [ "python" ,"webdash.py" ]
