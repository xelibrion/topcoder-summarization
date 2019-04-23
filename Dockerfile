FROM python:3.7-slim

WORKDIR /app

ADD pkg/ /
ADD src /app
ADD requirements.txt /

RUN pip install -r /requirements.txt

CMD /test.sh /data/test/ val_summary.txt test_summary.txt
