FROM python:3.7-slim

WORKDIR /app

COPY en_core_web_lg-2.1.0.tar.gz /
RUN pip install /en_core_web_lg-2.1.0.tar.gz

COPY requirements.txt /
RUN pip install -r /requirements.txt

ADD pkg/ /
ADD src /app

CMD /test.sh /data/test/ val_summary.txt test_summary.txt
