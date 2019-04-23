FROM python:3.7-slim

ADD pkg/ /

CMD /test.sh
