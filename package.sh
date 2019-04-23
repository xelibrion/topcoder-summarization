#!/usr/bin/env bash
set -e

docker build . -t topcoder-summarization
docker run -it --rm -v $PWD/data:/data topcoder-summarization
