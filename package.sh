#!/usr/bin/env bash
set -e

docker build . -t topcoder-summarization
docker run -it --rm -v $PWD/data:/data topcoder-summarization

mkdir -p .dist/solution
mkdir -p .dist/code

mv data/test/*_summary.txt .dist/solution/

cp Dockerfile .dist/code/
cp -r src .dist/code/

pushd .dist
zip -r submission.zip solution
zip -r submission.zip code
popd

mv .dist/submission.zip .
