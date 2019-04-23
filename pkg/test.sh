#!/usr/bin/env bash
set -e

TEST_DATA_DIR=$1
VAL_OUTPUT_NAME=$2
TEST_OUTPUT_NAME=$3

python /app/inference.py "$TEST_DATA_DIR/val_article.txt" "$TEST_DATA_DIR/$VAL_OUTPUT_NAME"
python /app/inference.py "$TEST_DATA_DIR/test_article.txt" "$TEST_DATA_DIR/$TEST_OUTPUT_NAME"

# A sample call to your testing script (single line):
# ./test.sh /data/test/ val_summary.txt test_summary.txt
# In this case you can assume that the testing data looks like this:
#  data/
#    test/
#      test_article.txt

#      val_article.txt
