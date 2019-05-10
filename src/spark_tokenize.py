#!/usr/bin/env python

# NOTE: Java8 required by Spark
# export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)

import os
import argparse
from pyspark.sql import SparkSession, Column
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.column import _to_seq, _to_java_column


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def ssplit_udf(col, spark):
    udf = spark._sc._jvm.com.databricks.spark.corenlp.functions.ssplit()
    return Column(udf.apply(_to_seq(spark._sc, [col], _to_java_column)))


def tokenize_udf(col, spark):
    udf = spark._sc._jvm.com.databricks.spark.corenlp.functions.tokenize()
    return Column(udf.apply(_to_seq(spark._sc, [col], _to_java_column)))


def abstract_to_sentences(abstract_series):
    def map_op(abstract):
        abstract_sents = abstract.split('</s>')
        abstract_sents = [x.replace('<s>', '').strip() for x in abstract_sents if x != '']
        return abstract_sents

    return abstract_series.apply(map_op)


def tokenize_sentence_list(df, column_name, spark):
    df_tmp = df.select(
        'article_id',
        F.explode(col(column_name)).alias('single_sentence'),
        F.monotonically_increasing_id().alias('sentence_seq'),
    )

    windowSpec = Window.partitionBy('article_id').orderBy('sentence_seq')
    df_tmp = df_tmp.select(
        'article_id',
        'sentence_seq',
        tokenize_udf(col('single_sentence'), spark).alias('sentence_tokens'),
    )
    df_result = df_tmp.select(
        'article_id',
        'sentence_seq',
        F.collect_list('sentence_tokens').over(windowSpec).alias('column_tokens'),
    )
    return df_result.groupby('article_id') \
                    .agg(F.last('column_tokens').alias('column_tokens'))


def run_task(standford_models_jar):
    num_executors = os.cpu_count()

    spark = SparkSession.builder \
                        .appName('spark_tokenize') \
                        .master(f'local[{num_executors}]') \
                        .config('spark.jars', standford_models_jar) \
                        .config('spark.jars.packages', 'databricks:spark-corenlp:0.4.0-spark2.4-scala2.11') \
                        .config('spark.driver.memory', '10G') \
                        .getOrCreate()

    df = spark.read.json(rel_path('../data/unpacked.jsonl'))

    abstract_to_sentences_udf = F.pandas_udf(abstract_to_sentences, 'array<string>')

    df = df.repartition(50) \
           .withColumn('abstract_sentences', abstract_to_sentences_udf(col('abstract'))) \
           .withColumn('article_sentences', ssplit_udf(col('article'), spark)) \
           .cache()

    df_abstracts = tokenize_sentence_list(df, 'abstract_sentences', spark) \
        .withColumnRenamed('column_tokens', 'abstract_tokens').cache()

    df_articles = tokenize_sentence_list(df, 'article_sentences', spark) \
        .withColumnRenamed('column_tokens', 'article_tokens').cache()

    df = df_abstracts.join(df_articles, on=['article_id'], how='inner') \
                     .orderBy('article_id')
    print(df.columns)

    df.repartition(1).write.json(rel_path('../data/tokenized'))


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'stanford_nlp_dir',
        type=str,
        help="",
    )
    args = parser.parse_args()
    stanford_files = os.listdir(os.path.expanduser(args.stanford_nlp_dir))
    standford_models_jar = [x for x in stanford_files if x.endswith('models.jar')][0]
    run_task(standford_models_jar)


if __name__ == '__main__':
    main()
