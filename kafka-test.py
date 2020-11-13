from __future__ import print_function 
from pyspark import SparkContext
import pyspark
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml.classification import LogisticRegression, NaiveBayes, NaiveBayesModel, GBTClassifier, RandomForestClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Word2Vec, IDF, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import PipelineModel
from pyspark.sql.types import (StructType, StringType, StructField, LongType)
from pyspark.sql import Row
import sys
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
import time

## Kafka VM IP
kafka_topic = 'from-pubsub'
zk = '10.182.0.2:2181'
app_name = "from-pubsub"
sc = SparkContext(appName="KafkaPubsub")
ssc = StreamingContext(sc, 0.1)
kafkaStream = KafkaUtils.createStream(ssc, zk, app_name, {kafka_topic: 1})
### Load Saved Model
pipelineFit = PipelineModel.load('gs://wcs_word/NB_pipeline')
print("1")
model = NaiveBayesModel.load('gs://wcs_word/NB_FullTrainedModel')
print("2")
spark = SparkSession(sc)

## Global Variable to calculate Latency
init_time = None
count = 0

## Parsing Function
def row_generate(r):
    return Row(star = float(r[0]) -1, useful = float(r[1]), funny = float(r[2]), cool = float(r[3]), text = str(r[4]))

## Parsing Function
def lambda_func(r):
    return r[25:-2].split(',')

def func(ks):
    ### Loop to ensure the job doesn't finish
    try:
        global init_time, count
        dat = ks.map(lambda r:r[1])
        dat1 = dat.map(lambda_func)
        # print("Start")
        for r in dat1.collect():
            count += 1
        # print("End")
        dat2 = dat1.map(row_generate)
        dat3 = spark.createDataFrame(dat2)
        dat3.show()
        dat_transform = pipelineFit.transform(dat3)
        pred = model.transform(dat_transform)
        pred.select("Prediction").show()
        print("recieved rdd")
        ### Latency Calc Part
        if(init_time == None):
            init_time = time.time()
        elif(time.time() - init_time > 9.9):
            time_elapsed = time.time() - init_time
            print("Messages Recieved: ", count)
            print("Time Elapsed: ", time_elapsed)
            print("Latency: ", time_elapsed/count)
            count = 0
            init_time = time.time()
    except:
        pass

def empty_rdd():
    print("The current RDD is empty. Wait for the next complete RDD")
print("3")
kafkaStream.foreachRDD(lambda rdd: func(rdd))
print("4")

ssc.start() # Start the computation
ssc.awaitTermination()