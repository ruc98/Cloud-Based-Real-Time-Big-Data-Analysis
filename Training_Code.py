from __future__ import print_function 
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, Word2Vec, IDF, Tokenizer, StopWordsRemover, CountVectorizer, Normalizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.ml.feature import SQLTransformer, HashingTF
import pyspark
import sys

# This is to check whether we provided the output path to save the model
if len(sys.argv) != 2:
  raise Exception("Exactly 1 arguments are required: <inputUri> <outputUri>")

outputUri=sys.argv[1]


# Start the Spark Session
sc = SparkContext()
spark = SparkSession(sc)

########################################################### 1. Data Loading #####################################################################################
# Bucket path
filepath = "gs://ae16b005_1/yelp.json"

# Loading the json file directly from path , other option can be load the json file in bigquery and then load the table
# But loading directly from the bucket is faster than the loading from the bigquery table
df = spark.read.json(filepath)

# Visualise the data size and schema of the dataset
print((df.count(), len(df.columns)))

df.printSchema()

# Print top 10 rows in the dataset
df.show()

########################################################### 2. Pipeline ###########################################################################################

# Do note that we saved the pipeline for this using different python file
print("**********************************************************************************************")
print("Pipeline is Started")
print("**********************************************************************************************")

# Tokenisation -  Split the text into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Stopword Removal - Remove the high frequency words, as they wont be useful
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

# Here we can use HashingTF or Count Vectorizer for term frequency. Its a tradeoff between faster processing Vs Higher Accuracy
#hashtf = HashingTF(inputCol="filtered", outputCol="hash_tf")
hashtf = CountVectorizer(inputCol="filtered", outputCol="hash_tf")

# IDF - Inverse Document Frequency
idf = IDF(inputCol="hash_tf", outputCol="tfidf")

# Normalizer - Here this is very important to get accuracy (Jumped around 0.58 to 0.631)
# This is because of the pyspark implementation is different real tfidf, where notm is missing in the pyspark idf implementation
norm = Normalizer(inputCol = 'tfidf', outputCol = 'norm_tfidf')

# Renaming all predictors to feature
clean_data = VectorAssembler(inputCols = ['norm_tfidf', 'funny'], outputCol = 'features')

# Transforing the data using Tokenizer (No fitting because nothing to train)
token_df = tokenizer.transform(df)

# Transforming the data using Stopword Removal (No fitting because nothing to train)
stop_df = stopwordsRemover.transform(token_df)

# If you apply Count Vectorizer, here you will have to fit and then transform
# If you apply hashingtf, here you can use direct transformation on the data
#tf_df = hashtf.transform(stop_df)
tf_df = hashtf.fit(stop_df).transform(stop_df)

# Fit and Transforing the data using Tokenizer IDF
idf_df = idf.fit(tf_df).transform(tf_df)

# This is very important, L2 Normalisation is missing in pyspark idf implementation. We explicitly build this function
norm_df = norm.transform(idf_df)

# Clean data into features, this has to be done before you send the data for model training
raw_dataset = clean_data.transform(norm_df)

# This step will have a drastic change in training for naive bayes and logistic regression.
dataset = raw_dataset.withColumn("label", raw_dataset["stars"]-1)

# Have a look at the preprocessed data
print((dataset.count(), len(dataset.columns)))

dataset.printSchema()

dataset.show()


# Split the dataset into Training and Testing using random split
(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 100)


print("**********************************************************************************************")
print("Pipeline is done")
print("**********************************************************************************************")

########################################################### 3. Model Training #####################################################################################

print("**********************************************************************************************")
print("Training Started")
print("**********************************************************************************************")

# We tried Both model but Naive Bayes gave best
nb = NaiveBayes(labelCol = 'label')
#lr = LogisticRegression(featuresCol="features", labelCol = 'label')

# Fit the model
model = nb.fit(trainingData)
#model = lr.fit(trainingData)
'''
trainingSummary = lr.summary

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
'''


print("**********************************************************************************************")
print("Training Done")
print("**********************************************************************************************")

########################################################### 3. Model Evaluation #####################################################################################

# Training Predictions
predictions_train = model.transform(trainingData)

# Test Predictions
predictions_test = model.transform(testData)

print("**********************************************************************************************")
print("Predictions on Test data is done, printing accuracy is remaining")
print("**********************************************************************************************")

# Training and Test Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_train = evaluator.evaluate(predictions_train)
accuracy_test = evaluator.evaluate(predictions_test)


print("**********************************************************************************************")
print("Train accuracy without cross validation = %g" % (accuracy_train))
print("Test accuracy without cross validation = %g" % (accuracy_test))
print("**********************************************************************************************")

########################################################### 4. Save the Model #####################################################################################
## Save Model
print("**********************************************************************************************")
print("Model is saving")
print("**********************************************************************************************")

model.save(sys.argv[1] + '/Best_FullTrainedModel')

print("**********************************************************************************************")
print("Model Saved")
print("**********************************************************************************************")

