import numpy as np
import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, regexTokenizer
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import lit
from sklearn.linear_model import SGDClassifier
from pyspark.sql.functions import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pickle
from sklearn import model_selection
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler, 
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, regexTokenizer
import sklearn.linear_model as lm
from pyspark.sql import Row, Column
import sys

sc = SparkContext("local[2]", "PESU")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)



def converttojson(data):
	json=json.loads(data)
	list=list()
	for i in json:
		rows=tuple(json[i].values())
		list.append(rows)
	return list 	

def converttodf(data):
	global model, model_lm, model_sgd, model_mlp, output2,output3, output1, x, y
	if data.isEmpty():
		return

	ss=SparkSession(data.context)
	data=data.collect()[0]
	
	col=[f"feature{i}" for i in range(len(data[0]))]
	try:
		df=ss.createDataFrame(data,col)
	except:
		return
	
	datasettemp=[('ham','ham','ham')]
	rowData=ss.createDataFrame(datasettemp, col)
	df=df.union(rowData)
	
		
	
	print('\nStages-----------------------------\n')
	new_df=df

	regEx = regexTokenizer(inputCol= 'feature1' , outputCol= 'tokens', pattern= '\\W')

	print("regEx completed")

	
	sw_remover = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	print("Stopwords completed")

	stage_1 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	
	
	print("Word2vec completed")

	
	st_indexing = StringIndexer(inputCol="feature2", outputCol="categoryIndex", stringOrderType='alphabetAsc')

	print("TargetCol completed")
	
	pipeline=Pipeline(stages=[regEx, sw_remover, stage_1, st_indexing])
	pipelineFit=pipeline.fit(df)
	dataset=pipelineFit.transform(df)

	dataset=dataset.filter(dataset.feature1!='ham')
	new_df=dataset.select(['vector'])
	new_df_target=dataset.select(['categoryIndex'])


	x=np.array(new_df.select('vector').collect())
	y=np.array(new_df_target.select('categoryIndex').collect())
	
	x = [np.concatenate(i) for i in x]

	
	model_lm=lm.LogisticRegression(warm_start=True)
	model_lm=model_lm.fit(x,y.ravel())
	output1=model_lm.score(x, y)
	print("Logistic Regression accuracy: ", output1)

	
	model_sgd=SGDClassifier(alpha=0.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
	
	model_sgd.partial_fit(x,y.ravel(), classes=[0.0,1.0])
	
	
	output2=model_sgd.score(x, y)
	print("SGD accuacy: ",output2)
	model_mlp=MLPClassifier(random_state=1, max_iter=300)
	model_mlp.partial_fit(x,y.ravel(), classes=[0.0,1.0])
	output3=model_mlp.score(x, y)
	print("MLP accuacy: ",output3)	
	
	
	


lines = ssc.socketTextStream("localhost",6100).map(converttojson).foreachRDD(converttodf)

ssc.start() 
ssc.awaitTermination(200)
ssc.stop()

filename='lm_1000.sav'
pickle.dump(model_lm, open(filename, 'wb'))
print("LM Model saved")

filename='sgd_1000.sav'
pickle.dump(model_sgd, open(filename, 'wb'))
print("SGD Model saved")

filename='mlp_1000.sav'
pickle.dump(model_mlp, open(filename, 'wb'))
print("MLP Model saved")

results=[output1, output2, output3]
names=['logistic', 'SGD', 'MLP']

plt.bar(names, results)
plt.show()
