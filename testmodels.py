# importing required libraries
# higher accuracy for larger batches (500 and 700 performed better than 100)
#false negatives are low - not predicting ham's as spam which is good!

import numpy as np
import sys, pyspark, json
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, regExTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StringIndexer,VectorAssembler #OneHotEncoderEstimator, 
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, regExTokenizer
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.sql import SparkSession,Row,Column
from pyspark.sql.functions import lit
from sklearn.linear_model import SGDClassifier
from pyspark.sql.functions import array
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import model_selection
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
import sklearn.linear_model as lms
from pyspark.sql import Row, Column

sc = SparkContext("local[2]", "PESU")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

model_lm=pickle.load(open('saved_models/model_lm_1000.sav', 'rb'))
model_sgd=pickle.load(open('saved_models/model_sgd_1000.sav', 'rb'))
model_mlp=pickle.load(open('saved_models/model_mlp_1000.sav', 'rb'))
#loaded_model_kmeans=pickle.load(open('saved_models/model_clustering_1000.sav', 'rb'))
output1=0
output2=0
output3=0
count=0


def converttojson(data):
	jsn=json.loads(data)
	l=list()
	for i in jsn:
		rows=tuple(jsn[i].values())
		l.append(rows)
	return l 	

def converttodf(data):

	global model
	global x
	global y
	global output1
	global output2
	global output3
	global count
	global kmeans
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
	#df.show()
	
	print('\n\nStages----------------------------------------------------------------\n')
	df_new=df

	regEx = regExTokenizer(inputCol= 'feature1' , outputCol= 'tokens', pattern= '\\W')

	print("regEx done")

	
	sw_remover = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	print("Stopwords done")

	stage_1 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	
	
	
	print("Word2vec done")

	
	st_indexing = StringIndexer(inputCol="feature2", outputCol="categoryIndex",  stringOrderType='alphabetAsc')

	print("Target column Done")
	
	pipeline=Pipeline(stages=[regEx, sw_remover, stage_1, st_indexing])
	pipelineFit=pipeline.fit(df)
	dataset=pipelineFit.transform(df)
	dataset=dataset.filter(dataset.feature1!='ham')
	new_dataframe=dataset.select(['vector'])
	new_dataframe_target=dataset.select(['categoryIndex'])
	new_dataframe.show(5)


	x=np.array(new_dataframe.select('vector').collect())
	y=np.array(new_dataframe_target.select('categoryIndex').collect())
	
	x = [np.concatenate(i) for i in x]
	
	kmeans=MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=1000)
	kmeans=kmeans.partial_fit(x)
	
	output1=output1+model_lm.score(x, y)
	print("Logistic regression accuracy: ",output1)
	
	output2=output2+model_sgd.score(x, y)
	print("SGD Classifier accuracy: ",output2)
	
	output3=output3+model_mlp.score(x, y)
	print("MLP Classiifier accuracy: ",output3)
	
	pred=model_lm.predict(x)
	print(confusion_matrix(y, pred))
	
	print(classification_report(y,pred))
	
	count=count+1
	
lines = ssc.socketTextStream("localhost",6100).map(converttojson).foreachRDD(converttodf)



ssc.start() 
ssc.awaitTermination(50)
ssc.stop()

result1=(output1*100)/(count+1)
result2=(output2*100)/(count+1)
result3=(output3*100)/(count+1)

results=[result1, result2, result3]
names=['Logistic Regression', 'SGD Classifier', 'MLP Classifer']

#plt.bar(names, results)
#plt.title("Average performance of models on test dataset (batch size 1000)")
#plt.show()


#clustering
kpred=kmeans.predict(x)
print(kpred)
pca=PCA(n_components=2)
scatter_plot_points=pca.fit_transform(x)
colors=['b','y']
x_axis=[o[0] for o in scatter_plot_points]
y_axis=[o[1] for o in scatter_plot_points]
fig, ax=plt.subplots(figsize=(20,10))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kpred])
plt.show()

