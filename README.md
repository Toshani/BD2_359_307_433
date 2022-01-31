# Email Spam Detection - 5th Semester Big Data Project
Team Number:  BD2_359_307_433

This is the final Big Data course project repository where we have implemented Machine Learning with Spark Streaming.
We have used the email spam detection data set and implemented Logistic regression, SGD Classifier and MLP Classiifier.

# Dataset Description:
The dataset given to us was aready cleaned and ready for pre-processing, having the following features: 
1. Each record consists of 3 features - the subject, the email content and the label
2. Each email is one of 2 classes, spam or ham
3. 30k examples in train and 3k in test

Link to the original dataset: https://www.kaggle.com/wanderfj/enron-spam (Enron Email Spam Detection Dataset)

# Libraries Used: 
PySpark, Pickle, Numpy

# Steps:
1. Create a SparkContext, StreamingContext and SQLContext to stream data real time and convert it to SQL readable data.
2. SQL is not Spark readable so convert it to JSON which is Spark readable. Then convert it to a dataframe and populate the dataset.
3. Pre-processing - Regex Tokeniser (Breaking down sentence into words), Stopword Remover (Removes stopwords/unnecessary words), Word2Vec (Converts words to vectors), String Indexer (Converts labels to indices, acts as data encoder).
4. Put all these into a pipeline, perform fit and transform.
5. You have your (vector, category) tuples ready, use this to train the various inbuilt models and determine the accuracy score.
6. Deploy it on local host and use Pickle to store it on the disk.
7. Perform the same tasks with the test data set. 

# Acknowledgements
I'd like to thank Prof. Animesh Giri and the TAs - Aditeya Baral, Ansh Sarkar and Vishesh P - for their guidance throughout the project. I'd also like to thank my teammates - Samriddhi Vishwakarma and Sohan Beela for their contribution in the project.

