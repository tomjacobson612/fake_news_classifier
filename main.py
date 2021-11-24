import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
dataframe_train = pandas.read_csv("data/train.csv")
dataframe_test = pandas.read_csv("data/test.csv")


vectorized = TfidfVectorizer(analyzer='word', stop_words='english')

matrix_train = vectorized.fit_transform(dataframe_train['title'].values.astype('U'))
matrix_test = vectorized.transform(dataframe_test['title'].values.astype('U'))

tokens = vectorized.get_feature_names_out()

x = pandas.DataFrame(data=matrix_train.toarray(), index=dataframe_train['label'], columns=tokens)
y = pandas.DataFrame(data=matrix_test.toarray(), index=dataframe_test['title'], columns=tokens)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(matrix_train, dataframe_train['label'])
y_pred = pac.predict(matrix_test)

dataframe_submit = pandas.read_csv("data/submit.csv")

print(len(dataframe_submit['label']))
print(len(y_pred))
prediction = accuracy_score(dataframe_submit['label'], y_pred)

print(prediction)
'''

# Read the data CAREFULLY
dataframe = pandas.read_csv("data/news.csv")


# Split the data
x_train, x_test, y_train, y_test = train_test_split(dataframe['text'], dataframe.label, test_size=.2, random_state=7) # rand state is a seed, remove to have it random

# Vectorize

vectorized = TfidfVectorizer(max_df=.7, stop_words='english')

matrix_train = vectorized.fit_transform(x_train)
matrix_test = vectorized.transform(x_test)

# Passive Agressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(matrix_train, y_train)
y_pred = pac.predict(matrix_test)

prediction = accuracy_score(y_test, y_pred)

print(prediction)
