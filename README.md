"# Spam-News-Detection" 
"# Spam-News-Detection" 
import pandas as pd
import numpy as np
True_news = pd.read_csv("True_news.csv") 
#loads the True_news data into True_news
Fake_news = pd.read_csv("Fake_news.csv")
#loads the Fake_news data into Fake_news
#Add label values
True_news['label']=0
Fake_news['label']=1
dataset1 = True_news[[ 'text' , 'label' ]]
dataset2 = Fake_news[[ 'text' , 'label']]
#Selecting only two columns and loading into dataset1 and dataset2 
dataset = pd.concat([ dataset1, dataset2])
#concatenating two datasets 
#checking for null values
dataset.isnull().sum()          #returns text =0 and label 0
#For shuffling the data
dataset = dataset.sample(frac = 1)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer ps = WordNetLemmatizer()
stopwords=stopwords.words('english') nltk.download('wordnet')def clean_row(row):
row=row.lower()
row=re.sub('[^a-ZA-Z]', ' ',row)
token=row.split()
news=[ps.lemmatize(word) for word in token if not word in stopwords] cleanned_news=''.join(news)
return cleanned_news
dataset['text']=dataset['text'].apply(lambda x: clean_row(x))
from sklearn.feature_extraction.text import Tfidfvectorizer
vectorizer = Tfidfvectorizer(max_features = 50000, lowercase=False, 
ngram_range=(1,2))
x=dataset.iloc[:35000,0]
y=dataset.iloc[:35000,1]
from sklearn.model_selection import train_test_split
train_data,test_data,train_label,test_label=train_test_split(x,y,test_size=0.2,rand om_state=0)
vec_train_data=vectorizer.fit_transform(train_data) vec_train_data=vec_test_data.toarray()
vec_test_data=vectorizer.fit_transform(test_data) vec_test_data=vec_test_data.toarray() training_data=pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names()) testing_data=pd.DataFrame(vec_#Model
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(training_data,train_label)
y_pred=clf.predict(testing_data)
from sklearn.metrics import accuracy_score
accuracy_score(test_label,y_pred)    #returns the accuracy of prediction y_pred_train=clf.predict(training_data)
txt="Some news we will give here"
news=clean_row(txt)
pred=clf.predict(vectorizer.transform([news]).toarray()) txt=input("Enter News:")
news=clean_row(str(txt))
pred=clf.predict(vectorizer.transform([news]).toarray()) if pred == 0:
print("News is correct") else:
print("News is fake")  
