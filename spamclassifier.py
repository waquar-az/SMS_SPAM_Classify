import pandas as pd
messages = pd.read_csv('D:/NLP/smsspamclassy/SMSSpamCollection', sep="\t", names=["label","message"])


#data cleaning
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
   review = re.sub('[^a-zA-Z]'," ", messages['message'][i])
   review = review.lower()
   review = review.split()
   
   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
   review = " ".join(review)
   corpus.append(review)

#creating bag of words model
   
from sklearn.feature_extraction.text import  CountVectorizer             #TfidfVectorizer
cv = CountVectorizer(max_features=5000)
x =cv.fit_transform(corpus).toarray()   # x is independent feature


y=pd.get_dummies(messages['label'])
#remove one column in ham spam y ) 0 ham , 1 spam
y=y.iloc[:,1].values             #y is dependent variables  on x

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y, test_size= 0.20, random_state = 0)

#training model by naive byes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model =MultinomialNB().fit(x_train,y_train)

y_predict = spam_detect_model.predict(x_test)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_predict)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_predict)





