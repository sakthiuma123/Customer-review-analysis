from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
38
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import re
def remove_pattern(input_txt,pattern):
r = re.findall(pattern,input_txt)
for i in r:
input_txt = re.sub(i,'',input_txt)#sub(characters we want to keep, removed character
replaced by space, string to work on)
return input_txt
data_filtered["Reviews"] = np.vectorize(remove_pattern)(data_filtered["Reviews"],
"@[\w]*")
data_filtered["Reviews"] = data_filtered["Reviews"].str.replace("[^a-zA-Z#]", " ")
data_filtered["Reviews"] = data_filtered["Reviews"].apply(lambda x: ' '.join([w for w in
x.split() if len(w)>3 ])) #tokenization
tokenized = data_filtered["Reviews"].apply(lambda x: x.split())
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized = tokenized.apply(lambda x : [stemmer.stem(i) for i in x])
tokenized.head()
all_words = ' '.join([text for text in data_filtered["Reviews"]])
data_filtered["Reviews"]
#Spilt Train And Test data
from sklearn.model_selection import train_test_split
X_train_data,x_test_data,Y_train_data,y_test_data=train_test_split(data_filtered["Reviews"],
data_filtered["r"],test_size=0.2)
Y_train_data.head()
X_train_data.head()