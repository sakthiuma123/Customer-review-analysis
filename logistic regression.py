from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
clf = Pipeline([('tfidf',tfidf), ('clf',lr_model)])
# it will first do vectorization and then it will do classification
clf.fit(X_train_data, Y_train_data)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(x_test_data)
print(y_pred)
cm=confusion_matrix(y_test_data, y_pred)
print(cm)
accuracy_score(y_test_data, y_pred)
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
iris = load_iris()
iris.data[0:5], iris.target
X = iris.data
target = iris.target
names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(verbosity=2, max_time_mins=10)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.fitted_pipeline_
print(tpot.score(X_test, y_test))
sdf = df[['Reviews','Rating']]
sdf.head(2)
def assign_sentiment(Rating):
if float(Rating) >= 3:
return "Positive"
elif float(Rating) <= 2:
return "neutral"
else:
return "Negative"
sdf['sentiment'] = sdf['Rating'].apply(assign_sentiment)
sdf.head(3)
sdf.drop('Rating', inplace=True, axis=1)
sdf.head()
sdf['Reviews'] = sdf['Reviews'].astype(str)
sdf.head()
sdf = sdf[sdf['Reviews'].map(len) <= 2000]
sdf.head()
sdf.dropna(inplace=True)
sdf.reset_index(inplace=True, drop=True)
actual_sentiment = sdf['sentiment']
actual_output = pd.DataFrame(actual_sentiment)
actual_output