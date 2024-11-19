from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
tfidf = TfidfVectorizer()
classifier = LinearSVC()
clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])
# it will first do vectorization and then it will do classification
39
clf.fit(X_train_data, Y_train_data)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(x_test_data)
print(y_pred)
cm=confusion_matrix(y_test_data, y_pred)
print(cm)
accuracy_score(y_test_data, y_pred)
print(f'\nClassification Report:\n{classification_report(y_test_data,y_pred)}')