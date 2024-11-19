from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
clf = Pipeline([('tfidf',tfidf), ('clf',rf_model)])
# it will first do vectorization and then it will do classification
clf.fit(X_train_data, Y_train_data)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
40
y_pred = clf.predict(x_test_data)
print(y_pred)
cm=confusion_matrix(y_test_data, y_pred)
print(cm)
accuracy_score(y_test_data, y_pred)