from sklearn import tree
tree_model = tree.DecisionTreeClassifier()
clf = Pipeline([('tfidf',tfidf), ('clf',tree_model)])
# it will first do vectorization and then it will do classification
clf.fit(X_train_data, Y_train_data)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(x_test_data)
print(y_pred)
cm=confusion_matrix(y_test_data, y_pred)
print(cm)
accuracy_score(y_test_data, y_pred)