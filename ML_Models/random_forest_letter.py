#Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#Importing the dataset
dataset=pd.read_csv('letter-recognitiondata.csv',header=None)
X=dataset.iloc[:,1:17].values
Y=dataset.iloc[:,0]

#Splitting the Data and Rescaling
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=0.20)

print(X_tr.shape)
print(y_tr.shape)

#Building the model(Simple CV)
random_forest =RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
random_forest.fit(X_tr, y_tr)
pred = random_forest.predict(X_cv)
a_cv=accuracy_score(y_cv, pred,normalize=True)* float(100)
print('The Validation accuracy of the model is :',a_cv)

#Predicting on test data 
pred_test = random_forest.predict(X_test)
a_test=accuracy_score(y_test,pred_test,normalize=True)* float(100)
print('The accuracy of the model is :',a_test)

#Confusion matrix and Metrics 
conf_mat = confusion_matrix(y_test, pred_test)
report = classification_report(y_test, pred_test)

#Cross Validation
cv_scores = []
random_forest =RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
scores = cross_val_score(random_forest, X_train, y_train, cv=10, scoring='accuracy')
cv_scores.append(scores.mean()) 