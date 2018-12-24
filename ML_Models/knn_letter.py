#Importing the libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


#Importing the dataset
dataset=pd.read_csv('letter-recognitiondata.csv',header=None)
X=dataset.iloc[:,1:17].values
Y=dataset.iloc[:,0]

#Splitting the Data and Rescaling
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=0.20)

print(X_tr.shape)
print(y_tr.shape)

#Building the model
for m in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=m,algorithm='kd_tree',metric='euclidean',p=0,weights='uniform',leaf_size=20)
    knn.fit(X_tr, y_tr)
    pred = knn.predict(X_cv)
    a=accuracy_score(y_cv, pred,normalize=True)* float(100)
    print('\nCV accuracy for k = %d is %d%%' % (m, a))
    
#Evaluation on Test set
knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',metric='euclidean',p=0,weights='uniform',leaf_size=20)
knn.fit(X_tr, y_tr)
pred = knn.predict(X_test)
a=accuracy_score(y_test, pred,normalize=True)* float(100)
print('The accuracy of the model is :',a)

#Cross Validation
k_range = list(range(0,20))
k_values = list(filter(lambda x: x % 2 != 0, k_range))
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree',metric='euclidean',p=0,weights='uniform',leaf_size=20)
    scores = cross_val_score(knn, X_train, y_train, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())
    

# Misclassification and Determining best k
Mean_square_error = [1 - x for x in cv_scores]
optimal_k = k_values[Mean_square_error.index(min(Mean_square_error))]
print(optimal_k)

#Building model with optimum k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k,algorithm='kd_tree',metric='euclidean',p=0,weights='uniform',leaf_size=20)
knn_optimal.fit(X_train, y_train)
pred = knn_optimal.predict(X_test)
a=accuracy_score(y_test, pred,normalize=True)* float(100)
print('The accuracy of the model is :',a)

#Accuracy,Confusion matrix,Classification report
acc=accuracy_score(y_test, pred,normalize=True)* float(100)
conf_mat = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))