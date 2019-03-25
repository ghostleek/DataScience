#!/usr/bin/env python
# coding: utf-8

# ## ML
# Just making some models

# In[164]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#options for pandas
pd.options.display.max_columns = 30


df = pd.read_csv('file:///Users/kahhow/Downloads/capstone_starter/profiles.csv', delimiter = ',')
df.head()



# In[75]:


plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()


# In[ ]:





# In[45]:


df.count()


# In[46]:


df.income.min()


# In[47]:


df.income.max()


# In[163]:


df.head(50)


# In[165]:


# Create a dictionary using which we 
# will remap the values 
dict = {'never' : 0, 'sometimes' : 1, 'often' : 2} 
  
# Print the dictionary 
print(dict) 
  
# Remap the values of the dataframe 
df['drugs']= df['drugs'].map(dict) 
  
# Print the DataFrame after modification 
print(df) 


# In[104]:


df.head(80)


# In[166]:


# Create a dictionary using which we 
# will remap the values 
dict1 = {'not at all' : 0, 'rarely': 1, 'socially' : 2, 'often' : 3, 'very often':4, 'desperately':5} 

# Print the dictionary 
print(dict1) 
  
# Remap the values of the dataframe 
df['drinks']= df['drinks'].map(dict1) 
  
print(df)


# In[167]:


df['drinks'] = df['drinks'].fillna(0)
df['drugs']=df['drugs'].fillna(0)
print(df)


# In[226]:


# map smokes responses to a number code

smokes_codes = {
    "no": 0,
    "trying to quit":1,
    "when drinking": 2,
    "sometimes": 3,
    "yes": 4,
}

df["smokes"] = df['smokes'].map(smokes_codes)


# In[227]:


education_codes={
    "working on space camp":1,
    "working on two-year college":2,
    "working on college/university":3,
    "graduated from college/university":4,
    "working on masters program":5,
    "graduated from masters program":6
}

df["education"] = df['education'].map(education_codes)


# In[228]:


df.head(80)


# In[230]:


# check which columns have NaN values
df.isna().any()
#remove NaNs in columns
other_columns_with_nan_values = [
    'drinks', 
    'drugs', 
    'height', 
    'smokes',
    'education'
]
df.dropna(subset=other_columns_with_nan_values, inplace=True)


# In[237]:


import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


x = df[['drugs','drinks','height','education','smokes','income']]

y = df[['age']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

# Input code here:

print("Train score:")
print(lm.score(x_train,y_train))

print("Test score:")
print(lm.score(x_test,y_test))

residuals = y_predict - y_test
plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis')

plt.show()


# In[ ]:





# In[256]:



model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict, alpha=0.1)
plt.plot(range(50), range(50))

plt.xlabel("Age: $Y_i$")
plt.ylabel("Predicted age: $\hat{Y}_i$")
plt.title("Actual Age vs Predicted Age using MLR")

plt.show()


# In[264]:


from sklearn.neighbors import KNeighborsRegressor

kn_regressor = KNeighborsRegressor(n_neighbors = 250, weights = "distance")
kn_regressor.fit(x_train, y_train)

knr_training_score = kn_regressor.score(x_train, y_train)

print("KNR training set score: %s" %(knr_training_score))

knr_test_score = kn_regressor.score(x_test, y_test)

print("KNR test set score: %s" %(knr_test_score))

#generate plot showing difference between predicted age and actual age using K Neighbors Regressor

knr_predictions = kn_regressor.predict(x_test)

plt.scatter(y_test, knr_predictions,  alpha=0.1)
plt.title("Actual age versus predicted age using K Neighbors Regression")
plt.xlabel("Actual age")
plt.ylabel("Predicted age")
plt.show()


# In[269]:


#can we use age, drinks, smoking, income to predict drug use?


# select features and scale data for regression

from sklearn.preprocessing import scale

list_of_features_classification = [
    'drinks',
    'smokes',
    'age',
    'income'
]
features_classification = df[list_of_features_classification]
scaled_features_classification = scale(features_classification, axis=0)
labels = df['drugs']


# In[270]:


# generate training and test sets

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(scaled_features_classification, labels, test_size=0.2, random_state=40)


# In[272]:


# build K Nearest Neighbors model to predict drug use

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=76)
knn_classifier.fit(train_data, train_labels)

knn_training_score = knn_classifier.score(train_data, train_labels)

print("KNN training set score: %s" %(knn_training_score))

knn_test_score = knn_classifier.score(test_data, test_labels)

print("KNN test set score: %s" %(knn_test_score))


# In[273]:


# generate classification report and confusion matrix for K Nearest Neighbors classifier

from sklearn.metrics import classification_report, confusion_matrix 

knn_predictions = knn_classifier.predict(test_data)
print(confusion_matrix(test_labels, knn_predictions))
print(classification_report(test_labels, knn_predictions))


# In[274]:


# generate plot showing K Nearest Neighbors classifier score based on different n_neighbors values 
# and print out n_neighbors value that results in highest score

scores = []

highest_score = {
    'k': 0,
    'score': 0
}

for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    score = classifier.score(test_data, test_labels)
    scores.append(score)
    if score > highest_score['score']:
        highest_score = {
            'k': k,
            'score': score
        }

x_values = range(1, 101)

y_values = scores

plt.plot(x_values, y_values)
plt.title('Accuracy of K Nearest Neighbours classifier based on different k values')
plt.ylabel('Accuracy')
plt.xlabel('k value')
plt.show()
print(highest_score)


# In[277]:


#Use of support vector machine classification
from sklearn.svm import SVC

svc_classifier = SVC(kernel='rbf', gamma=10)
svc_classifier.fit(train_data, train_labels)

svc_training_score = svc_classifier.score(train_data, train_labels)

print("SVC training set score: %s" %(svc_training_score))

svc_test_score = classifier.score(test_data, test_labels)

print("SVC test set score: %s" %(svc_test_score))


# In[278]:


# generate classification report and confusion matrix for Support Vector Machines classifier

from sklearn.metrics import classification_report, confusion_matrix 

svc_predictions = svc_classifier.predict(test_data)
print(confusion_matrix(test_labels, svc_predictions))
print(classification_report(test_labels, svc_predictions))


# In[279]:


# generate plot showing SVC Classifier score based on different gamma values 
# and print out gamma value that results in highest score

scores = []

highest_score = {
    'gamma': 0,
    'score': 0
}

gamma_values = np.arange(0.1, 1.0, 0.1)

for val in gamma_values:
    svc_classifier = SVC(kernel='rbf', gamma=val)
    svc_classifier.fit(train_data, train_labels)
    score = svc_classifier.score(test_data, test_labels)
    scores.append(score)
    if score > highest_score['score']:
        highest_score = {
            'gamma': val,
            'score': score
        }

x_values = gamma_values

y_values = scores

plt.plot(x_values, y_values)
plt.title('Accuracy of Support Vector Machines classifier based on different gamma values')
plt.ylabel('Accuracy')
plt.xlabel('Gamma value')
plt.show()
print(highest_score)

