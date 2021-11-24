#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


#%%
dirpath = os.getcwd() 
filepath = os.path.join( dirpath ,'airline_passenger_satisfaction.csv')
df = pd.read_csv(filepath, index_col=[0]) 


# %%
print("data shape: ")
df.shape
df.dtypes


# %%
#df.describe()
df.head()

# %%
#check na values and delete na values
df.isnull().sum()
df=df.dropna()

# %%
# smart question 3: What factors have a strong correlation to loyal customers? 
print(" We are also starting with decision tree model to find out the answer.")

# We need to change some categorical variables to numerical variables for the model. 
df2=df.copy()
replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
                        'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}

df2.replace(replace_map, inplace=True)
#%%
df3 = df2.drop('customer_type',axis=1)
cols = df3.iloc[:,:]
cols

# %%
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

# %%
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
# train set
print('train set evaluation: /n')
print("score:", clf.score(x_train, y_train))
print("Confusion Matrix: \n",confusion_matrix(y_train, clf.predict(x_train)))
print("Classification report:\n",classification_report(y_train, clf.predict(x_train)))

# %%
# test set
print('test set evaluation: /n')
print("score:", clf.score(x_test, y_test))
print("Confusion Matrix: \n",confusion_matrix(y_test, clf.predict(x_test)))
print("Classification report:\n",classification_report(y_test, clf.predict(x_test)))

# %%
# Tree depth & leafs
print ('Tree Depth:', clf.get_depth())
print ('Tree Leaves:', clf.get_n_leaves())

# %%
# find best depth
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(x_train, y_train)
    return(clf.score(x_train,y_train), clf.score(x_test,y_test))
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
depths = np.arange(1,20)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, "bestdepth_score:", te_scores[te_best_index])
# %%
