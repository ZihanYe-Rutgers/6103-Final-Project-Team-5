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
from sklearn import tree
from sklearn.tree import export_graphviz  


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
#tree with all features
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
depths = np.arange(1,10)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
te_scores = [s[1] for s in scores]
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, "bestdepth_score:", te_scores[te_best_index])
# %%
# best depth plot
depths = np.arange(1,10)
plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree')
plt.ylabel('Scores')
plt.plot(depths, te_scores, label='test_scores')
plt.plot(depths, tr_scores, label='train_scores')
plt.legend()


# %%
# best depth tree
clf2 = DecisionTreeClassifier(max_depth=9)
clf2 = clf2.fit(x_train, y_train)

print("train set evaluation:")
print("score:", clf2.score(x_train, y_train))
print("Confusion Matrix: \n",confusion_matrix(y_train, clf2.predict(x_train)))
print("Classification report:\n",classification_report(y_train, clf2.predict(x_train)))

#%%
print("test set evaluation:")
print("score:", clf2.score(x_test, y_test))
print("Confusion Matrix: \n",confusion_matrix(y_test, clf2.predict(x_test)))
print("Classification report:\n",classification_report(y_test, clf2.predict(x_test)))


#%%
# Most important features
features = cols.columns
importances = clf2.feature_importances_
leading_indices = (-importances).argsort()[:6]
print ("Features sorted by importance:")
for i in range (6):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

#%%
# Save the tree dot
with open("tree.dot", 'w') as file:
    file = tree.export_graphviz(clf2, out_file=file)




#%%