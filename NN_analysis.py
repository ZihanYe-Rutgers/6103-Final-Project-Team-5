
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
dirpath = os.getcwd() # print("current directory is : " + dirpath)
filepath = os.path.join( dirpath ,'airline_passenger_satisfaction.csv')
df = pd.read_csv(filepath, index_col=[0]) 


#%%
# the datatypes of columns
print("Shape of df:")
df.shape
df.dtypes

#%%
# top 5 data from data dataframe
df.head()


#%%
# checking which columns have null values and count of nulls
df.isnull().sum()
#%%
# we can drop those null rows in the arrival_delay_in_minutes column
df=df.dropna()
print("shape of the df after dropping nulls:")
df.shape

############################################################################
# SMART question 2:
# What are the factors that satisfy the passenger?
############################################################################
#%%
# We are starting with decision tree model to find out the answer.
# For decision tree we need the values to be numeric.
df2=df.copy()
# replace_map_gender = {'Gender': {'Male': 0,'Female': 1 }}
# replace_map_ctype = {'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1}}
# replace_map_travel = {'type_of_travel': {'Personal Travel': 0,'Business travel': 1}}
# replace_map_class = {'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2}}
# replace_map_sat = {'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}}

replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
                        'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}

df2.replace(replace_map, inplace=True)

from sklearn.model_selection import train_test_split

cols = df2.iloc[:,:-1]
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

tree1 = DecisionTreeClassifier(max_depth=None)
tree1.fit(x_train,y_train)
tree1_training_score = 100*tree1.score(x_train, y_train)
print ('Tree Depth:', tree1.get_depth())
print ('Tree Leaves:', tree1.get_n_leaves())
tree1_test_score = 100*tree1.score(x_test, y_test)
print("Decision Tree accuracy:-\nTrain : ",tree1_training_score ,"%")
print("Test: ",tree1_test_score ,"%. ")


#%%

# Tree depth dependency
max_d = tree1.get_depth()
tree1_training_score, tree1_test_score = np.zeros(max_d), np.zeros(max_d)
for i in range (max_d):
  tree1 = DecisionTreeClassifier(max_depth=i+1)
  tree1.fit(x_train,y_train)
  tree1_training_score[i] = 100*tree1.score(x_train, y_train)
  tree1_test_score[i] = 100*tree1.score(x_test, y_test)

print (np.around (tree1_training_score, decimals=2))  
print (np.around (tree1_test_score, decimals=2))
plt.plot (tree1_training_score)
plt.plot(tree1_test_score)

#%%

# Get most important tree features
features = cols.columns
importances = tree1.feature_importances_
leading_indices = (-importances).argsort()[:22]
print ("Features sorted by importance:")
for i in range (22):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')
#%%