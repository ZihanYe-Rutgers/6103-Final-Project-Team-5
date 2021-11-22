
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

x = df2.values 
y = df2['satisfaction'].values

# # Normalize features
# for feature in range (x.shape[1]):
#     min = x[:,feature].min()
#     max = x[:,feature].max()
#     x[:,feature] = (x[:,feature]-min) / (max-min)
    
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(max_depth=None)
dct.fit(x_train,y_train)
dct_training_score = 100*dct.score(x_train, y_train)
print ('Tree Depth:', dct.get_depth())
print ('Tree Leaves:', dct.get_n_leaves())
dct_test_score = 100*dct.score(x_test, y_test)
print("Decision Tree accuracy:-\nTrain : ",dct_training_score ,"%")
print("Test: ",dct_test_score ,"%. ")

#%%