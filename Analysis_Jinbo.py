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

x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

tree1 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf = tree1.fit(x_train,y_train)
y_train_pred = tree1.predict(x_train)
y_test_pred = tree1.predict(x_test)

#%%
# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

#%%
# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report: \n ",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree1.get_depth())
print ('Tree Leaves:', tree1.get_n_leaves())

print("The train accuracy is 100% but the test accuracy is around 97%. \
 The tree has too many featues. Let's find out the which features have more importance.")

#%%
# Get most important tree features
features = cols.columns
importances = tree1.feature_importances_
leading_indices = (-importances).argsort()[:22]
print ("Features sorted by importance:")
for i in range (22):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')


#####################################################################################################
# TREE 2
#####################################################################################################
#%%

print("From here let's build a new model with the top 5 important features.")


cols = df2[['flight_distance', 'type_of_travel', 'age', 'satisfaction', 'departure_arrival_time_convenient']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

tree2 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf2 = tree2.fit(x_train,y_train)
y_train_pred = tree2.predict(x_train)
y_test_pred = tree2.predict(x_test)

#%%
# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

#%%
# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report: \n ",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree2.get_depth())
print ('Tree Leaves:', tree2.get_n_leaves())

print("The train accuracy is 88% but the test accuracy is around 99%. I don't think it is a good model")


#%%
#################################################################################
# TREE 3
####################################################################################

# only keep the relevant variables

print("Now we'd be making another model with the columns that represents airlines reviews so that,\
 we can actually find out the important factors of passenger satisfection. The features are: \
 \n1. inflight_wifi_service\
 \n2. departure_arrival_time_convenient\
 \n3. ease_of_online_booking \
 \n4. gate_location\
 \n5. food_and_drink \
 \n6. online_boarding \
 \n7. seat_comfort  \
 \n8. inflight_entertainment \
 \n9. onboard_service \
 \n10. leg_room_service \
 \n11. baggage_handling \
 \n12. checkin_service  \
 \n13. inflight_service\
 \n14. cleanliness")


cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking', 'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort','inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'inflight_service', 'cleanliness']]


# %%
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

# %%
tree3 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf3 = tree3.fit(x_train,y_train)
y_train_pred = tree3.predict(x_train)
y_test_pred = tree3.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree3.get_depth())
print ('Tree Leaves:', tree3.get_n_leaves())
print("The test accuracy is 86.26% but the train set accuracy is around 99.29%. \
 The tree has too many featues. Let's find out the which features have more importance.")

#%%
# Get most important tree features
features = cols.columns
importances = tree3.feature_importances_
leading_indices = (-importances).argsort()[:14]
print ("Features sorted by importance:")
for i in range (14):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')


#%%
############################################################################
# TREE 4
############################################################################

# tree with top 6 important features
print("From here let's build a new model with the top 6 important features.")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding', 'gate_location', 'leg_room_service', 'checkin_service']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree4 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf4 = tree4.fit(x_train,y_train)
y_train_pred = tree4.predict(x_train)
y_test_pred = tree4.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree4.get_depth())
print ('Tree Leaves:', tree4.get_n_leaves())
print("The test accuracy is 88.35% but the train set accuracy is around 90.10%.")

#%%
# Get most important tree features
features = cols.columns
importances = tree4.feature_importances_
leading_indices = (-importances).argsort()[:6]
print ("Features sorted by importance:")
for i in range (6):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

#%%
############################################################################
# TREE 5
############################################################################

# tree with top 5 important features
print("From here let's build a new model with the top 5 important features.")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding', 'gate_location', 'checkin_service']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree5 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf5 = tree5.fit(x_train,y_train)
y_train_pred = tree5.predict(x_train)
y_test_pred = tree5.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree5.get_depth())
print ('Tree Leaves:', tree5.get_n_leaves())
print("The test accuracy is 88.97% but the train set accuracy is around 89.49%.")

#%%
# Get most important tree features
features = cols.columns
importances = tree5.feature_importances_
leading_indices = (-importances).argsort()[:5]
print ("Features sorted by importance:")
for i in range (5):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')






#%%
#%%
############################################################################
# TREE 6
############################################################################

# tree with top 4 important features
print("From here let's build a new model with the top 4 important features.")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding', 'gate_location']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree6 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf6 = tree6.fit(x_train,y_train)
y_train_pred = tree6.predict(x_train)
y_test_pred = tree6.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree6.get_depth())
print ('Tree Leaves:', tree6.get_n_leaves())
print("The test accuracy is 88.95% but the train set accuracy is around 89.23%.")

#%%
# Get most important tree features
features = cols.columns
importances = tree6.feature_importances_
leading_indices = (-importances).argsort()[:4]
print ("Features sorted by importance:")
for i in range (4):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')



#%%
############################################################################
# TREE 7
############################################################################

# tree with top 3 important features
print("From here let's build a new model with the top 4 important features.")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree7 = DecisionTreeClassifier(max_depth=None,criterion='entropy')

clf7 = tree7.fit(x_train,y_train)
y_train_pred = tree7.predict(x_train)
y_test_pred = tree7.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree7.get_depth())
print ('Tree Leaves:', tree7.get_n_leaves())
print("The test accuracy is 86.58% but the train set accuracy is around 86.75%.\
So our leading 5 parameters can predict both the training and test sets to about 86% accuracy,\
 with tree depth 12, and only 175 leaves.")

#%%
# Get most important tree features
features = cols.columns
importances = tree7.feature_importances_
leading_indices = (-importances).argsort()[:3]
print ("Features sorted by importance:")
for i in range (3):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

print("Overall departure_arrival_time_convenient and online_boarding are covering the major importance.\
Together they are covering more than 70% of importance. ")


#%%
############################################################################
# TREE 8
############################################################################

print("Now we are making a final tree with depth 2.")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree8 = DecisionTreeClassifier(max_depth=2,criterion='entropy')

clf8 = tree8.fit(x_train,y_train)
y_train_pred = tree8.predict(x_train)
y_test_pred = tree8.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree8.get_depth())
print ('Tree Leaves:', tree8.get_n_leaves())
print("The test accuracy is 83.84% but the train set accuracy is around 83.46%.\
So our leading 5 parameters can predict both the training and test sets to about 83% accuracy,\
 with tree depth 2, and only 4 leaves.")

#%%
# Get most important tree features
features = cols.columns
importances = tree8.feature_importances_
leading_indices = (-importances).argsort()[:3]
print ("Features sorted by importance:")
for i in range (3):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

print("Now departure_arrival_time_convenient covering the major importance of more than 50% alone.")





#%%
# Graphing the tree
from sklearn.tree import export_graphviz  

filename = 'tree8'
import os
print(os.getcwd())
export_graphviz(tree8, out_file = filename + '.dot' , feature_names =cols.columns) 

#%%
import pydot
(graph,) = pydot.graph_from_dot_file('tree8.dot')
graph.write_png(filename+'.png')