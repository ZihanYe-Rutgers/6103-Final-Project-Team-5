
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

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Instantiate dtree
tree1 = DecisionTreeClassifier(max_depth=None,criterion='entropy')
# Fit dt to the training set
clf = tree1.fit(x_train,y_train)
# tree1_training_score = 100*tree1.score(x_train, y_train)
# print ('Tree Depth:', tree1.get_depth())
# print ('Tree Leaves:', tree1.get_n_leaves())
# tree1_test_score = 100*tree1.score(x_test, y_test)
# print("Decision Tree accuracy:-\nTrain : ",tree1_training_score ,"%")
# print("Test: ",tree1_test_score ,"%. ")
y_train_pred = tree1.predict(x_train)
y_test_pred = tree1.predict(x_test)

# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report",classification_report(y_train, y_train_pred))


# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report",classification_report(y_test, y_test_pred))


#%%

# # Tree depth dependency
# max_d = tree1.get_depth()
# tree1_training_score, tree1_test_score = np.zeros(max_d), np.zeros(max_d)
# for i in range (max_d):
#   tree1 = DecisionTreeClassifier(max_depth=i+1)
#   tree1.fit(x_train,y_train)
#   tree1_training_score[i] = 100*tree1.score(x_train, y_train)
#   tree1_test_score[i] = 100*tree1.score(x_test, y_test)

# print (np.around (tree1_training_score, decimals=2))  
# print (np.around (tree1_test_score, decimals=2))
# plt.plot (tree1_training_score)
# plt.plot(tree1_test_score)
# plt.show()

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

print("From here let's build a new model with the top 6 important features.")


cols = df2[['online_boarding', 'inflight_wifi_service', 'type_of_travel', 'inflight_entertainment', 'customer_type', 'checkin_service']]
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree2 = DecisionTreeClassifier(max_depth=None,criterion='entropy')
# Fit dt to the training set
clf2 = tree2.fit(x_train,y_train)
y_train_pred = tree2.predict(x_train)
y_test_pred = tree2.predict(x_test)

# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report",classification_report(y_train, y_train_pred))


# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report",classification_report(y_test, y_test_pred))


#%%
#################################################################################
# TREE 3
####################################################################################

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
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree3 = DecisionTreeClassifier(max_depth=None,criterion='entropy')
# Fit dt to the training set
clf3 = tree3.fit(x_train,y_train)
y_train_pred = tree3.predict(x_train)
y_test_pred = tree3.predict(x_test)

# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report",classification_report(y_train, y_train_pred))


# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report",classification_report(y_test, y_test_pred))

#%%
# Get most important tree features
features = cols.columns
importances = tree3.feature_importances_
leading_indices = (-importances).argsort()[:14]
print ("Features sorted by importance:")
for i in range (14):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

print("\nFrom this model we can see that the most important feature for\
 passenger satisfection is online baording fascility followed by inflight\
 wifi service and legroom service. Weirdly, seat comfort is having the least\
 importance. \nOverall, the model test accuracy is not that satisfecory. Let's\
 build another model with the top 5 important features from here.")

#%%
############################################################################
# TREE 4
############################################################################


# cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'gate_location', 'online_boarding', 'inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'cleanliness']]
cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'online_boarding', 'inflight_entertainment', 'leg_room_service']]
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree4 = DecisionTreeClassifier(max_depth=None,criterion='entropy')
# Fit dt to the training set
clf4 = tree4.fit(x_train,y_train)
y_train_pred = tree4.predict(x_train)
y_test_pred = tree4.predict(x_test)

# Evaluate train-set accuracy
print('train set evaluation: ')
print("Accuracy score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",confusion_matrix(y_train, y_train_pred))
print("Classification report",classification_report(y_train, y_train_pred))


# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report",classification_report(y_test, y_test_pred))

#%%
# Get most important tree features
features = cols.columns
importances = tree4.feature_importances_
leading_indices = (-importances).argsort()[:5]
print ("Features sorted by importance:")
for i in range (5):
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')



#%%
################################################################################################
#%%
# # Graphing the tree
# from sklearn.tree import export_graphviz  
  
# # export the decision tree to a tree.dot file 
# # for visualizing the plot easily anywhere 

# filename = 'tree1'
# # import os
# # print(os.getcwd())
# export_graphviz(tree1, out_file = filename + '.dot' , feature_names =df2.columns[:-1]) 

# #%%
# import pydot
# (graph,) = pydot.graph_from_dot_file('tree1.dot')
# graph.write_png(filename+'.png')

#%%

# from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# from sklearn.tree import export_graphviz
# import pydotplus
# dot_data = StringIO()
# export_graphviz(tree1, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names = df2.columns[:-1],class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('tree1.png')
# Image(graph.create_png())

#%%
from sklearn import tree
tree.plot_tree(tree4,filled=True, rounded=True, feature_names = df2.columns[:-1],class_names=['0','1'])
plt.show()
# %%
from sklearn.tree import export_graphviz  
filename = 'tree4'
# import os
# print(os.getcwd())
export_graphviz(tree4, out_file = filename + '.dot' , feature_names =cols.columns) 

# %%
