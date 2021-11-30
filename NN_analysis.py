
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
print(" We are starting with decision tree model to find out the answer.")
# For decision tree we need the values to be numeric. So, we are converting the following features to numeric.
# 'Gender': {'Male': 0,'Female': 1 }
# 'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1}
# 'type_of_travel': {'Personal Travel': 0,'Business travel': 1}
# 'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2}
# 'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}

df2=df.copy()
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

# Train-Test split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# Instantiate dtree
tree1 = DecisionTreeClassifier(max_depth=None,criterion='entropy',class_weight='balanced')
# Fit dt to the training set
clf = tree1.fit(x_train,y_train)
y_train_pred = tree1.predict(x_train)
y_test_pred = tree1.predict(x_test)

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

print("The train accuracy is 100% but the test set accuracy is around 95%. \
 The tree has too many featues. Let's find out the which features have more importance.")

#%%

# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree1.feature_importances_
leading_indices = (-importances).argsort()[:x]
res=pd.DataFrame(columns = ['name','val'])

print ("Features sorted by importance:")
for i in range (x):
    name=features[leading_indices[i]]
    val=round(100*importances[leading_indices[i]],2)
    res = res.append({'name': name, 'val':val},ignore_index=True)
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='name',y='val', data=res)
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(res.val[i]), (x, y), ha='center')

plt.title('Features sorted by importance:')
plt.ylabel("Percentage of Importance(%)")
plt.xticks(rotation=90)
plt.show()
# add a graph

#####################################################################################################
# TREE 2
#####################################################################################################
#%%

print("From here let's build a new model with the top 6 important features.")


cols = df2[['online_boarding', 'inflight_wifi_service', 'type_of_travel', 'inflight_entertainment', 'customer_type', 'checkin_service']]
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree2 = DecisionTreeClassifier(max_depth=None,criterion='entropy',class_weight='balanced')
# Fit dt to the training set
clf2 = tree2.fit(x_train,y_train)
y_train_pred = tree2.predict(x_train)
y_test_pred = tree2.predict(x_test)

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
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree2.get_depth())
print ('Tree Leaves:', tree2.get_n_leaves())

print("The test accuracy is 93.6% and the train set accuracy is around 93.8%. \
 It means it is a good model.")
#%%
# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree2.feature_importances_
leading_indices = (-importances).argsort()[:x]
res=pd.DataFrame(columns = ['name','val'])

print ("Features sorted by importance:")
for i in range (x):
    name=features[leading_indices[i]]
    val=round(100*importances[leading_indices[i]],2)
    res = res.append({'name': name, 'val':val},ignore_index=True)
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

plt.figure(figsize=(6, 4))
ax = sns.barplot(x='name',y='val', data=res)
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(res.val[i]), (x, y), ha='center')

plt.title('Features sorted by importance:')
plt.ylabel("Percentage of Importance(%)")
plt.xticks(rotation=90)
plt.show()

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

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree3 = DecisionTreeClassifier(max_depth=None,criterion='entropy',class_weight='balanced')
# Fit dt to the training set
clf3 = tree3.fit(x_train,y_train)
y_train_pred = tree3.predict(x_train)
y_test_pred = tree3.predict(x_test)

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
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree3.get_depth())
print ('Tree Leaves:', tree3.get_n_leaves())

print("The test accuracy is 99% but the train set accuracy is around 92%. \
 Again, the tree seems to have too many featues. Let's find out the which features have more importance\
 and try to make the test accuracy better.")

#%%
# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree3.feature_importances_
leading_indices = (-importances).argsort()[:x]
res=pd.DataFrame(columns = ['name','val'])

print ("Features sorted by importance:")
for i in range (x):
    name=features[leading_indices[i]]
    val=round(100*importances[leading_indices[i]],2)
    res = res.append({'name': name, 'val':val},ignore_index=True)
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

plt.figure(figsize=(8, 6))
ax = sns.barplot(x='name',y='val', data=res)
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(res.val[i]), (x, y), ha='center')

plt.title('Features sorted by importance:')
plt.ylabel("Percentage of Importance(%)")
plt.xticks(rotation=90)
plt.show()

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

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree4 = DecisionTreeClassifier(max_depth=None,criterion='entropy',class_weight='balanced')
# Fit dt to the training set
clf4 = tree4.fit(x_train,y_train)
y_train_pred = tree4.predict(x_train)
y_test_pred = tree4.predict(x_test)

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
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree4.get_depth())
print ('Tree Leaves:', tree4.get_n_leaves())

print("The train accuracy is 91.84% but the test set accuracy is around 90.9%. \
 So our leading 5 parameters can predict both the training and test sets to about 91% accuracy,\
 with tree depth 18, and only 177 leaves.")


#%%
#%%
# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree4.feature_importances_
leading_indices = (-importances).argsort()[:x]
res=pd.DataFrame(columns = ['name','val'])

print ("Features sorted by importance:")
for i in range (x):
    name=features[leading_indices[i]]
    val=round(100*importances[leading_indices[i]],2)
    res = res.append({'name': name, 'val':val},ignore_index=True)
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

plt.figure(figsize=(5, 4))
ax = sns.barplot(x='name',y='val', data=res)
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(res.val[i]), (x, y), ha='center')

plt.title('Features sorted by importance:')
plt.ylabel("Percentage of Importance(%)")
plt.xticks(rotation=90)
plt.show()

print("Overall online boarding and inflight wifi service are covering the major importance.\
Together they are covering more than 70% of importance. ")

#%%
############################################################################
# TREE 5
############################################################################

print("Now we are making a final tree with depth 3 just for visualization purpose.")

# cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'gate_location', 'online_boarding', 'inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'cleanliness']]
cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'online_boarding', 'inflight_entertainment', 'leg_room_service']]
x = cols.values
y = df2['satisfaction'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%

# Instantiate dtree
tree5 = DecisionTreeClassifier(max_depth=3,criterion='entropy',class_weight='balanced')
# Fit dt to the training set
clf5 = tree5.fit(x_train,y_train)
y_train_pred = tree5.predict(x_train)
y_test_pred = tree5.predict(x_test)

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
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree5.get_depth())
print ('Tree Leaves:', tree5.get_n_leaves())

print("The train accuracy is 84.4% but the test set accuracy is around 83.8%. \
 So our leading 4 parameters can predict both the training and test sets to about 84% accuracy,\
 with tree depth 3, and only 8 leaves. But overall, this isn't a good model to predict accurately.")


#%%
# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree5.feature_importances_
leading_indices = (-importances).argsort()[:x]
res=pd.DataFrame(columns = ['name','val'])

print ("Features sorted by importance:")
for i in range (x):
    name=features[leading_indices[i]]
    val=round(100*importances[leading_indices[i]],2)
    res = res.append({'name': name, 'val':val},ignore_index=True)
    print (i+1, features[leading_indices[i]], round(100*importances[leading_indices[i]],2), '%')

plt.figure(figsize=(6, 4))
ax = sns.barplot(x='name',y='val', data=res)
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(res.val[i]), (x, y), ha='center')

plt.title('Features sorted by importance:')
plt.ylabel("Percentage of Importance(%)")
plt.xticks(rotation=90)
plt.show()
print("Now online boarding by itself covering the major importance of more than 50% alone.")


#%%[markdown]
# 


# While precision refers to the percentage of our results which are relevant,
# recall refers to the percentage of total relevant results correctly classified by our algorithm.
# Under this paradigm, having high Precision will be more important for our business
# problem. To correctly identify the crucial factors leading to customer satisfaction,
# the model prediction of the positive class, ‘Satisfied’ needs to be very reliable.
#
# For that we think the best decision tree would be the one with 5 most important features
# It has 89% test precision and 91% recall rate.
#
# The most important features to focous on is Online Boarding and Inflight Wi-Fi service.
# Moreover, as we found out before these 2 features aren't having good average rating as well.
# Online boarding rating is 3.25 and Inflight Wi-Fi is having only 2.73 ratings.
#
# So Airlines should focus on these 2 features at first to get better customer satisfaction.

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
