# What factors mostly lead to loyal customers? 

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

#%%
dirpath = os.getcwd() 
filepath = os.path.join( dirpath ,'airline_passenger_satisfaction.csv')
df = pd.read_csv(filepath, index_col=[0]) 


#%%
print("data shape: ")
df.shape
df.dtypes


#%%
df.describe()
df.head()

#%%
#  #check na values and delete na values
df.isnull().sum()
df=df.dropna()

# %%
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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Train Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

#%%
# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report: \n ",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree1.get_depth())
print ('Tree Leaves:', tree1.get_n_leaves())

print("The train accuracy is 100% but the test accuracy is around 97%. \
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

plt.figure(figsize=(12, 6))
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
print("This is giving an interesting result, the flight distance is playing the major role for customer loyalty.\
 Type of travel is having the second highest. As we have seen before, there are two types of travel, business and\
 personal travel, possibly the airlines can focus on these particular groups by giving vouchers or special discounts.\
 We can see that satisfaction is having the 4th highest factor for being loyal customer. Which makes sense as satisfaction leads to loyality.")




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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

#%%
# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report: \n ",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree2.get_depth())
print ('Tree Leaves:', tree2.get_n_leaves())

print("The train accuracy is 99% but the test accuracy is around 88%.\
 We don't think it is a good model as the differece of accuracy is pretty high.")


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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Train Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree3.get_depth())
print ('Tree Leaves:', tree3.get_n_leaves())
print("The test accuracy is 86.21% but the train set accuracy is around 99.29%. \
 The tree has too many featues. Let's find out the which features have more importance.")

#%%
# Get most important tree features
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
print("Now we can see departure arrival time convenient is having the highest importance followed by\
 ease of online booking and online boarding. Whereas food and drink and cleanliness is having the least importance.\
 Looks like the convenience of air trips is having a major impact. As the technology as advanced and life schedule has become\
 very congested, passengers are giving more importance to the outside airplane services than inside/in-cabin services.\
 These three features are high enough to save a lot of time and give a more convenient schedule for the day.")


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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Train Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree4.get_depth())
print ('Tree Leaves:', tree4.get_n_leaves())
print("The test accuracy is 88.34% and the train set accuracy is around 90.10%. It is a good model.")

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
print("Now online boarding by itself covering the major importance of more than 50% alone. Let's\
 make another tree with top 5 features.")


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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Train Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree5.get_depth())
print ('Tree Leaves:', tree5.get_n_leaves())
print("Even removing 1 feature the test accuracy is still 88.97% but the train set accuracy is around 89.49%.")

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
print("Now departure_arrival_time_convenience covering 31% the total model importance.")


#%%
# Find depth via Cross-Validation
print("Now, we want to find out which depth of the tree would be the best model in this scenario.")

def run_cross_validation_on_trees(x, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, x, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(x, y).score(x, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores



def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()

sm_tree_depths = range(1,18)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(x_train, y_train, sm_tree_depths)

plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')



#%%
############################################################################
# TREE 6
############################################################################

print("We can see from the picture above, when depth is 8, it achieves a high average accuracy score.\
 From depth 13, the score is almost same and there is not much improvements in average accuracy score\
 from depth 8 to depth 13. And importantly, in depth 8, the lower bound of the confidence interval of\
 the accuracy is high enough to make the value significant. So we are choosing depth 8 to build the model. ")

print("Now we are making a final tree with depth 8.\n")

cols = df2[['departure_arrival_time_convenient', 'ease_of_online_booking', 'online_boarding', 'gate_location', 'checkin_service']]
x = cols.values
y = df2['customer_type'].values

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=1)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
tree6 = DecisionTreeClassifier(max_depth=8,criterion='entropy')

clf6 = tree6.fit(x_train,y_train)
y_train_pred = tree6.predict(x_train)
y_test_pred = tree6.predict(x_test)

# train set
print('train set evaluation: ')
print("Score: ",accuracy_score(y_train, y_train_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Train Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

# %%
# test set
print('test set evaluation: ')
print("Score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree6.get_depth())
print ('Tree Leaves:', tree6.get_n_leaves())
print("The test accuracy is 87.38% and the train set accuracy is around 87.43%.\
So our leading 5 parameters can predict both the training and test sets to about 87% accuracy,\
 with tree depth 8, and only 201 leaves.")

#%%

# Get most important tree features
features = cols.columns
x=len(cols.columns)
importances = tree6.feature_importances_
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


print("Now departure_arrival_time_convenient and ease_of_online_booking covering the major importance\
 of more than 50% together. Checkin_service importancec is having only 1.5% of importance for being loyal customer.")

#%%
# ROC-AUC
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = clf6.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC-AUC of the final Decision Tree")
# show the legend
plt.legend()
# show the plot
plt.show()

print("This ROC-AUC score also an inidication that this model is good model to predict loyal customers.")

#%%
# # Graphing the tree
# from sklearn.tree import export_graphviz  

# filename = 'tree6'
# import os
# print(os.getcwd())
# export_graphviz(tree6, out_file = filename + '.dot' , feature_names =cols.columns) 
