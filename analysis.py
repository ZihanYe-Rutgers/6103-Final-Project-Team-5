
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.palettes import color_palette
# plt.style.use('fivethirtyeight')

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


#%%

# For data visualization purpose we are convering our variables to appropiate data types \
#  i.e. numeric, categorical, ordered categorical and the dataframe name is df_vis

df_rating=pd.DataFrame(columns=['Feature','Rate'])
df_vis = df.copy()
print('Shape of the df_vis:')
df_vis.shape

#%%

## Numeric variables:

# age - numeric
print("Lets inspect the age column: ")
print(df_vis.age.describe())
print(df_vis.age.value_counts())
# histogram
sns.histplot(x ='age', data = df_vis)
plt.title("Distribution of age")
plt.show()

#%%
# flight_distance - numeric
print("Lets inspect the flight_distance column: ")
print(df_vis.flight_distance.describe())
print(df_vis.flight_distance.value_counts())
# histogram
sns.histplot(x ='flight_distance', data = df_vis)
plt.title("Distribution of flight_distance")
plt.show()

#%%
# departure_delay_in_minutes - numeric
print("Lets inspect the departure_delay_in_minutes column: ")
print(df_vis.departure_delay_in_minutes.describe())
print(df_vis.departure_delay_in_minutes.value_counts())
# histogram
plt.figure(figsize=(5, 5))
sns.histplot(x ='departure_delay_in_minutes', data = df_vis)
plt.xlim(0,200)
plt.ylim(0,4000)
plt.title("Distribution of departure_delay_in_minutes")
plt.show()

#%%
# arrival_delay_in_minutes - numeric
print("Lets inspect the arrival_delay_in_minutes column: ")
print(df_vis.arrival_delay_in_minutes.describe())
print(df_vis.arrival_delay_in_minutes.value_counts())
# histogram
sns.histplot(x ='arrival_delay_in_minutes', data = df_vis)
plt.xlim(0,200)
plt.ylim(0,3000)
plt.title("Distribution of arrival_delay_in_minutes")
plt.show()

################################################################################################

#%%
## Categorical Variables:


# Gender - categorical
print("Lets inspect the Gender column: ")
print(df_vis.Gender.describe())
print(df_vis.Gender.value_counts())
# Converting Gender to Categorical- Male, Female
df_vis['Gender'] = pd.Categorical(df_vis['Gender'], ordered=False)
# count plot
sns.countplot(x ='Gender', data = df_vis)
plt.title("Distribution of Gender")
plt.show()

#%%
# customer_type - categorical

print("Lets inspect the customer_type column: ")
print(df_vis.customer_type.describe())
print(df_vis.customer_type.value_counts())
# Converting customer_type to Categorical- Loyal Customer, disloyal Customer
df_vis['customer_type'] = pd.Categorical(df_vis['customer_type'], ordered=False)
# count plot
sns.countplot(x ='customer_type', data = df_vis)
plt.title("Distribution of customer_type")
plt.show()

#%%
# type_of_travel - categorical

print("Lets inspect the type_of_travel column: ")
print(df_vis.type_of_travel.describe())
print(df_vis.type_of_travel.value_counts())
# Converting customer_type to Categorical- Business travel, Personal Travel
df_vis['type_of_travel'] = pd.Categorical(df_vis['type_of_travel'], ordered=False)
# count plot
sns.countplot(x ='type_of_travel', data = df_vis)
plt.title("Distribution of type_of_travel")
plt.show()

#%%
# satisfaction - categorical
print("Lets inspect the satisfaction column: ")
print(df_vis.satisfaction.describe())
print(df_vis.satisfaction.value_counts())
# Converting satisfaction to Categorical- neutral or dissatisfied, satisfied
df_vis['satisfaction'] = pd.Categorical(df_vis['satisfaction'], ordered=False)
# count plot
sns.countplot(x ='satisfaction', data = df_vis)
plt.title("Distribution of satisfaction")
plt.show()

#%%
## Ordered Calegorical Variables: 

# customer_class - categorical ordered
print("Lets inspect the customer_class column: ")
print(df_vis.customer_class.describe())
print(df_vis.customer_class.value_counts())
# Converting customer_class to Ordered Categorical- Eco, Eco Plus, Business
df_vis['customer_class'] = pd.Categorical(df_vis['customer_class'], categories=['Eco', 'Eco Plus', 'Business'], ordered=True)
# count plot
sns.countplot(x ='customer_class', data = df_vis)
plt.title("Distribution of customer_class")
plt.show()


#%%
# Converting inflight_wifi_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_wifi_service'] = pd.Categorical(df_vis['inflight_wifi_service'], ordered=True)
# count plot
sns.countplot(x ='inflight_wifi_service', data = df_vis)
plt.title("Distribution of inflight_wifi_service")
plt.show()

# Avg rating 
df_rating = df_rating.append({"Feature":"inflight_wifi_service", "Rate":df['inflight_wifi_service'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['inflight_wifi_service'].mean(),2))

#%%
# departure_arrival_time_convenient - ordered categorical
print("Lets inspect the departure_arrival_time_convenient column: ")
print(df_vis.departure_arrival_time_convenient.describe())
print(df_vis.departure_arrival_time_convenient.value_counts())
# Converting departure_arrival_time_convenient to Ordered Categorical- 0,1,2,3,4,5
df_vis['departure_arrival_time_convenient'] = pd.Categorical(df_vis['departure_arrival_time_convenient'], ordered=True)
# count plot
sns.countplot(x ='departure_arrival_time_convenient', data = df_vis)
plt.title("Distribution of departure_arrival_time_convenient")
plt.show()


df_rating=df_rating.append({"Feature":"departure_arrival_time_convenient", "Rate":df['departure_arrival_time_convenient'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['departure_arrival_time_convenient'].mean(),2))

#%%
# ease_of_online_booking - ordered categorical
print("Lets inspect the ease_of_online_booking column: ")
print(df_vis.ease_of_online_booking.describe())
print(df_vis.ease_of_online_booking.value_counts())
# Converting ease_of_online_booking to Ordered Categorical- 0,1,2,3,4,5
df_vis['ease_of_online_booking'] = pd.Categorical(df_vis['ease_of_online_booking'], ordered=True)
# count plot
sns.countplot(x ='ease_of_online_booking', data = df_vis)
plt.title("Distribution of ease_of_online_booking")
plt.show()

df_rating=df_rating.append({"Feature":"ease_of_online_booking", "Rate":df['ease_of_online_booking'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['ease_of_online_booking'].mean(),2))

#%%
# gate_location  - ordered categorical
print("Lets inspect the gate_location column: ")
print(df_vis.gate_location.describe())
print(df_vis.gate_location.value_counts())
# Converting gate_location to Ordered Categorical- 0,1,2,3,4,5
df_vis['gate_location'] = pd.Categorical(df_vis['gate_location'], ordered=True)
# count plot
sns.countplot(x ='gate_location', data = df_vis)
plt.title("Distribution of gate_location")
plt.show()

df_rating=df_rating.append({"Feature":"gate_location", "Rate":df['gate_location'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['gate_location'].mean(),2))


#%%
# food_and_drink - ordered categorical

print("Lets inspect the food_and_drink column: ")
print(df_vis.food_and_drink.describe())
print(df_vis.food_and_drink.value_counts())
# Converting food_and_drink to Ordered Categorical- 0,1,2,3,4,5
df_vis['food_and_drink'] = pd.Categorical(df_vis['food_and_drink'], ordered=True)
# count plot
sns.countplot(x ='food_and_drink', data = df_vis)
plt.title("Distribution of food_and_drink")
plt.show()

df_rating=df_rating.append({"Feature":"food_and_drink", "Rate":df['food_and_drink'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['food_and_drink'].mean(),2))

#%%
# online_boarding - ordered categorical
print("Lets inspect the online_boarding column: ")
print(df_vis.online_boarding.describe())
print(df_vis.online_boarding.value_counts())
# Converting online_boarding to Ordered Categorical- 0,1,2,3,4,5
df_vis['online_boarding'] = pd.Categorical(df_vis['online_boarding'], ordered=True)
# count plot
sns.countplot(x ='online_boarding', data = df_vis)
plt.title("Distribution of online_boarding")
plt.show()


df_rating=df_rating.append({"Feature":"online_boarding", "Rate":df['online_boarding'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['online_boarding'].mean(),2))

#%%
# seat_comfort - ordered categorical
print("Lets inspect the seat_comfort column: ")
print(df_vis.seat_comfort.describe())
print(df_vis.seat_comfort.value_counts())
# Converting seat_comfort to Ordered Categorical- 0,1,2,3,4,5
df_vis['seat_comfort'] = pd.Categorical(df_vis['seat_comfort'], ordered=True)
# count plot
sns.countplot(x ='seat_comfort', data = df_vis)
plt.title("Distribution of seat_comfort")
plt.show()

df_rating=df_rating.append({"Feature":"seat_comfort", "Rate":df['seat_comfort'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['seat_comfort'].mean(),2))


#%%
# inflight_entertainment - ordered categorical
print("Lets inspect the Gender column: ")
print(df_vis.inflight_entertainment.describe())
print(df_vis.inflight_entertainment.value_counts())
# Converting inflight_entertainment to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_entertainment'] = pd.Categorical(df_vis['inflight_entertainment'], ordered=True)
# count plot
sns.countplot(x ='inflight_entertainment', data = df_vis)
plt.title("Distribution of inflight_entertainment")
plt.show()

df_rating=df_rating.append({"Feature":"inflight_entertainment", "Rate":df['inflight_entertainment'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['inflight_entertainment'].mean(),2))




#%%
# onboard_service - ordered categorical

print("Lets inspect the onboard_service column: ")
print(df_vis.onboard_service.describe())
print(df_vis.onboard_service.value_counts())
# Converting onboard_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['onboard_service'] = pd.Categorical(df_vis['onboard_service'], ordered=True)
# count plot
sns.countplot(x ='onboard_service', data = df_vis)
plt.title("Distribution of onboard_service")
plt.show()

df_rating=df_rating.append({"Feature":"onboard_service", "Rate":df['onboard_service'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['onboard_service'].mean(),2))



#%%
# leg_room_service - ordered categorical
print("Lets inspect the leg_room_service column: ")
print(df_vis.leg_room_service.describe())
print(df_vis.leg_room_service.value_counts())
# Converting leg_room_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['leg_room_service'] = pd.Categorical(df_vis['leg_room_service'], ordered=True)
# count plot
sns.countplot(x ='leg_room_service', data = df_vis)
plt.title("Distribution of leg_room_service")
plt.show()

df_rating=df_rating.append({"Feature":"leg_room_service", "Rate":df['leg_room_service'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['leg_room_service'].mean(),2))



#%%
# baggage_handling- ordered categorical
print("Lets inspect the baggage_handling column: ")
print(df_vis.baggage_handling.describe())
print(df_vis.baggage_handling.value_counts())
# Converting baggage_handling to Ordered Categorical- 0,1,2,3,4,5
df_vis['baggage_handling'] = pd.Categorical(df_vis['baggage_handling'], ordered=True)
# count plot
sns.countplot(x ='baggage_handling', data = df_vis)
plt.title("Distribution of baggage_handling")
plt.show()

df_rating=df_rating.append({"Feature":"baggage_handling", "Rate":df['baggage_handling'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['baggage_handling'].mean(),2))



#%%
# checkin_service - ordered categorical
print("Lets inspect the Gender column: ")
print(df_vis.checkin_service.describe())
print(df_vis.checkin_service.value_counts())
# Converting checkin_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['checkin_service'] = pd.Categorical(df_vis['checkin_service'], ordered=True)
# count plot
sns.countplot(x ='checkin_service', data = df_vis)
plt.title("Distribution of checkin_service")
plt.show()

df_rating=df_rating.append({"Feature":"checkin_service", "Rate":df['checkin_service'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['checkin_service'].mean(),2))


#%%
# inflight_service - ordered categorical
print("Lets inspect the inflight_service column: ")
print(df_vis.inflight_service.describe())
print(df_vis.inflight_service.value_counts())
# Converting inflight_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_service'] = pd.Categorical(df_vis['inflight_service'], ordered=True)
# count plot
sns.countplot(x ='inflight_service', data = df_vis)
plt.title("Distribution of inflight_service")
plt.show()

df_rating=df_rating.append({"Feature":"inflight_service", "Rate":df['inflight_service'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['inflight_service'].mean(),2))


#%%
# cleanliness - ordered categorical
print("Lets inspect the cleanliness column: ")
print(df_vis.cleanliness.describe())
print(df_vis.cleanliness.value_counts())
# Converting cleanliness to Ordered Categorical- 0,1,2,3,4,5
df_vis['cleanliness'] = pd.Categorical(df_vis['cleanliness'], ordered=True)
# count plot
sns.countplot(x ='cleanliness', data = df_vis)
plt.title("Distribution of cleanliness")
plt.show()

df_rating=df_rating.append({"Feature":"cleanliness", "Rate":df['cleanliness'].mean()}, ignore_index=True)
print("Overall Rating: ", round(df['cleanliness'].mean(),2))

#%%

#This is correlation matrix and heatmap of categorical variables
################Categorical correlation
categorical_features=df.copy()
replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
                        'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}

categorical_features.replace(replace_map, inplace=True)
# print(categorical_features.dtypes)

corr_categorical = categorical_features.corr()
ax2 = plt.subplots(figsize=(10,10))
ax2 = sns.heatmap(corr_categorical,annot=True,annot_kws={"fontsize":8})
sns.set_context(font_scale=10)
plt.title('Relationships between categorical features')
plt.show()

#%%
#################################################################################
# EDA 1
#################################################################################
# 1.1. Does arrival/departure have any effect on customer satisfaction?
# departure_delay_in_minutes - satisfaction
#

print("Our 1st SMART Question is  Does arrival/departure have any effect on\
 customer satisfaction? \n At first we are starting with a correlation plot.")
# df3=df.copy()
# replace_map = {'Gender': {'Male': 0,'Female': 1 },
#                         'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
#                         'type_of_travel': {'Personal Travel': 1,'Business travel': 2},
#                         'customer_class': {'Eco': 1,'Eco Plus': 2 , 'Business': 3},
#                         'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
# }
# df3.replace(replace_map, inplace=True)



# dd_sat = df3[['departure_delay_in_minutes','arrival_delay_in_minutes','satisfaction']]

# plt.figure(figsize=(6, 6))
# sns.heatmap(dd_sat.corr(), annot=True)
# sns.set_context(font_scale=10)
# plt.title('Relationships between delays and satisfection',fontdict=dict(size=18))
# plt.show()


#%%
################ Numerical correlation
numerical_features =df[['age','flight_distance','departure_delay_in_minutes','arrival_delay_in_minutes','customer_type','satisfaction']].copy()
replace_map = {'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
               'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}
numerical_features.replace(replace_map, inplace=True)
corr_numerical = numerical_features.corr()

ax2 = sns.heatmap(corr_numerical,annot=True)
sns.set_context(font_scale=10)
plt.title('Relationships between numerical features')
plt.show()
#This is about numerical correlation.


print("From the correlation matrix it doesn't seems there's much effect on the delays\
 and passenger satisfection. It could be because the delays are mostly not controlled by\
 the airlines but airport or weather condition.")


#%%
print("Now we'd like to find out if there can differentiate the delays and satisfaction by categorizing delay ranges.")
#########################departure_delay_in_minutes - satisfaction
dep_sat = df[['departure_delay_in_minutes','satisfaction']].copy()
# dep_sat

dep_sat = df[['departure_delay_in_minutes','satisfaction']].copy()
dep_sat['dep_group'] = 0
# manual cut
bins = pd.IntervalIndex.from_tuples([(0,0.9), (1, 30), (31, 60), (61, 1600)],closed='left')
dep_group = pd.cut(x=dep_sat.departure_delay_in_minutes.to_list(),bins=bins)
dep_group.categories = ['0', '1-30', '31-60','>60']
dep_sat['dep_group'] = dep_group
# dep_sat

sns.countplot(x='dep_group', hue='satisfaction', data=dep_sat)
plt.xlabel("Departure delays")
plt.title("Departure Delays vs Staisfaction")
plt.show()

print("For this step, firstly we divided departure delay time into four groups which are not delay, shorter\
 than half an hour, between 30 to 60 minutes, and longer than 1 hour. Then we did barplot to show the\
 proportion of passenger's satisfaction in each branch. We found that in each branch the proportion is similar,\
 so which means departure is influencital factor. The reason is mentioned in correlation matrix. Delay may be \
 caused by extreme weather. Extreme weather will cause many flight delays in the airport, not only this flight. \
 So majority of people know that it is not airplines' fault to delay the flight. ")
#%%
############################# arrival_delay_in_minutes - satisfaction

print("Now, let's observe the arrival delay's.")

arr_sat = df[['arrival_delay_in_minutes','satisfaction']].copy()
# arr_sat

arr_sat = df[['arrival_delay_in_minutes','satisfaction']].copy()

arr_sat['arr_group'] = 0
# manual cut
bins = pd.IntervalIndex.from_tuples([(0,0.9), (1, 30), (31, 60), (61, 1600)],closed='left')
arr_group = pd.cut(x=arr_sat.arrival_delay_in_minutes.to_list(),bins=bins)
arr_group.categories = ['0', '1-30', '31-60','>60']
arr_sat['arr_group'] = arr_group
arr_sat

sns.countplot(x='arr_group', hue='satisfaction', data=arr_sat)
plt.xlabel("Arrival delays")
plt.title("Arrival Delays vs Staisfaction")
plt.show()

print("Again, the result is the same as previous.")
# Same as departure delay time part. 

##################################################################################
# EDA 2
##################################################################################
#%%
#### 1.2. Which age group is traveling more frequently and how much satisfied they are?
### age-satisfaction

print("Our next SMART question is Which age group is traveling more frequently and how much satisfied they are? ")

print("At first, making age-group by dividing ages to several intervals")
df3=df.copy()
replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 1,'Business travel': 2},
                        'customer_class': {'Eco': 1,'Eco Plus': 2 , 'Business': 3},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}
df3.replace(replace_map, inplace=True)


ag_sat = df3[['age','satisfaction']].copy()
# ag_sat['age_group'] = pd.cut(ag_sat['age'], 3)

ag_sat['age_group'] = 0
# manual cut
bins = pd.IntervalIndex.from_tuples([(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90)],
                                    closed='left')
age_group = pd.cut(x=ag_sat.age.to_list(),bins=bins)
age_group.categories = ["0-10", '11-20', '21-30', '31-40','41-50','51-60','61-70','71-80','81-90']
ag_sat['age_group'] = age_group


sns.countplot(x ='age_group',hue='satisfaction', palette=['r','b'], data = ag_sat)
plt.title("Age groups by satisfaction")
plt.show()

print("I looks like most travellers are from age between 21 to 60.\
 Highest number of travellers are of age 21-30 but most of them are not satisfied.\
 The second highest travellers are of 41-50 but most of them are stisfied of the airline service. ")

# %%
## CHI-TEST age-satisfaction
print("Now I'm going to run a Chi-Squared test of independence to determine whether there\
 is an association between age_groups and satisfaction.\n")

from scipy.stats import chi2_contingency
table = pd.crosstab(ag_sat.age_group, ag_sat.satisfaction)
csq=chi2_contingency(table)
# print(table)
print(csq)
print("\nThe second value of the above test output", csq[1],\
 "represents the p-value of the test. As evident, the p-value is\
 less than 0.05 and very significant, hence we reject the null hypothesis that satisfaction is\
 not associated with age groups and conclude that there is some\
 relationship between them.")

#%%

# finding percentage of satisfaction in the groups
# % of satisfaction = ((no. of satisfied)/(row total saitsfied+unsatisfied)) * 100%

per_sat=table.copy()
per_sat["total"] = per_sat.sum(axis=1)
per_sat['satisfaction_rate'] = (per_sat[1]/per_sat['total'])*100
per_sat['unsatisfaction_rate'] = (per_sat[0]/per_sat['total'])*100

## plot
pt=per_sat[['satisfaction_rate','unsatisfaction_rate']]
pt.plot(kind='bar', color=['#00B2FF','#CC3351'])
plt.title("Age groups satisfaction rates")
plt.ylabel("Percentage of satisfaction(%)")
plt.legend()
plt.show()

print("The highest rate of neutral/dissatisfaction is from children groups. Even though airplane isn't a child friendly\
 place but airlines should find some way to improve their satisfaction. On the other side, 70+ aged passengers are\
 also having a great percentage of dissatisfaction. Airlines should take a look at them as well.")



#%%
#################################################################################
# EDA 3
#################################################################################
# 1.3. Which ticket class has more satisfaction?
# 

print("Our 3rd SMART question is Which ticket class has more satisfaction?")

cc_sat = df3[['customer_class','satisfaction']]

# count plot
sns.countplot(x ='customer_class',hue='satisfaction', palette=['#B70013','#1200A7'], data = df_vis)
plt.title("Passenger Travel Class by satisfaction")
plt.show()

print("Looks like economy class passengers are not that stisfied with the airlines service.")

#%%

table = pd.crosstab(cc_sat.customer_class, cc_sat.satisfaction)

# finding percentage of satisfaction in the groups
# % of satisfaction = ((no. of satisfied)/(row total saitsfied+unsatisfied)) * 100%

cc_per_sat=table.copy()
cc_per_sat["total"] = cc_per_sat.sum(axis=1)
cc_per_sat['satisfaction_rate'] = (cc_per_sat[1]/cc_per_sat['total'])*100
cc_per_sat['unsatisfaction_rate'] = (cc_per_sat[0]/cc_per_sat['total'])*100

#%%
## plot
cc_pt=cc_per_sat[['satisfaction_rate', 'unsatisfaction_rate']]
cc_pt.plot(kind='bar',color=['#3AB6F5','#CF3057'])
plt.title("Passenger Satisfaction Rates by Travel Class")
plt.ylabel("Percentage of Satisfaction(%)")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

print("Here we can see, around 80% economy class passengers are not so satisfied and\
 more than 70% Business class passengers are mostly satisfied.\
 Also, as we observed before, economy plus passengers are least in numbers but now we can see only less than 30% of them are satisfied.\
 Airlines needs to make economy class better also improve themself more to satisfy business passengers as well because they are paying more than others")

#%%
#################################################################################
# EDA 4
#################################################################################
# 1.4. Who are more satisfied? Male or Female?

print("Our last SMART question for EDA is Who are more satisfied? Male or Female?")
# count plot
sns.countplot(x ='Gender',hue='satisfaction', data = df_vis, palette=['#E12C1E','#2780D8'])
plt.title("Satisfaction by Male/Female")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

print("Overall female seems to be less satisfied than men. Now let's explore it by their ages\
 to find out if all ages are having equal satisfaction among both male and female.")

#%%
mf_sat = df_vis[['Gender','age','satisfaction']]
mf_sat_ = mf_sat[mf_sat['satisfaction'] == 'satisfied']

# count plot
sns.kdeplot(
    data=mf_sat_,
    x="age", hue="Gender", shade = True,palette=['r','b'],
)
plt.title("Satisfied")
plt.show()

print("Looks like both male and female of same ages are equally satisfied.")

mf_unsat = mf_sat[mf_sat['satisfaction'] == 'neutral or dissatisfied']

# count plot
sns.kdeplot(
    data=mf_unsat,
    x="age", hue="Gender", shade = True,palette=['red','green'],
)
plt.title("Neutral or Dissatisfied")
plt.show()

print("Here looks like more females of age between 20-30 are a bit more unsatisfied than males.")


#%%

############################################################################################################
# Exploring through classification and modeliing                                                           #
############################################################################################################

# 1. What are the factors that satisfy the passenger?

print("Our goal is to find out the features that are most important for customer satisfaction\
 For that we are building model through dicision tree classifiers.")

print("Before that let's look into the features and their overall ratings.")


df_rating=df_rating.sort_values('Rate')

plt.plot(df_rating['Rate'],df_rating['Feature'])
plt.title('Features Sorted by Ratings:')
plt.xlabel("Ratings")
plt.xticks(rotation=90)
plt.show()

print("Here we can see that the highest rating is of inflight service and baggage handling.\
 But looking at the rate it's only slightly above 3.6 which is neutral type.")

print("Now, let's build some tree's and ovserve the feature importances.")

#%%

# For decision tree we need the values to be numeric. So, we are converting the following features to numeric.
# 'Gender': {'Male': 0,'Female': 1 }
# 'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1}
# 'type_of_travel': {'Personal Travel': 0,'Business travel': 1}
# 'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2}
# 'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}

dfn=df.copy()
replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
                        'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}

dfn.replace(replace_map, inplace=True)

from sklearn.model_selection import train_test_split

cols = dfn.iloc[:,:-1]
x = cols.values
y = dfn['satisfaction'].values

# Train-Test split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)
print("The training data size is : {} ".format(x_train.shape))
print("The test data size is : {} ".format(x_test.shape))

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
#%%
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

print("Online boarding is having the highest imporance follwed by onboard wifi service.")

#####################################################################################################
# TREE 2
#####################################################################################################
#%%

print("From here let's build a new model with the top 6 important features.")


cols = dfn[['online_boarding', 'inflight_wifi_service', 'type_of_travel', 'inflight_entertainment', 'customer_type', 'checkin_service']]
x = cols.values
y = dfn['satisfaction'].values

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

print("Among the top 6 features checkin_service is having the least importance.")

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


cols = dfn[['inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking', 'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort','inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'inflight_service', 'cleanliness']]
x = cols.values
y = dfn['satisfaction'].values

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
cols = dfn[['inflight_wifi_service', 'departure_arrival_time_convenient', 'online_boarding', 'inflight_entertainment', 'leg_room_service']]
x = cols.values
y = dfn['satisfaction'].values

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
print("Classification report:\n",classification_report(y_test, y_test_pred))

# Tree depth & leafs
print ('Tree Depth:', tree4.get_depth())
print ('Tree Leaves:', tree4.get_n_leaves())

print("The train accuracy is 91.84% but the test set accuracy is around 90.9%.\
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

print("Overall, online boarding and inflight wifi service are covering the major importance.\
Together they are covering more than 70% of importance. ")

#%%
# ROC-AUC
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = clf4.predict_proba(x_test)
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

print("The score is indicating this model is a very good model to predict customer satisfaction.")
#%%
############################################################################
# TREE 5
############################################################################

print("Now we are making a final tree with depth 3 just for visualization purpose.")

# cols = df2[['inflight_wifi_service', 'departure_arrival_time_convenient', 'gate_location', 'online_boarding', 'inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'cleanliness']]
cols = dfn[['inflight_wifi_service', 'departure_arrival_time_convenient', 'online_boarding', 'inflight_entertainment', 'leg_room_service']]
x = cols.values
y = dfn['satisfaction'].values

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
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,cmap="Reds", values_format=""))
plt.title("Confusion Matrix of Test Data ")
plt.show()
print("Classification report:\n",classification_report(y_train, y_train_pred))

#%%

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix: \n",ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred,cmap="Blues"))
plt.title("Confusion Matrix of Test Data ")
plt.show()
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
# The most important features to focus on is Online Boarding and Inflight Wi-Fi service.
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

#%%

############################################################################################################
# Exploring through classification and modeliing                                                           #
############################################################################################################

# 2. What factors mostly lead to loyal customers? 

print("Now we want to find out the loyal customers. What features impresses them that they prefer\
this airlines. We are again doing decision tree\
 classifier to find what features make loyal customers.")

# #%%
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import classification_report
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sklearn.tree import export_graphviz  


# #%%
# dirpath = os.getcwd() 
# filepath = os.path.join( dirpath ,'airline_passenger_satisfaction.csv')
# df = pd.read_csv(filepath, index_col=[0]) 


# # %%
# print("data shape: ")
# df.shape
# df.dtypes


# # %%
# #df.describe()
# df.head()

# # %%
# #check na values and delete na values
# df.isnull().sum()
# df=df.dropna()

# %%
# smart question 3: What factors have a strong correlation to loyal customers? 

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

#################################################################################################
#          CONCLUSION                                                                           #
#################################################################################################
#%%[markdown]
# Conclusions:
#
# We found out from the EDA that:-
#
# * Most satisfied age group is 41-50.
# * Airlines should make flight more comfortable for children and 70+ passengers
# * Female age between age 20-30 seems less satisfied than male
# * Arrival/departure delays have small effect on passenger satisfaction
# * Airlines needs to make economy class better
# * Airlines need to improve themself to satisfy business passengers as well because they are paying more than others
# * Overall ratings aren't that satisfactory as all of them are somewhat neutral. 
# * Inflight WiFi service is having the lowest rating and inflight service the highest.
#
#
# Form our Classification Modellings we can conclude that:-
#
# * Online boarding and inflight wifi service are covering the major importance on customer satisfaction.
# Airlines should highly focus on inflight Wi-Fi experience as it has lowest rating but high influence on satisfaction..
# * Departure/Arrival time convenient and ease of online booking makes more loyal customer.
# Airlines should keep improving these features to hold onto the loyal customers.
#
# Best model to predict customer satisfaction: Decision tree with top 5 reviews, 
# * online_boarding
# * inflight_wifi_service
# * leg_room_service
# * departure_arrival_time_convenient
# * inflight_entertainment
# 
# Both train and test accuracy is around 90%.
#
# Best model to predict loyal customers: Decision tree with 5 features and depth predefined as 8.
# * departure_arrival_time_convenient
# * ease_of_online_booking
# * online_boarding
# * gate_location
# * checkin_service
#
# Both train and test accuracy is around 87%.


#%%
##################################################################################################