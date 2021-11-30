
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
#################################################################################
# EDA 1 Zihan YE
#################################################################################
# 1.1. Does arrival/departure have any effect on customer satisfaction?
# departure_delay_in_minutes - satisfaction
#
df3=df.copy()
replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 1,'Business travel': 2},
                        'customer_class': {'Eco': 1,'Eco Plus': 2 , 'Business': 3},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}
df3.replace(replace_map, inplace=True)



dd_sat = df3[['departure_delay_in_minutes','arrival_delay_in_minutes','satisfaction']]

plt.figure(figsize=(6, 6))
sns.heatmap(dd_sat.corr(), annot=True)
sns.set_context(font_scale=10)
plt.title('Relationships between delays and satisfection',fontdict=dict(size=18))
plt.show()

print("From the correlation matrix it doesn't seems there's much effect on the delays\
 and passenger satisfection. It could be because the delays are mostly not controlled by\
 the airlines but airport or weather condition.")

#### more from Zihan Ye
#%%
################Numerical correlation
numerical_features =df[['age','flight_distance','departure_delay_in_minutes','arrival_delay_in_minutes','satisfaction','customer_type']]
replace_map = {'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
               'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}
numerical_features.replace(replace_map, inplace=True)
corr_numerical = numerical_features.corr()

# ax2 = sns.heatmap(corr_numerical,annot=True)
# sns.set_context(font_scale=10)
# plt.title('Relationships between numerical features')
# plt.show()
# #%%
# ################Categorical correlation
# categorical_features=df.copy()
# replace_map = {'Gender': {'Male': 0,'Female': 1 },
#                         'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
#                         'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
#                         'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
#                         'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
# }

# categorical_features.replace(replace_map, inplace=True)
# print(categorical_features.dtypes)

# corr_categorical = categorical_features.corr()
# ax2 = plt.subplots(figsize=(10,10))
# ax2 = sns.heatmap(corr_categorical,annot=True,annot_kws={"fontsize":8})
# sns.set_context(font_scale=10)
# plt.title('Relationships between categorical features')
# plt.show()
# #%%
# #########################departure_delay_in_minutes - satisfaction
# dep_sat = df[['departure_delay_in_minutes','satisfaction']].copy()
# dep_sat
# # %%
# dep_sat = df[['departure_delay_in_minutes','satisfaction']].copy()

# dep_sat['dep_group'] = 0
# # manual cut
# bins = pd.IntervalIndex.from_tuples([(0,0.9), (1, 30), (31, 60), (61, 1600)],closed='left')
# dep_group = pd.cut(x=dep_sat.departure_delay_in_minutes.to_list(),bins=bins)
# dep_group.categories = ['0', '1-30', '31-60','>60']
# dep_sat['dep_group'] = dep_group
# dep_sat
# # %%
# sns.countplot(x='dep_group', hue='satisfaction', data=dep_sat)
# #%%
# ############################# arrival_delay_in_minutes - satisfaction
# arr_sat = df[['arrival_delay_in_minutes','satisfaction']].copy()
# arr_sat
# # %%
# arr_sat = df[['arrival_delay_in_minutes','satisfaction']].copy()

# arr_sat['arr_group'] = 0
# # manual cut
# bins = pd.IntervalIndex.from_tuples([(0,0.9), (1, 30), (31, 60), (61, 1600)],closed='left')
# arr_group = pd.cut(x=arr_sat.arrival_delay_in_minutes.to_list(),bins=bins)
# arr_group.categories = ['0', '1-30', '31-60','>60']
# arr_sat['arr_group'] = arr_group
# arr_sat
# # %%
# sns.countplot(x='arr_group', hue='satisfaction', data=arr_sat)

##################################################################################
# EDA 2
##################################################################################
#%%
#### 1.2. Which age group is traveling more frequently and how much satisfied they are?
### age-satisfaction
##  making age-group by dividing ages to several intervals
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

#%%

##################################################################################
# EDA 3
##################################################################################

## 1.3. Which age group is travelling more?
# count plot
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
# EDA 4
#################################################################################
# 1.4. Which ticket class has more satisfaction?
# 

cc_sat = df3[['customer_class','satisfaction']]

# count plot
sns.countplot(x ='customer_class',hue='satisfaction', palette=['#B70013','#1200A7'], data = df_vis)
plt.title("Passenger Travel Class by satisfaction")
plt.show()

print("Looks like economy class passengers are not that stisfied with the airlines service")

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
# EDA 5
#################################################################################
# 1.5. Who are more satisfied? Male or Female?

# count plot
sns.countplot(x ='Gender',hue='satisfaction', data = df_vis, palette=['#E12C1E','#2780D8'])
plt.title("Satisfaction by Male/Female")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()

print("Overall female seems to be less satisfied than men. Now let's explore it by their ages.")

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
# (Nawshin parts)

#%%

############################################################################################################
# Exploring through classification and modeliing                                                           #
############################################################################################################

# 2. What factors have a strong correlation to loyal customers? 

# (Jinbo part)

#%%