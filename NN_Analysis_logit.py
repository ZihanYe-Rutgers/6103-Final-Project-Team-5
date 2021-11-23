
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
# Now We are modeling with logistics regression model to find out the answer.

df3=df.copy()

replace_map = {'Gender': {'Male': 0,'Female': 1 },
                        'customer_type': {'disloyal Customer': 0,'Loyal Customer': 1},
                        'type_of_travel': {'Personal Travel': 0,'Business travel': 1},
                        'customer_class': {'Eco': 0,'Eco Plus': 1 , 'Business': 2},
                        'satisfaction': {'neutral or dissatisfied': 0,'satisfied': 1}
}

df3.replace(replace_map, inplace=True)


import statsmodels.api as sm
from statsmodels.formula.api import glm

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
 \n14. cleanliness\n")

# cols = df3[['inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking', 'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort','inflight_entertainment', 'onboard_service','leg_room_service','baggage_handling','checkin_service', 'inflight_service', 'cleanliness']]

#%%


model1LogitFit = glm(formula='satisfaction ~ inflight_wifi_service+departure_arrival_time_convenient+ease_of_online_booking+gate_location+food_and_drink+online_boarding+seat_comfort+inflight_entertainment+onboard_service+leg_room_service+baggage_handling+checkin_service+inflight_service+cleanliness', data=df3, family=sm.families.Binomial()).fit()
print( model1LogitFit.summary() )
modelpredicitons = pd.DataFrame( columns=['Logit'], data= model1LogitFit.predict(df3)) 
# print(modelpredicitons.head())
# dfChkBasics(modelpredicitons)

print("Observing the p-values, only inflight_service is not a good predictors \
as there p-values are greater than 0.05. On the other hand, departure_arrival_time_convenient,\
 ease_of_online_booking and food_and_drink features have a negative relationship with satisfection.\
 So, let's remove these features, and build a new model.")

#%%


model1LogitFit = glm(formula='satisfaction ~ inflight_wifi_service+gate_location+online_boarding+seat_comfort+inflight_entertainment+onboard_service+leg_room_service+baggage_handling+checkin_service+cleanliness', data=df3, family=sm.families.Binomial()).fit()
print( model1LogitFit.summary() )
modelpredicitons = pd.DataFrame( columns=['Logit'], data= model1LogitFit.predict(df3)) 
# print(modelpredicitons.head())
# dfChkBasics(modelpredicitons)

print("Observing the p-values, only cleanliness is not a good predictors\
 as there p-values are greater than 0.05. On the other hand, gate_location\
 has a negative relationship with satisfection.\
 So, let's remove these features, and build a new model.")

#%%


model1LogitFit = glm(formula='satisfaction ~ inflight_wifi_service+online_boarding+seat_comfort+inflight_entertainment+onboard_service+leg_room_service+baggage_handling+checkin_service', data=df3, family=sm.families.Binomial()).fit()
print( model1LogitFit.summary() )
modelpredicitons = pd.DataFrame( columns=['Logit'], data= model1LogitFit.predict(df3)) 
# print(modelpredicitons.head())
# dfChkBasics(modelpredicitons)

print("All the p-values are significant.")

#%%
model_odds = pd.DataFrame(np.exp(model1LogitFit.params), columns= ['OR'])
# model_odds2['z-value']= model2.pvalues
# model_odds2[['2.5%', '97.5%']] = np.exp(model2.conf_int())

print("Let's observe the growth/decay factors for each variable.\n",model_odds)

o=model_odds.iloc[0]["OR"]
p=o/(1+o)
print("\nThe odds of a satisfection is decreases by a factor of ", model_odds.iloc[2]["OR"]," for each score increase in review.")

print("The probability of satisfection is ", round(p*100, 2), '%')


#%%