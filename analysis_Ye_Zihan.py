
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


#%%

# For data visualization purpose we are convering our variables to appropiate data types \
#  i.e. numeric, categorical, ordered categorical and the dataframe name is df_vis

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
plt.title("Distributuion of age")
plt.show()

#%%
# flight_distance - numeric
print("Lets inspect the flight_distance column: ")
print(df_vis.flight_distance.describe())
print(df_vis.flight_distance.value_counts())
# histogram
sns.histplot(x ='flight_distance', data = df_vis)
plt.title("Distributuion of flight_distance")
plt.show()

#%%
# departure_delay_in_minutes - numeric
print("Lets inspect the departure_delay_in_minutes column: ")
print(df_vis.departure_delay_in_minutes.describe())
print(df_vis.departure_delay_in_minutes.value_counts())
# histogram
sns.histplot(x ='departure_delay_in_minutes', data = df_vis)
plt.title("Distributuion of departure_delay_in_minutes")
plt.show()

#%%
# arrival_delay_in_minutes - numeric
print("Lets inspect the arrival_delay_in_minutes column: ")
print(df_vis.arrival_delay_in_minutes.describe())
print(df_vis.arrival_delay_in_minutes.value_counts())
# histogram
sns.histplot(x ='arrival_delay_in_minutes', data = df_vis)
plt.title("Distributuion of arrival_delay_in_minutes")
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
plt.title("Distributuion of Gender")
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
plt.title("Distributuion of customer_type")
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
plt.title("Distributuion of type_of_travel")
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
plt.title("Distributuion of satisfaction")
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
plt.title("Distributuion of customer_class")
plt.show()


#%%
# Converting inflight_wifi_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_wifi_service'] = pd.Categorical(df_vis['inflight_wifi_service'], ordered=True)
# count plot
sns.countplot(x ='inflight_wifi_service', data = df_vis)
plt.title("Distributuion of inflight_wifi_service")
plt.show()

#%%
# departure_arrival_time_convenient - ordered categorical
print("Lets inspect the departure_arrival_time_convenient column: ")
print(df_vis.departure_arrival_time_convenient.describe())
print(df_vis.departure_arrival_time_convenient.value_counts())
# Converting departure_arrival_time_convenient to Ordered Categorical- 0,1,2,3,4,5
df_vis['departure_arrival_time_convenient'] = pd.Categorical(df_vis['departure_arrival_time_convenient'], ordered=True)
# count plot
sns.countplot(x ='departure_arrival_time_convenient', data = df_vis)
plt.title("Distributuion of departure_arrival_time_convenient")
plt.show()

#%%
# ease_of_online_booking - ordered categorical
print("Lets inspect the ease_of_online_booking column: ")
print(df_vis.ease_of_online_booking.describe())
print(df_vis.ease_of_online_booking.value_counts())
# Converting ease_of_online_booking to Ordered Categorical- 0,1,2,3,4,5
df_vis['ease_of_online_booking'] = pd.Categorical(df_vis['ease_of_online_booking'], ordered=True)
# count plot
sns.countplot(x ='ease_of_online_booking', data = df_vis)
plt.title("Distributuion of ease_of_online_booking")
plt.show()

#%%
# gate_location  - ordered categorical
print("Lets inspect the gate_location column: ")
print(df_vis.gate_location.describe())
print(df_vis.gate_location.value_counts())
# Converting gate_location to Ordered Categorical- 0,1,2,3,4,5
df_vis['gate_location'] = pd.Categorical(df_vis['gate_location'], ordered=True)
# count plot
sns.countplot(x ='gate_location', data = df_vis)
plt.title("Distributuion of gate_location")
plt.show()

#%%
# food_and_drink - ordered categorical

print("Lets inspect the food_and_drink column: ")
print(df_vis.food_and_drink.describe())
print(df_vis.food_and_drink.value_counts())
# Converting food_and_drink to Ordered Categorical- 0,1,2,3,4,5
df_vis['food_and_drink'] = pd.Categorical(df_vis['food_and_drink'], ordered=True)
# count plot
sns.countplot(x ='food_and_drink', data = df_vis)
plt.title("Distributuion of food_and_drink")
plt.show()

#%%
# online_boarding - ordered categorical
print("Lets inspect the online_boarding column: ")
print(df_vis.online_boarding.describe())
print(df_vis.online_boarding.value_counts())
# Converting online_boarding to Ordered Categorical- 0,1,2,3,4,5
df_vis['online_boarding'] = pd.Categorical(df_vis['online_boarding'], ordered=True)
# count plot
sns.countplot(x ='online_boarding', data = df_vis)
plt.title("Distributuion of online_boarding")
plt.show()

#%%
# seat_comfort - ordered categorical
print("Lets inspect the seat_comfort column: ")
print(df_vis.seat_comfort.describe())
print(df_vis.seat_comfort.value_counts())
# Converting seat_comfort to Ordered Categorical- 0,1,2,3,4,5
df_vis['seat_comfort'] = pd.Categorical(df_vis['seat_comfort'], ordered=True)
# count plot
sns.countplot(x ='seat_comfort', data = df_vis)
plt.title("Distributuion of seat_comfort")
plt.show()

#%%
# inflight_entertainment - ordered categorical
print("Lets inspect the Gender column: ")
print(df_vis.inflight_entertainment.describe())
print(df_vis.inflight_entertainment.value_counts())
# Converting inflight_entertainment to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_entertainment'] = pd.Categorical(df_vis['inflight_entertainment'], ordered=True)
# count plot
sns.countplot(x ='inflight_entertainment', data = df_vis)
plt.title("Distributuion of inflight_entertainment")
plt.show()

#%%
# onboard_service - ordered categorical

print("Lets inspect the onboard_service column: ")
print(df_vis.onboard_service.describe())
print(df_vis.onboard_service.value_counts())
# Converting onboard_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['onboard_service'] = pd.Categorical(df_vis['onboard_service'], ordered=True)
# count plot
sns.countplot(x ='onboard_service', data = df_vis)
plt.title("Distributuion of onboard_service")
plt.show()

#%%
# leg_room_service - ordered categorical
print("Lets inspect the leg_room_service column: ")
print(df_vis.leg_room_service.describe())
print(df_vis.leg_room_service.value_counts())
# Converting leg_room_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['leg_room_service'] = pd.Categorical(df_vis['leg_room_service'], ordered=True)
# count plot
sns.countplot(x ='leg_room_service', data = df_vis)
plt.title("Distributuion of leg_room_service")
plt.show()

#%%
# baggage_handling- ordered categorical
print("Lets inspect the baggage_handling column: ")
print(df_vis.baggage_handling.describe())
print(df_vis.baggage_handling.value_counts())
# Converting baggage_handling to Ordered Categorical- 0,1,2,3,4,5
df_vis['baggage_handling'] = pd.Categorical(df_vis['baggage_handling'], ordered=True)
# count plot
sns.countplot(x ='baggage_handling', data = df_vis)
plt.title("Distributuion of baggage_handling")
plt.show()

#%%
# checkin_service - ordered categorical
print("Lets inspect the Gender column: ")
print(df_vis.checkin_service.describe())
print(df_vis.checkin_service.value_counts())
# Converting checkin_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['checkin_service'] = pd.Categorical(df_vis['checkin_service'], ordered=True)
# count plot
sns.countplot(x ='checkin_service', data = df_vis)
plt.title("Distributuion of checkin_service")
plt.show()

#%%
# inflight_service - ordered categorical
print("Lets inspect the inflight_service column: ")
print(df_vis.inflight_service.describe())
print(df_vis.inflight_service.value_counts())
# Converting inflight_service to Ordered Categorical- 0,1,2,3,4,5
df_vis['inflight_service'] = pd.Categorical(df_vis['inflight_service'], ordered=True)
# count plot
sns.countplot(x ='inflight_service', data = df_vis)
plt.title("Distributuion of inflight_service")
plt.show()
#%%
# cleanliness - ordered categorical
print("Lets inspect the cleanliness column: ")
print(df_vis.cleanliness.describe())
print(df_vis.cleanliness.value_counts())
# Converting cleanliness to Ordered Categorical- 0,1,2,3,4,5
df_vis['cleanliness'] = pd.Categorical(df_vis['cleanliness'], ordered=True)
# count plot
sns.countplot(x ='cleanliness', data = df_vis)
plt.title("Distributuion of cleanliness")
plt.show()


# %% Then we will dod logistic regression and some prediction models to analyze numeric varibales.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.formula.api import glm
# %%
modelNumericLogit = glm(formula='satisfaction ~ age+ flight_distance + departure_delay_in_minutes + arrival_delay_in_minutes',
                            data=df, family=sm.families.Binomial())
modelNumericLogitFit = modelNumericLogit.fit()
print(modelNumericLogitFit.summary())
#We find that all numeric variables are significan because P-value is all extremely small.
# %% Exponential coefficient.
modelNumericLogitFit.params.apply(np.exp)
#From these results, we can say, for example:
# The increasing of age number of 1 will decrease the log ratio by a factor of 0.984774.
# The increasing of flight_distance of 1 will decrease the log ratio by a factor of 0.999377.
# The increasing of departure_delay_in_minutes of 1 minute will decrease the log ratio by a factor of 0.996530.
# The increasing of arrival_delay_in_minutes of 1 minute will decrease the log ratio by a factor of 1.006997.

# %% Prediction model
modelpredicitons1 = pd.DataFrame( columns=['satisfaction'], data= modelNumericLogitFit.predict(df)) 
print(modelpredicitons1.shape)
print( modelpredicitons1)
# %%

# %%
