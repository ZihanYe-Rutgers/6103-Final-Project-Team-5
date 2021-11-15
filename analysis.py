
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
df.dtypes

#%%
# top 5 data from data dataframe
df.head()


#%%
# checking which columns have null values and count of nulls
df.isnull().sum()
# we can drop thosse null rows in the arrival_delay_in_minutes column

#%%
# Gender - categorical
print("Lets inspect the Gender column: ")
print(df.Gender.describe())
print(df.Gender.value_counts())

#%%
# customer_type - categorical

print("Lets inspect the customer_type column: ")
print(df.customer_type.describe())
print(df.customer_type.value_counts())

#%%
# age - numeric
print("Lets inspect the age column: ")
print(df.age.describe())
print(df.age.value_counts())

#%%
# type_of_travel - categorical

print("Lets inspect the type_of_travel column: ")
print(df.type_of_travel.describe())
print(df.type_of_travel.value_counts())

#%%
# customer_class - categorical ordered?
print("Lets inspect the customer_class column: ")
print(df.customer_class.describe())
print(df.customer_class.value_counts())

#%%
# flight_distance - numeric
print("Lets inspect the flight_distance column: ")
print(df.flight_distance.describe())
print(df.flight_distance.value_counts())

#%%
# inflight_wifi_service - ordered categorical
print("Lets inspect the inflight_wifi_service column: ")
print(df.inflight_wifi_service.describe())
print(df.inflight_wifi_service.value_counts())

#%%
# departure_arrival_time_convenient - ordered categorical
print("Lets inspect the departure_arrival_time_convenient column: ")
print(df.departure_arrival_time_convenient.describe())
print(df.departure_arrival_time_convenient.value_counts())

#%%
# ease_of_online_booking - ordered categorical
print("Lets inspect the ease_of_online_booking column: ")
print(df.ease_of_online_booking.describe())
print(df.ease_of_online_booking.value_counts())

#%%
# gate_location  - ordered categorical
print("Lets inspect the gate_location column: ")
print(df.gate_location.describe())
print(df.gate_location.value_counts())

#%%
# food_and_drink - ordered categorical

print("Lets inspect the food_and_drink column: ")
print(df.food_and_drink.describe())
print(df.food_and_drink.value_counts())

#%%
# online_boarding - ordered categorical
print("Lets inspect the online_boarding column: ")
print(df.online_boarding.describe())
print(df.online_boarding.value_counts())

#%%
# seat_comfort - ordered categorical
print("Lets inspect the seat_comfort column: ")
print(df.seat_comfort.describe())
print(df.seat_comfort.value_counts())

#%%
# inflight_entertainment - ordered categorical
print("Lets inspect the Gender column: ")
print(df.inflight_entertainment.describe())
print(df.inflight_entertainment.value_counts())

#%%
# onboard_service - ordered categorical

print("Lets inspect the onboard_service column: ")
print(df.onboard_service.describe())
print(df.onboard_service.value_counts())

#%%
# leg_room_service - ordered categorical
print("Lets inspect the leg_room_service column: ")
print(df.leg_room_service.describe())
print(df.leg_room_service.value_counts())

#%%
# baggage_handling- ordered categorical
print("Lets inspect the baggage_handling column: ")
print(df.baggage_handling.describe())
print(df.baggage_handling.value_counts())

#%%
# checkin_service - ordered categorical
print("Lets inspect the Gender column: ")
print(df.checkin_service.describe())
print(df.checkin_service.value_counts())

#%%
# inflight_service - ordered categorical
print("Lets inspect the inflight_service column: ")
print(df.inflight_service.describe())
print(df.inflight_service.value_counts())

#%%
# cleanliness - ordered categorical
print("Lets inspect the cleanliness column: ")
print(df.cleanliness.describe())
print(df.cleanliness.value_counts())

#%%
# departure_delay_in_minutes - numeric
print("Lets inspect the departure_delay_in_minutes column: ")
print(df.departure_delay_in_minutes.describe())
print(df.departure_delay_in_minutes.value_counts())

#%%
# arrival_delay_in_minutes - numeric
print("Lets inspect the arrival_delay_in_minutes column: ")
print(df.arrival_delay_in_minutes.describe())
print(df.arrival_delay_in_minutes.value_counts())

#%%
# satisfaction - ordered? categorical
print("Lets inspect the satisfaction column: ")
print(df.satisfaction.describe())
print(df.satisfaction.value_counts())

# %%
