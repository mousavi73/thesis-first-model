import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
from datetime import datetime
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
from sklearn.model_selection import train_test_split

## __________load datasets__________
data = pd.read_csv("checkins.csv",sep= '\t' , header=None, names=['user', 'check-in time', 'latitude', 'longitude', 'location id'])
data_df =pd.DataFrame(data)
data_df['check-in time'] = data_df['check-in time'].map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
print(data_df)
# print("___________________________________________")
# ______________________________________________

## __________creating mean of latitude and longitude for users__________
lat_long_mean_users = data_df.groupby('user').mean()[['latitude', 'longitude']]
print(lat_long_mean_users)
# print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

## __________creating mean of latitude and longitude for locations__________
lat_long_mean_locations = data_df.groupby('location id').mean()[['latitude', 'longitude']]
print(lat_long_mean_locations)
# print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

## __________find visits for all locations__________
places = data_df.groupby('location id').agg(['mean', 'count'])[['latitude', 'longitude']]
places.columns = ['lat_mean', 'count_drop', 'long_mean', 'visits']
places = places.drop(columns=['count_drop'])
# print(places.sort_values('visits'))
#
sorted_places = pd.DataFrame(places.sort_values('visits', ascending=False).reset_index())
# print(sorted_places)
# print("___________________________________________")



movie_data = pd.merge(data_df, sorted_places, on='location id')
# print(movie_data)

# split dataset
X_train, X_test = train_test_split(movie_data, test_size=0.8, random_state=1)
# print(X_train.shape, X_test.shape)
# new_table = pd.pivot_table(X_train, index='user', columns='location id', values='visits')
# out= print(new_table)
# file = open('out.txt', 'w')
# file.write(out)
# file.close()
# mean_visits = movie_data.groupby(['user', 'location id'])['visits'].mean().head()
# # mean_visits = movie_data.groupby(['user', 'location id'])['visits'].mean().sort_values(ascending=False).head()
# print(mean_visits)
# count_visits = movie_data.groupby(['user', 'location id'])['visits'].count().sort_values(ascending=False).head()
# print(count_visits)
#
#
# ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
# ratings_mean_count['rating_counts']=pd.DataFrame(movie_data.groupby('title')['rating'].count())
# ratings_mean_count.head()

# __________finding all the values in the rating column__________
# print(new_table.visits.unique())

# ***********
# __________filling the missing values in the pivot table with 0 and converting to DataFrame __________
df = pd.DataFrame(pd.pivot_table(X_train, index='user', columns='location id', values='visits', fill_value=0))
# print(df)
# print("___________________________________________")

# file = open('out.txt', 'w')
# file.write(pr_df)
# file.close()

# __________Q6. Find the 5 most similar user for user with user Id 10 __________
# __________first compute pairwise distances between the rows, that is the users, based on the cosine similarity __________
# __________computing pairwise distances using cosine similarity between the users __________

# distance_matrix = pairwise_distances(df, metric='cosine')
# ## __________finding the top 5 most similar users __________
# ## argsort()function on the array to return the indices
# print((distance_matrix[0]).argsort()[1:6] + 1)
# print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

# __________Q7.find out the names of movies, user 2 and user 338 have watched in common __________
# __________finding the intersection of the sets of the location that both users 2 and 338 have watched __________
print(set(df.loc[df.user==2, 'location id']).intersection(set(df.loc[df.user==5, 'location id'])))
print("___________________________________________")




## __________printing the ratings given by users __________
# print('location ID 318: ', data_df.loc[data_df.location id==318, 'location id'].values[0])
# print('visiting by User 2: ', data_df.loc[((data_df.user==2) & (data_df.location id==318)), 'visits'].values[0])
# print('visiting by User 338: ', data_df.loc[((data_df.user==338) & (data_df.location id==318)), 'visits'].values[0])
# print()
# print('location ID 6874: ', data_df.loc[data_df.location id==6874, 'location id'].values[0])
# print('Rating by User 2: ', data_df.loc[((data_df.user==2) & (data_df.movieId==6874)), 'visits'].values[0])
# print('Rating by User 338: ', data_df.loc[((data_df.user==338) & (data_df.movieId==6874)), 'visits'].values[0])
# print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

## __________Q8.find out the common location id between user 2 and user 338 with least rating of 4.0 __________

# set(data_df.loc[((data_df.userId==2) & (data_df.rating>=4.0)), 'movieId']).intersection(set(data_df.loc[
# ((data_df.userId==338) & (data_df.rating>=4.0)),'movieId']))
#
# # printing the title of the movie
# print('Movie ID 6874: ', data_df.loc[data_df.location id==6874, 'location id'].values[0])

## __________Q9. Create a pivot table for representing the similarity among movies using correlation# __________
## __________merging the dataframes ratings and movies__________
# df = data_df.merge(movies, on='location id')

# creating the pivot table and convert it to a DataFrame
# df = pd.DataFrame(pd.pivot_table(df, index='title', columns='userId', values='rating', fill_value=0))
# df

# As we can see from the pivot table above, the last movieId exceeds the number of rows. That means that the movieId
# and the row numbers are not the same. This is because only those movies who have atleast 1 rating or tag have been
# chosen.

# Q10.Find the top 5 movies which are similar to the movie “Godfather”.
# To solve Question 10, we first compute pairwise distances between the rows, that is the movies, based on the
# correlation.To do this, we use the pairwise_distances() function from the ScikitLearn library and set the value of
# the parameter metric to 'correlation'.

# computing pairwise distances using correlation between the movies
# distance_matrix = pairwise_distances(df, metric='correlation')
# Next, we find the movies which contain the word Godfather.To do this, we use string functions and filter the dataframe.
# finding the movies which contain the word Godfather

# df[df.index.str.contains('Godfather')]

# As we can see, there are 5 movies which contain the word Godfather.However, we choose the first movie - Godfather,
# The(1972), which is the oldest.
# Since the movieId and row number is not the same, we find the index of the Godfather movie.To do this,
# we first make a list of all the movies and then find the index of the Godfather movie in the list.

# finding index of Godfather movie

# indices = list(df.index)
# indices.index('Godfather, The (1972)')

# Next, we use the index previously found and splice the distance_matrix.Post that, we sort the correlations in the
# descending order to find the highest correlation between movies, and hence the most similar movies.We use the argsort()
# function on the array to return the indices.Then, we splice the array to find the top 5 similar movies.To do this,
# we splice values from 1 to 6. We leave out the index 0 because it is by default the same movie.Then, we print the
# movies
# by using the indices and finding their values in the indices list we previously made.

# finding the top 5 similar movies
# similar_indices = (distance_matrix[3499].argsort()[1:6])
# for i in similar_indices:
#     print(indices[i])

# As we can see, the movies Godfather: Part II, The(1974), Goodfellas(1990), One Flew Over the Cuckoo 's Nest (1975),
# Reservoir Dogs(1992), and Fargo(1996) are the most similar to Godfather, The(1972).