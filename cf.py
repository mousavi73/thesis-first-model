import numpy as np
import pandas as pd
from datetime import datetime
from math import sqrt
#import matplotlib.pyplot as plt


## __________load datasets__________
# data = pd.read_csv("checkins.csv", header = None, names = ['user', 'check-in time', 'latitude', 'longitude', 'location id'])
# dataset_df =pd.DataFrame(data)
# print(dataset_df)

## __________creating dataframes__________
# edges = pd.read_csv('Gowalla_edges.txt', sep='\t', names=['u1', 'u2'])
# edges_df =pd.DataFrame(edges)
# print(edges_df)

Checkins = pd.read_csv("Gowalla_totalCheckins.txt", sep='\t', header = None, names=['user', 'check-in time', 'latitude', 'longitude', 'location id'])
Checkins_df =pd.DataFrame(Checkins)
Checkins_df['check-in time'] = Checkins_df['check-in time'].map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
# print(Checkins_df)

## __________ creating mean of latitude and longitude for users__________
# lat_long_mean_users = Checkins_df.groupby('user').mean()[['latitude', 'longitude']]
# print(lat_long_mean_users)

# __________find visits for all locations__________
places = Checkins_df.groupby('location id').agg(['mean', 'count'])[['latitude', 'longitude']]
places.columns = ['lat_mean', 'count_drop', 'long_mean', 'visits']
places = places.drop(columns = ['count_drop'])
# print(places.sort_values('visits'))
sorted_places = places.sort_values('visits', ascending=False).reset_index()
# sorted_places[:10000]
# print(sorted_places[:10000])

# __________custom function to create unique set of web series__________
# def unique_items():
#     unique_items_list = []
#     for person in Checkins_df['user', 'visits']:
#         for items in Checkins_df[person]:
#             unique_items_list.append(items)
#     unique_items_list = list(set(unique_items_list))
#     print(unique_items_list)

# __________custom function to create pearson correlation method from scratch__________
# def person_corelation(person1,person2):
#     both_rated = {}
#     for item in Checkins_df[person1]:
#         if item in Checkins_df[person2]:
#             both_rated[item] = 1
#
#     number_of_ratings = len(both_rated)
#     if number_of_ratings == 0:
#         return 0
#
#     person1_preferences_sum = sum([Checkins_df[person1][item] for item in both_rated])
#     person2_preferences_sum = sum([Checkins_df[person2][item] for item in both_rated])
#
#     # __________Sum up the squares of preferences of each user__________
#     person1_square_preferences_sum = sum([pow(Checkins_df[person1][item], 2) for item in both_rated])
#     person2_square_preferences_sum = sum([pow(Checkins_df[person2][item], 2) for item in both_rated])
#
#     # __________Sum up the product value of both preferences for each item__________
#     product_sum_of_both_users = sum([Checkins_df[person1][item] * Checkins_df[person2][item] for item in both_rated])
#
#     # __________Calculate the pearson score__________
#     numerator_value = product_sum_of_both_users - (person1_preferences_sum * person2_preferences_sum / number_of_ratings)
#     denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))
#     if denominator_value == 0:
#         return 0
#     else:
#         r = numerator_value / denominator_value
#         return r

# #__________ Phase1 - User Similarity by using Pearson Correlation method__________
# # custom function to check most similar users
#
#     def most_similar_users(target_person, no_of_users):
#
# # __________Used list comprehension for finding pearson similarity between users__________
#         scores = [(person_corelation(target_person, other_person), other_person) for other_person in Checkins_df if other_person != target_person]
#
# # __________sort the scores in descending order__________
#         scores.sort(reverse=True)
#
# # __________return the scores between the target person & other persons__________
#         return scores[0:no_of_users]
#
# # __________function check by input one person name & returns the similarity score__________
#     print( most_similar_users('2', 6))

#     # __________custom function to filter the seen movies and unseen movies of the target user__________
#
#     def target_movies_to_users(target_person):
#         target_person_movie_lst = []
#         unique_list = unique_items()
#         for movies in dataset[target_person]:
#             target_person_movie_lst.append(movies)
#
#         s = set(unique_list)
#         recommended_movies = list(s.difference(target_person_movie_lst))
#         a = len(recommended_movies)
#         if a == 0:
#             return 0
#         return recommended_movies, target_person_movie_lst
#
#     # __________function check__________
#
#     unseen_movies, seen_movies = target_movies_to_users('Nirbhay')
#
#     dct = {"Unseen Movies": unseen_movies, "Seen Movies": seen_movies}
#     pd.DataFrame(dct)