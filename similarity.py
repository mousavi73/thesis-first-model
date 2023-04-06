import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
from sklearn.model_selection import train_test_split
from math import sqrt

## __________load datasets__________
movies = pd.read_csv("checkins.csv", header=None, names=['user', 'check-in time', 'latitude', 'longitude', 'location id'])
movies_df =pd.DataFrame(movies)
movies_df['check-in time'] = movies_df['check-in time'].map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
# print(movies_df)
# print("___________________________________________")

## __________creating mean of latitude and longitude for users__________
# lat_long_mean_users = movies_df.groupby('user').mean()[['latitude', 'longitude']]
# print(lat_long_mean_users)
# print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

## __________find visits for all locations__________
places = movies_df.groupby('location id').agg(['mean', 'count'])[['latitude', 'longitude']]
places.columns = ['lat_mean', 'count_drop', 'long_mean', 'visits']
places = places.drop(columns=['count_drop'])
# # print(places.sort_values('visits'))
#
sorted_places = pd.DataFrame(places.sort_values('visits', ascending=False).reset_index())
# print(sorted_places)
# print("___________________________________________")



movie_data = pd.merge(movies_df, sorted_places, on='location id')
# print(movie_data)
# # split dataset
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


# ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
# ratings_mean_count['rating_counts']=pd.DataFrame(movie_data.groupby('title')['rating'].count())
# ratings_mean_count.head()

# __________finding all the values in the rating column__________
# print(new_table.visits.unique())


# __________filling the missing values in the pivot table with 0 and converting to DataFrame __________
df = pd.DataFrame(pd.pivot_table(X_train, index='user', columns='location id', values='visits', fill_value=0))
#print(df)
# # print("___________________________________________")



def similarity_score(person1, person2):
    # this Returns the ration euclidean distancen score of person 1 and 2

    # To get both rated items by person 1 and 2
    both_viewed = {}

    for item in df[person1]:
        if item in df[person2]:
            both_viewed[item] = 1

        # The Conditions to check if they both have common rating items
        if len(both_viewed) == 0:
            return 0

        # Finding Euclidean distance
        sum_of_eclidean_distance = []

        for item in df[person1]:
            if item in df[person2]:
                sum_of_eclidean_distance.append(pow(df[person1][item] - df[person2][item], 2))
        sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

        return 1 / (1 + sqrt(sum_of_eclidean_distance))

def person_correlation(person1, person2):
    # To get both rated items
    both_rated = {}
    for item in df[person1]:
        if item in df[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    # Checking for ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    person1_preferences_sum = sum([df[person1][item] for item in both_rated])
    person2_preferences_sum = sum([df[person2][item] for item in both_rated])

    # Sum up the squares of preferences of each user
    person1_square_preferences_sum = sum([pow(df[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(df[person2][item], 2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([df[person1][item] * df[person2][item] for item in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (
                person1_preferences_sum * person2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
                person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))

    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r


def most_similar_users(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person
    scores = [(person_correlation(person, other_person), other_person) for other_person in df if
              other_person != person]

    # Sort the similar persons so the highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]


def user_recommendations(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in df:
        # don't compare me to myself
        if other == person:
            continue
        sim = person_correlation(person, other)
        # print ">>>>>>>",sim

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in df[other]:

            # only score movies i haven't seen yet
            if item not in df[person] or df[person][item] == 0:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] += df[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

        # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score, recommend_item in rankings]
    return recommendataions_list


print(user_recommendations(1))