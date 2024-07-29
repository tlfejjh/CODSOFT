import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Load MovieLens dataset
# Added encoding parameter to handle different file encodings
movies = pd.read_csv('/content/drive/MyDrive/Datasets/movies.csv') # Try 'latin-1' or 'ISO-8859-1' if 'latin-1' doesn't work
ratings = pd.read_csv('/content/drive/MyDrive/Datasets/ratings.csv') # Try 'latin-1' or 'ISO-8859-1' if 'latin-1' doesn't work

# Merge movies and ratings datasets
data = pd.merge(ratings, movies, on='id')

# Create a user-item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
user_movie_matrix.fillna(0, inplace=True)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to get movie recommendations for a user
def get_recommendations(user_id, user_movie_matrix, user_similarity_df, num_recommendations=5):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]

    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Initialize a dictionary to store the weighted sum of ratings
    weighted_ratings = {}

    # Loop through similar users
    for similar_user in similar_users.index:
        if similar_user == user_id:
            continue

        # Get the similarity score
        similarity_score = similar_users[similar_user]

        # Get the similar user's ratings
        similar_user_ratings = user_movie_matrix.loc[similar_user]

        # Calculate the weighted sum of ratings
        for movie, rating in similar_user_ratings.items():
            if movie not in weighted_ratings:
                weighted_ratings[movie] = 0
            weighted_ratings[movie] += similarity_score * rating

    # Convert the weighted ratings to a series
    weighted_ratings_series = pd.Series(weighted_ratings)

    # Normalize the ratings by the sum of similarities
    weighted_ratings_series /= similar_users.sum()

    # Filter out movies the user has already rated
    recommendations = weighted_ratings_series[user_ratings == 0]

    # Sort and return the top recommendations
    return recommendations.sort_values(ascending=False).head(num_recommendations)

# Example: Get recommendations for user 1
user_id = 1
recommendations = get_recommendations(user_id, user_movie_matrix, user_similarity_df)
print(f"Top recommendations for user {user_id}:\n{recommendations}")
