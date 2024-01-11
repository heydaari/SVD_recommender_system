import numpy as np
import pandas as pd

def data_loader():
    # Load data from CSV files
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    # Create a user-item matrix (Ratings matrix)
    user_movie_ratings = pd.merge(ratings, movies, on='movieId')[['userId', 'movieId', 'rating']]

    # Create a pivot table for user-item matrix
    user_movie_ratings_pivot = user_movie_ratings.pivot_table(index='userId', columns='movieId', values='rating')

    # Fill NaN values with zeros (assuming no rating means a rating of zero)
    user_movie_ratings_matrix = user_movie_ratings_pivot.fillna(0).values

    return user_movie_ratings_matrix, user_movie_ratings_pivot, movies


def SVD(matrix, num_iterations):
    # Initialize matrices
    m, n = matrix.shape # m = number of users, n = number of movies
    U = np.random.rand(m, m)  # Random initialization for U
    Vt = np.random.rand(n, n)  # Random initialization for Vt

    print('SVD started : \n')
    for i in range(num_iterations):

        print(f'Iteration {i + 1} / {num_iterations}')

        # Power Iteration for U
        U = np.dot(matrix, Vt.T) # U = X * Vt
        U = U / np.linalg.norm(U, axis=0) # Normalize U

        # Power Iteration for Vt
        Vt = np.dot(U.T, matrix) # Vt = U.T * X
        Vt = Vt / np.linalg.norm(Vt, axis=0) # Normalize Vt

    # Calculate singular values
    sigma = np.diag(np.dot(U.T, np.dot(matrix, Vt))) # sigma = U.T * X * Vt

    print('SVD finished \n')

    return U, sigma, Vt


user_movie_ratings_matrix, user_movie_ratings_pivot, movies = data_loader()

# Perform Singular Value Decomposition
U, sigma, Vt = SVD(user_movie_ratings_matrix, num_iterations = 5)

# Choose the number of latent features (k) - it's a hyperparameter
k = 10

# Use the first k singular values and vectors to approximate the original matrix
U_k = U[:, :k] # U = [U_1, U_2, ..., U_k]
sigma_k = np.diag(sigma[:k]) # sigma = [sigma_1, sigma_2, ..., sigma_k]
Vt_k = Vt[:k, :] # Vt = [Vt_1, Vt_2, ..., Vt_k]

# Make predictions
predicted_ratings = np.dot(np.dot(U_k, sigma_k), Vt_k) # X = U * sigma * Vt

# Get user ID from input
user_id_input = int(input("Enter a user ID: "))

# Check if the entered user ID is valid
if user_id_input not in user_movie_ratings_pivot.index:
    print(f"User ID {user_id_input} not found in the dataset.")
else:
    # Get the index corresponding to the user ID
    user_index = user_movie_ratings_pivot.index.get_loc(user_id_input)

    # Get predicted ratings for the user
    user_ratings = predicted_ratings[user_index, :]

    # Find indices of movies with highest predicted ratings
    recommended_movie_indices = np.argsort(user_ratings)[::-1][:10]

    # Print recommended movies
    print(f"\nTop 10 Recommended Movies for User {user_id_input} : \n")
    for index in recommended_movie_indices:
        movie_id = user_movie_ratings_pivot.columns[index] # Get movie ID from column index
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0] # Get movie title from movie ID
        print(f"Title: {movie_title}")