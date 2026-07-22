"""
collaborative_filtering.py

Collaborative filtering using TruncatedSVD (matrix factorization)
on the sparse user-item ratings matrix.
Designed to work on standard hardware by avoiding full matrix reconstruction.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def train_svd(sparse_matrix, n_components: int = 50, random_state: int = 42):
    """
    Fit a TruncatedSVD model on the sparse user-item matrix.

    Parameters
    ----------
    sparse_matrix  : scipy sparse matrix (users x movies)
    n_components   : number of latent factors (default 50)
    random_state   : for reproducibility

    Returns
    -------
    svd            : fitted TruncatedSVD object
    user_factors   : user latent factor matrix (users x components)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd.fit_transform(sparse_matrix)
    return svd, user_factors


def recommend_for_user(user_id: int, svd, user_factors: np.ndarray,
                        user_to_idx: dict, idx_to_movie: dict,
                        ratings: pd.DataFrame, movies: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
    """
    Recommend top N movies for a given user without reconstructing
    the full prediction matrix. Computes predictions for one user only.

    Parameters
    ----------
    user_id      : real userId to generate recommendations for
    svd          : fitted TruncatedSVD object
    user_factors : full user factor matrix from fit_transform
    user_to_idx  : maps real userId to matrix row index
    idx_to_movie : maps matrix column index to real movieId
    ratings      : original ratings DataFrame (to exclude already-seen)
    movies       : movies DataFrame (for title/genre lookup)
    n            : number of recommendations
    """
    if user_id not in user_to_idx:
        raise ValueError(f"userId {user_id} not found in training data")

    user_idx = user_to_idx[user_id]

    # Compute predictions for THIS USER ONLY — memory safe
    user_predicted = np.dot(user_factors[user_idx], svd.components_)

    # Exclude movies this user has already rated
    already_rated = set(
        ratings[ratings['userId'] == user_id]['movieId'].values
    )

    # Score all unrated movies
    scores = [
        (idx_to_movie[movie_idx], score)
        for movie_idx, score in enumerate(user_predicted)
        if idx_to_movie[movie_idx] not in already_rated
    ]

    scores.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(scores[:n], columns=['movieId', 'predicted_rating'])
    return result.merge(movies, on='movieId', how='left')[
        ['movieId', 'title', 'genres', 'predicted_rating']
    ]