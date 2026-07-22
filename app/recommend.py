"""
recommend.py

Service layer connecting the Streamlit UI to the underlying
recommendation models. All model loading and recommendation
logic lives here — app.py only calls these functions.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Add src/ to path so we can import our model modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from popularity_recommender import build_popularity_table
from collaborative_filtering import train_svd


# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────

def _path(relative):
    """Build an absolute path relative to the project root."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, relative)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    """
    Load all datasets and pre-built features needed by the app.
    Returns a dictionary of all loaded objects.
    """
    data = {}

    # Core movie and rating data
    data['movies']  = pd.read_csv(_path('data/processed/movies_clean.csv'))
    data['ratings'] = pd.read_csv(_path('data/processed/ratings_clean.csv'))

    # Genre features for content-based filtering
    data['genres_encoded'] = pd.read_csv(_path('data/features/genres_encoded.csv'))

    # TF-IDF tag features
    data['tfidf_matrix']    = load_npz(_path('data/features/tfidf_matrix.npz'))
    data['tfidf_movie_ids'] = pd.read_csv(
        _path('data/features/tfidf_movieId_index.csv')
    )['movieId']

    # Lookup tables for collaborative filtering
    data['user_to_idx'] = joblib.load(_path('data/features/user_to_idx.pkl'))
    data['idx_to_movie'] = joblib.load(_path('data/features/idx_to_movie.pkl'))

    return data


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_collaborative_model(ratings, user_to_idx, idx_to_movie,
                               n_components=50):
    """
    Train SVD on mean-centered ratings.
    Returns svd model, user_factors, user_means, and ID mappings.
    """
    train_users  = list(user_to_idx.keys())
    train_movies = list(idx_to_movie.values())

    user_means = ratings.groupby('userId')['rating'].mean()

    # Mean-center ratings
    ratings_centered = ratings.copy()
    ratings_centered['rating'] = (
        ratings['rating'] - ratings['userId'].map(user_means)
    )

    # Build sparse matrix
    valid = ratings_centered[
        ratings_centered['userId'].isin(user_to_idx) &
        ratings_centered['movieId'].isin({v: k for k, v in idx_to_movie.items()})
    ].copy()

    movie_to_idx = {mid: idx for idx, mid in idx_to_movie.items()}
    row_idx = valid['userId'].map(user_to_idx)
    col_idx = valid['movieId'].map(movie_to_idx)

    sparse = csr_matrix(
        (valid['rating'].values, (row_idx.values, col_idx.values)),
        shape=(len(train_users), len(train_movies))
    )

    svd, user_factors = train_svd(sparse, n_components=n_components)
    return svd, user_factors, user_means, movie_to_idx


# ─────────────────────────────────────────────
# MODEL 1: POPULARITY-BASED
# ─────────────────────────────────────────────

def get_popular_movies(ratings, movies, min_ratings=500, n=10,
                        genre_filter=None):
    """
    Return the top N most popular movies by average rating,
    optionally filtered by genre.
    """
    table = build_popularity_table(ratings, movies, min_ratings=min_ratings)

    if genre_filter and genre_filter != "All":
        table = table[table['genres'].str.contains(genre_filter, na=False)]

    return table.head(n)[['title', 'genres', 'avg_rating', 'rating_count']]


# ─────────────────────────────────────────────
# MODEL 2: CONTENT-BASED
# ─────────────────────────────────────────────

def get_similar_movies(movie_title, movies, genres_encoded,
                        tfidf_matrix, tfidf_movie_ids,
                        n=10, genre_filter=None):
    """
    Given a movie title, return the N most similar movies
    using genre + TF-IDF tag cosine similarity.
    """
    # Find the movieId for the given title
    match = movies[movies['title'] == movie_title]
    if match.empty:
        return None, f"Movie '{movie_title}' not found."

    movie_id = match.iloc[0]['movieId']

    # Genre features
    genre_features = genres_encoded.drop(columns=['movieId']).values
    genre_ids      = genres_encoded['movieId'].values
    genre_idx_map  = {mid: i for i, mid in enumerate(genre_ids)}

    if movie_id not in genre_idx_map:
        return None, f"No genre data for '{movie_title}'."

    query_genre_vec = genre_features[genre_idx_map[movie_id]].reshape(1, -1)
    genre_sims      = cosine_similarity(query_genre_vec, genre_features)[0]

    # TF-IDF tag features (if available for this movie)
    tag_ids     = tfidf_movie_ids.values
    tag_idx_map = {mid: i for i, mid in enumerate(tag_ids)}

    if movie_id in tag_idx_map:
        query_tag_vec = tfidf_matrix[tag_idx_map[movie_id]]
        tag_sims      = cosine_similarity(query_tag_vec, tfidf_matrix)[0]
        # Combined: 30% genre, 70% tags
        combined_sims = dict(zip(tag_ids, 0.3 * genre_sims[
            [genre_idx_map.get(mid, 0) for mid in tag_ids]
        ] + 0.7 * tag_sims))
    else:
        # Fallback: genre only
        combined_sims = dict(zip(genre_ids, genre_sims))

    # Build results — exclude the query movie itself
    scores = [
        (mid, score)
        for mid, score in combined_sims.items()
        if mid != movie_id
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(scores[:n*3], columns=['movieId', 'similarity_score'])
    result = result.merge(movies, on='movieId', how='left')

    if genre_filter and genre_filter != "All":
        result = result[result['genres'].str.contains(genre_filter, na=False)]

    result = result.head(n)[['title', 'genres', 'similarity_score']]
    return result, None


# ─────────────────────────────────────────────
# MODEL 3: COLLABORATIVE FILTERING
# ─────────────────────────────────────────────

def get_user_recommendations(user_id, ratings, movies,
                              svd, user_factors, user_means,
                              user_to_idx, movie_to_idx, idx_to_movie,
                              genres_encoded,
                              n=10, genre_filter=None):
    """
    Return top N personalized movie recommendations for a given user.
    Falls back to popularity if user is not in training data.
    """
    if user_id not in user_to_idx:
        return None, (
            f"User ID {user_id} not found in training data. "
            f"Valid user IDs range from 1 to 138,493."
        )

    user_idx    = user_to_idx[user_id]
    user_vec    = user_factors[user_idx]
    user_mean   = user_means.get(user_id, ratings['rating'].mean())

    # Predicted scores for all movies
    scores = np.dot(user_vec, svd.components_) + user_mean

    # Movies this user has already rated — exclude them
    already_rated = set(
        ratings[ratings['userId'] == user_id]['movieId'].values
    )

    # Build scored list
    scored = [
        (idx_to_movie[i], float(scores[i]))
        for i in range(len(scores))
        if idx_to_movie.get(i) not in already_rated
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(scored[:n*3], columns=['movieId', 'predicted_rating'])
    result = result.merge(movies, on='movieId', how='left')

    if genre_filter and genre_filter != "All":
        result = result[result['genres'].str.contains(genre_filter, na=False)]

    result = result.head(n)[['title', 'genres', 'predicted_rating']]
    return result, None

# ─────────────────────────────────────────────
# GENRE → COLOR MAPPING (for placeholder posters)
# ─────────────────────────────────────────────

GENRE_COLORS = {
    'Action':      ('#8B0000', '#E50914'),
    'Adventure':   ('#B8860B', '#FFA500'),
    'Animation':   ('#1E90FF', '#00CED1'),
    'Comedy':      ('#DAA520', '#FFD700'),
    'Crime':       ('#2F2F2F', '#4B0082'),
    'Documentary': ('#556B2F', '#8FBC8F'),
    'Drama':       ('#4B0082', '#9370DB'),
    'Fantasy':     ('#8A2BE2', '#DA70D6'),
    'Horror':      ('#1a1a1a', '#8B0000'),
    'Mystery':     ('#2F4F4F', '#483D8B'),
    'Romance':     ('#C71585', '#FF69B4'),
    'Sci-Fi':      ('#00008B', '#00BFFF'),
    'Thriller':    ('#333333', '#B22222'),
    'War':         ('#3C3C3C', '#708090'),
    'Western':     ('#8B4513', '#D2691E'),
}
DEFAULT_GRADIENT = ('#333333', '#1a1a2e')


def get_gradient_for_genres(genres_str):
    """Pick a two-color gradient based on a movie's first listed genre."""
    if not genres_str or genres_str == '(no genres listed)':
        return DEFAULT_GRADIENT
    first_genre = genres_str.split('|')[0]
    return GENRE_COLORS.get(first_genre, DEFAULT_GRADIENT)