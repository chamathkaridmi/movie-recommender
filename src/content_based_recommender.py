"""
content_based_recommender.py

Content-based recommendation using cosine similarity over movie
genre vectors and TF-IDF tag vectors.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between all movies
    based on a numeric feature matrix (e.g., one-hot encoded genres).
    """
    movie_ids = feature_matrix['movieId'].values
    features_only = feature_matrix.drop(columns=['movieId'])
    similarity = cosine_similarity(features_only)
    return pd.DataFrame(similarity, index=movie_ids, columns=movie_ids)


def recommend_similar_movies(movie_id: int, similarity_df: pd.DataFrame,
                              movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Given a movieId, return the N most similar movies by cosine similarity.
    """
    if movie_id not in similarity_df.index:
        raise ValueError(f"movieId {movie_id} not found in similarity matrix")

    scores = similarity_df.loc[movie_id].drop(movie_id)
    top_matches = scores.sort_values(ascending=False).head(n)

    result = pd.DataFrame({
        'movieId': top_matches.index,
        'similarity_score': top_matches.values
    })
    return result.merge(movies, on='movieId', how='left')[
        ['movieId', 'title', 'genres', 'similarity_score']
    ]


def build_combined_similarity(genres_df: pd.DataFrame,
                               tfidf_matrix,
                               tfidf_movie_ids,
                               genre_weight: float = 0.3,
                               tag_weight: float = 0.7) -> pd.DataFrame:
    """
    Build a combined similarity matrix blending genre and TF-IDF tag signals.
    For movies lacking tag data, genre-only similarity is used as fallback.
    """
    # Genre similarity for ALL movies
    all_movie_ids = genres_df['movieId'].values
    genre_features = genres_df.drop(columns=['movieId'])
    genre_sim = cosine_similarity(genre_features)
    genre_sim_df = pd.DataFrame(
        genre_sim, index=all_movie_ids, columns=all_movie_ids
    )

    # Tag similarity for movies WITH tags only
    tag_sim = cosine_similarity(tfidf_matrix)
    tag_movie_ids = tfidf_movie_ids.values
    tag_sim_df = pd.DataFrame(
        tag_sim, index=tag_movie_ids, columns=tag_movie_ids
    )

    # Start with genre-only as the base (scaled by genre_weight)
    combined = genre_sim_df.copy() * genre_weight

    # Find movies that exist in BOTH matrices
    genre_id_set = set(all_movie_ids)
    common_ids = [mid for mid in tag_movie_ids if mid in genre_id_set]
    print(f"Movies with both genre and tag data: {len(common_ids):,}")

    # Upgrade those movies to the full combined score
    combined.loc[common_ids, common_ids] = (
        genre_sim_df.loc[common_ids, common_ids] * genre_weight +
        tag_sim_df.loc[common_ids, common_ids] * tag_weight
    )

    return combined