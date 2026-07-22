"""
popularity_recommender.py

A baseline recommendation model that ranks movies by average rating,
subject to a minimum rating-count threshold to avoid small-sample bias.
"""

import pandas as pd


def build_popularity_table(ratings: pd.DataFrame, movies: pd.DataFrame,
                            min_ratings: int = 500) -> pd.DataFrame:
    """
    Calculate average rating and rating count per movie, filtered by
    a minimum ratings threshold, merged with movie titles and genres.

    Parameters
    ----------
    ratings : DataFrame with columns ['userId', 'movieId', 'rating']
    movies  : DataFrame with columns ['movieId', 'title', 'genres']
    min_ratings : minimum number of ratings a movie needs to qualify

    Returns
    -------
    DataFrame sorted by avg_rating (descending), with columns:
    ['movieId', 'title', 'genres', 'avg_rating', 'rating_count']
    """
    stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
    stats.columns = ['avg_rating', 'rating_count']

    qualified = stats[stats['rating_count'] >= min_ratings]

    result = qualified.merge(movies, on='movieId', how='left')
    result = result.sort_values('avg_rating', ascending=False).reset_index(drop=True)

    return result[['movieId', 'title', 'genres', 'avg_rating', 'rating_count']]


def recommend_popular(popularity_table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N movies from a pre-built popularity table."""
    return popularity_table.head(n)