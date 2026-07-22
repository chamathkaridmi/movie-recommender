"""
app.py

Streamlit application for the Movie Recommendation System.
Provides three recommendation modes:
  1. Popularity-Based  — best movies overall
  2. Content-Based     — movies similar to one you liked
  3. Collaborative     — personalised for your user ID
"""

import streamlit as st
import sys
import os

# Make src/ importable from app/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommend import (
    load_data,
    train_collaborative_model,
    get_popular_movies,
    get_similar_movies,
    get_user_recommendations,
)

# Custom CSS for better readability and professional appearance
st.markdown("""
    <style>
    /* Constrain and center main content */
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1100px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Base font size — target Streamlit's actual text containers */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stText"] {
        font-size: 17px !important;
        line-height: 1.7 !important;
        color: #333 !important;
    }

    /* Hero header */
    .hero {
        padding-bottom: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #E50914;
    }
    .hero h1 {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: #1a1a2e !important;
        margin-bottom: 0.3rem !important;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.15rem !important;
        color: #555 !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }

    /* Section headers */
    h2, h3,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #16213e !important;
        margin-top: 1.5rem !important;
    }

    /* Dataframe text */
    [data-testid="stDataFrame"] * {
        font-size: 15px !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        font-size: 16px !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.2rem !important;
    }
    .stRadio label, .stSelectbox label,
    .stSlider label, .stNumberInput label {
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        font-size: 16px !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.25);
    }

    [data-testid="stAlert"] p {
        font-size: 16px !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-size: 14px !important;
    }

    /* Movie card grid */
    .movie-card {
        border-radius: 12px;
        overflow: hidden;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.2rem;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }
    .poster {
        position: relative;
        height: 160px;
        display: flex;
        align-items: flex-end;
        padding: 12px;
    }
    .rank-badge {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0,0,0,0.55);
        color: #fff;
        font-weight: 700;
        font-size: 13px;
        padding: 3px 9px;
        border-radius: 6px;
    }
    .poster-title {
        color: #fff;
        font-weight: 700;
        font-size: 15px;
        line-height: 1.3;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .card-body {
        padding: 12px 14px 16px;
    }
    .rating-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .rating-badge {
        background: #1a1a2e;
        color: #FFD700;
        font-weight: 700;
        font-size: 13px;
        padding: 3px 10px;
        border-radius: 20px;
    }
    .rating-label {
        font-size: 12px;
        color: #888;
    }
    .genre-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .genre-pill {
        background: #F5F5F5;
        color: #555;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
    }

    </style>
""", unsafe_allow_html=True)


def render_movie_card(rank, title, genres, rating_label, rating_value):
    """Render a single Netflix-style movie card with a gradient poster."""
    from recommend import get_gradient_for_genres
    color1, color2 = get_gradient_for_genres(genres)

    genre_list = [g for g in genres.split('|') if g != '(no genres listed)'][:3]
    pills_html = ''.join(
        f'<span class="genre-pill">{g}</span>' for g in genre_list
    )

    st.markdown(f"""
    <div class="movie-card">
        <div class="poster" style="background: linear-gradient(135deg, {color1}, {color2});">
            <div class="rank-badge">#{rank}</div>
            <div class="poster-title">{title}</div>
        </div>
        <div class="card-body">
            <div class="rating-row">
                <span class="rating-badge">⭐ {rating_value}</span>
                <span class="rating-label">{rating_label}</span>
            </div>
            <div class="genre-pills">{pills_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CACHING — load data and train model only once
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading datasets...")
def cached_load_data():
    """
    st.cache_data stores the result after the first run.
    Every subsequent call returns the cached copy instantly
    instead of re-reading all the CSV and pickle files.
    """
    return load_data()


@st.cache_resource(show_spinner="Training recommendation model...")
def cached_train_model(ratings, user_to_idx, idx_to_movie):
    """
    st.cache_resource is used for objects that should not be
    serialised (like fitted sklearn models). Trains SVD once
    and keeps it in memory for the app's lifetime.
    """
    return train_collaborative_model(ratings, user_to_idx, idx_to_movie)


# ─────────────────────────────────────────────
# LOAD EVERYTHING AT STARTUP
# ─────────────────────────────────────────────

data = cached_load_data()

movies       = data['movies']
ratings      = data['ratings']
genres_encoded  = data['genres_encoded']
tfidf_matrix    = data['tfidf_matrix']
tfidf_movie_ids = data['tfidf_movie_ids']
user_to_idx     = data['user_to_idx']
idx_to_movie    = data['idx_to_movie']

svd, user_factors, user_means, movie_to_idx = cached_train_model(
    ratings, user_to_idx, idx_to_movie
)

# Sorted list of all movie titles for the dropdown
all_titles = sorted(movies['title'].dropna().unique().tolist())

# All unique genres for the filter
all_genres = sorted(set(
    g for genres in movies['genres'].dropna()
    for g in genres.split('|')
    if g != '(no genres listed)'
))

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
<h1>🎬 CineMatch</h1>
<p>Discover movies you'll love using three recommendation approaches — powered by the MovieLens 20M dataset.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

method = st.sidebar.radio(
    "Recommendation Method",
    options=["🏆 Popular Movies",
             "🎭 Similar Movies (Content-Based)",
             "👤 For You (Collaborative Filtering)"],
    help="Choose how you want movies recommended to you."
)

n_recs = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5, max_value=20, value=10, step=1
)

genre_filter = st.sidebar.selectbox(
    "Filter by Genre (optional)",
    options=["All"] + all_genres,
    index=0
)

st.sidebar.divider()
st.sidebar.markdown(
    "**About this app**\n\n"
    "Built with the MovieLens 20M dataset.\n\n"
    "Models: Popularity · Content-Based · SVD\n\n"
    "by Ridmi Kandevidana"
)

# ─────────────────────────────────────────────
# MAIN CONTENT — switches based on method
# ─────────────────────────────────────────────

# ── METHOD 1: POPULARITY ──────────────────────
if method == "🏆 Popular Movies":
    st.subheader("🏆 Most Popular Movies")
    st.markdown(
        "These are the highest-rated movies with at least 500 ratings. "
        "A great starting point if you're not sure what to watch."
    )

    min_ratings = st.slider(
        "Minimum number of ratings",
        min_value=100, max_value=5000, value=500, step=100
    )

    if st.button("Show Popular Movies", type="primary"):
        with st.spinner("Finding top movies..."):
            result = get_popular_movies(
                ratings, movies,
                min_ratings=min_ratings,
                n=n_recs,
                genre_filter=genre_filter
            )

        if result.empty:
            st.warning("No movies found with the current filters. "
                       "Try a lower minimum ratings threshold or "
                       "a different genre.")
        else:
            st.success(f"Top {len(result)} movies:")
            cols = st.columns(3)
            for i, row in enumerate(result.itertuples()):
                with cols[i % 3]:
                    render_movie_card(
                        rank=i + 1,
                        title=row.title,
                        genres=row.genres,
                        rating_label=f"{int(row.rating_count):,} ratings",
                        rating_value=round(row.avg_rating, 2)
                    )

# ── METHOD 2: CONTENT-BASED ───────────────────
elif method == "🎭 Similar Movies (Content-Based)":
    st.subheader("🎭 Find Movies Similar to One You Liked")
    st.markdown(
        "Select a movie you enjoyed and we'll find others "
        "with similar genres and themes."
    )

    selected_movie = st.selectbox(
        "Search and select a movie",
        options=all_titles,
        index=all_titles.index("Toy Story (1995)")
        if "Toy Story (1995)" in all_titles else 0,
        help="Type to search through all 27,278 movies."
    )

    if st.button("Find Similar Movies", type="primary"):
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            result, error = get_similar_movies(
                movie_title=selected_movie,
                movies=movies,
                genres_encoded=genres_encoded,
                tfidf_matrix=tfidf_matrix,
                tfidf_movie_ids=tfidf_movie_ids,
                n=n_recs,
                genre_filter=genre_filter
            )

        if error:
            st.error(error)
        elif result is None or result.empty:
            st.warning("No similar movies found with the current filters.")
        else:
            st.success(f"Top {len(result)} movies similar to "
                       f"**{selected_movie}**:")
            cols = st.columns(3)
            for i, row in enumerate(result.itertuples()):
                with cols[i % 3]:
                    render_movie_card(
                        rank=i + 1,
                        title=row.title,
                        genres=row.genres,
                        rating_label="match score",
                        rating_value=round(row.similarity_score, 2)
                    )

            # Show the selected movie's own details
            with st.expander("ℹ️ About the selected movie"):
                movie_info = movies[movies['title'] == selected_movie]
                if not movie_info.empty:
                    row = movie_info.iloc[0]
                    movie_ratings = ratings[
                        ratings['movieId'] == row['movieId']
                    ]['rating']
                    st.write(f"**Title:** {row['title']}")
                    st.write(f"**Genres:** {row['genres']}")
                    if len(movie_ratings) > 0:
                        st.write(
                            f"**Average Rating:** "
                            f"{movie_ratings.mean():.2f} "
                            f"({len(movie_ratings):,} ratings)"
                        )

# ── METHOD 3: COLLABORATIVE FILTERING ─────────
else:
    st.subheader("👤 Personalised Recommendations For You")
    st.markdown(
        "Enter your User ID to get personalised recommendations "
        "based on your rating history. "
        "Valid User IDs: **1 to 138,493**."
    )

    user_id_input = st.number_input(
        "Enter your User ID",
        min_value=1,
        max_value=138493,
        value=1,
        step=1,
        help="Enter any User ID between 1 and 138,493."
    )

    if st.button("Get My Recommendations", type="primary"):
        with st.spinner(f"Generating recommendations for User {user_id_input}..."):
            result, error = get_user_recommendations(
                user_id=int(user_id_input),
                ratings=ratings,
                movies=movies,
                svd=svd,
                user_factors=user_factors,
                user_means=user_means,
                user_to_idx=user_to_idx,
                movie_to_idx=movie_to_idx,
                idx_to_movie=idx_to_movie,
                genres_encoded=genres_encoded,
                n=n_recs,
                genre_filter=genre_filter
            )

        if error:
            st.error(error)
        elif result is None or result.empty:
            st.warning("No recommendations found with the current filters.")
        else:
            # Show how many movies this user has rated
            user_history = ratings[
                ratings['userId'] == int(user_id_input)
            ]
            st.success(
                f"Top {len(result)} recommendations for User "
                f"**{user_id_input}** "
                f"(based on {len(user_history):,} ratings in history):"
            )
            cols = st.columns(3)
            for i, row in enumerate(result.itertuples()):
                with cols[i % 3]:
                    render_movie_card(
                        rank=i + 1,
                        title=row.title,
                        genres=row.genres,
                        rating_label="predicted",
                        rating_value=round(row.predicted_rating, 2)
                    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "MovieLens 20M Dataset · GroupLens Research · "
    "Built with Python, pandas, scikit-learn & Streamlit"
)