import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
from PIL import Image
from io import BytesIO
import time

# Import utility modules
from utils.data_loader import (
    load_books_data, load_user_ratings, save_user_ratings, 
    add_user_rating, toggle_favorite, toggle_reading_list,
    get_user_rated_books, get_user_favorites, get_user_reading_list,
    search_books, filter_books, get_genres, get_subgenres
)
from utils.recommender import (
    prepare_content_features, get_content_based_recommendations,
    get_collaborative_recommendations, get_popularity_recommendations,
    get_hybrid_recommendations, get_recommendation_explanation
)
from utils.visualizations import (
    plot_genre_distribution, plot_rating_distribution,
    plot_user_genre_preferences, plot_recommendation_confidence
)

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .book-card {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .recommendation-confidence {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        # Return a default placeholder image
        return None

def display_book_card(book, user_id=None, show_rating=True, show_actions=True, explanation=None):
    """Display a book card with details and interactive elements"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Try to get book cover from URL
        image = load_image_from_url(book["cover_url"])
        if image:
            st.image(image, width=150)
        else:
            st.image("https://via.placeholder.com/150x225?text=No+Cover", width=150)
    
    with col2:
        st.subheader(book["title"])
        st.write(f"**Author:** {book['author']}")
        st.write(f"**Genre:** {book['genre']} - {book['subgenre']}")
        st.write(f"**Rating:** {'⭐' * int(round(book['rating']))} ({book['rating']})")
        st.write(f"**Publication Year:** {book['publication_year']}")
        
        if "series" in book and book["series"]:
            st.write(f"**Series:** {book['series']}")
        
        with st.expander("Description"):
            st.write(book["description"])
        
        if explanation:
            st.info(f"**Why recommended:** {explanation}")
        
        if show_actions and user_id:
            col_rate, col_fav, col_read = st.columns(3)
            
            # Rating widget
            if show_rating:
                with col_rate:
                    rating_key = f"rating_{book['book_id']}"
                    user_rating = st.session_state.get(rating_key, 0)
                    new_rating = st.select_slider(
                        "Your Rating",
                        options=[0, 1, 2, 3, 4, 5],
                        value=user_rating,
                        key=rating_key
                    )
                    
                    if new_rating != user_rating and new_rating > 0:
                        add_user_rating(user_id, book['book_id'], new_rating)
                        st.session_state[rating_key] = new_rating
                        st.experimental_rerun()
            
            # Favorite button
            with col_fav:
                user_data = st.session_state.user_ratings.get('users', {}).get(user_id, {})
                is_favorite = str(book['book_id']) in user_data.get('favorites', [])
                
                if st.button(
                    "❤️ Remove from Favorites" if is_favorite else "🖤 Add to Favorites",
                    key=f"fav_{book['book_id']}"
                ):
                    toggle_favorite(user_id, book['book_id'])
                    st.session_state.user_ratings = load_user_ratings()
                    st.experimental_rerun()
            
            # Reading list button
            with col_read:
                user_data = st.session_state.user_ratings.get('users', {}).get(user_id, {})
                in_reading_list = str(book['book_id']) in user_data.get('reading_list', [])
                
                if st.button(
                    "📚 Remove from Reading List" if in_reading_list else "📖 Add to Reading List",
                    key=f"read_{book['book_id']}"
                ):
                    toggle_reading_list(user_id, book['book_id'])
                    st.session_state.user_ratings = load_user_ratings()
                    st.experimental_rerun()
    
    st.markdown("---")

def initialize_session_state():
    """Initialize session state variables"""
    if 'books_df' not in st.session_state:
        st.session_state.books_df = load_books_data()
    
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = load_user_ratings()
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'default_user'
    
    if 'content_features' not in st.session_state:
        # Prepare content features for recommendation
        books_df, tfidf_matrix, indices = prepare_content_features(st.session_state.books_df)
        st.session_state.books_df = books_df
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.indices = indices
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = pd.DataFrame()
    
    if 'selected_books' not in st.session_state:
        st.session_state.selected_books = []
    
    if 'recommendation_method' not in st.session_state:
        st.session_state.recommendation_method = 'hybrid'

def sidebar_content():
    """Display sidebar content"""
    st.sidebar.title("📚 Book Recommendation Engine")
    
    # User identifier
    user_id = st.sidebar.text_input("Your Username:", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", 
        ["Home", "Search Books", "Your Profile", "Recommendations", "Explore Books"]
    )
    
    st.sidebar.markdown("---")
    
    # Recommendation method
    st.sidebar.header("Recommendation Settings")
    recommendation_method = st.sidebar.selectbox(
        "Recommendation Method:",
        ["Hybrid", "Content-Based", "Collaborative", "Popularity"],
        index=["hybrid", "content", "collaborative", "popularity"].index(
            st.session_state.recommendation_method
        )
    )
    st.session_state.recommendation_method = recommendation_method.lower()
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
    st.sidebar.markdown("---")
    
    # Information section
    st.sidebar.header("About")
    st.sidebar.info(
        "This book recommendation engine uses multiple algorithms to suggest books "
        "based on your preferences and reading history."
    )
    
    return page, num_recommendations

def home_page():
    """Display home page content"""
    st.title("📚 Welcome to the Book Recommendation Engine!")
    st.subheader("Discover your next favorite book")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How it works:
        1. **Rate books** you've already read
        2. **Select books** you enjoy
        3. **Get personalized recommendations**
        4. **Save** to your reading list
        
        Use the sidebar to navigate through the app.
        """)
        
        if st.button("Get Started", key="get_started"):
            st.session_state.page = "Search Books"
            st.experimental_rerun()
    
    with col2:
        # Display genre distribution chart
        st.write("### Explore our book collection")
        st.plotly_chart(plot_genre_distribution(st.session_state.books_df), use_container_width=True)
    
    # Display popular books
    st.header("Popular Books")
    popular_books = get_popularity_recommendations(st.session_state.books_df, num_recommendations=5)
    
    for _, book in popular_books.iterrows():
        display_book_card(book, user_id=st.session_state.user_id)

def search_page():
    """Display search page content"""
    st.title("🔍 Search Books")
    
    # Search options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search by title, author, or genre:", key="search_query")
    
    with col2:
        search_by = st.selectbox("Search by:", ["Title", "Author", "Genre", "All"], key="search_by")
    
    # Filter options
    st.subheader("Filter Options")
    col_genre, col_year, col_rating = st.columns(3)
    
    with col_genre:
        genres = ["All"] + get_genres(st.session_state.books_df)
        selected_genre = st.selectbox("Genre:", genres, key="filter_genre")
        if selected_genre == "All":
            selected_genre = None
    
    with col_year:
        min_year = int(st.session_state.books_df['publication_year'].min())
        max_year = int(st.session_state.books_df['publication_year'].max())
        year_range = st.slider("Publication Year:", min_year, max_year, (min_year, max_year), key="filter_year")
    
    with col_rating:
        min_rating = st.slider("Minimum Rating:", 1.0, 5.0, 3.0, 0.1, key="filter_rating")
    
    # Search button
    if st.button("Search", key="search_button"):
        # Perform search if query provided
        if search_query:
            results = search_books(
                st.session_state.books_df,
                search_query,
                search_by.lower()
            )
        else:
            results = st.session_state.books_df.copy()
        
        # Apply filters
        results = filter_books(
            results,
            genre=selected_genre,
            min_year=year_range[0],
            max_year=year_range[1],
            min_rating=min_rating
        )
        
        # Display results
        st.subheader(f"Search Results ({len(results)} books found)")
        
        if len(results) > 0:
            # Add multi-select for book selection
            selected_books = []
            for _, book in results.iterrows():
                col_select, col_card = st.columns([1, 10])
                
                with col_select:
                    is_selected = st.checkbox("", key=f"select_{book['book_id']}", 
                                             value=book['book_id'] in st.session_state.selected_books)
                    if is_selected:
                        selected_books.append(book['book_id'])
                
                with col_card:
                    display_book_card(book, user_id=st.session_state.user_id)
            
            # Update selected books
            st.session_state.selected_books = selected_books
            
            # Add recommendation button
            if st.button("Get Recommendations", key="get_recommendations"):
                if selected_books:
                    st.session_state.page = "Recommendations"
                    st.experimental_rerun()
                else:
                    st.warning("Please select at least one book to get recommendations.")
        else:
            st.info("No books found. Please try different search criteria.")

def profile_page():
    """Display user profile page content"""
    st.title("👤 Your Profile")
    
    # Get user data
    user_id = st.session_state.user_id
    user_rated_books = get_user_rated_books(user_id, st.session_state.books_df)
    user_favorites = get_user_favorites(user_id, st.session_state.books_df)
    user_reading_list = get_user_reading_list(user_id, st.session_state.books_df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", "⭐ Rated Books", "❤️ Favorites", "📚 Reading List"
    ])
    
    # Overview tab
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Reading Profile")
            st.write(f"**Username:** {user_id}")
            st.write(f"**Books Rated:** {len(user_rated_books)}")
            st.write(f"**Favorites:** {len(user_favorites)}")
            st.write(f"**Reading List:** {len(user_reading_list)}")
            
            if len(user_rated_books) > 0:
                st.write(f"**Average Rating Given:** {user_rated_books['user_rating'].mean():.1f}/5.0")
            
        with col2:
            if len(user_rated_books) > 0:
                st.plotly_chart(plot_user_genre_preferences(user_rated_books), use_container_width=True)
            else:
                st.info("Rate some books to see your genre preferences!")
    
    # Rated Books tab
    with tab2:
        st.subheader("Books You've Rated")
        
        if len(user_rated_books) > 0:
            # Sort by user rating
            user_rated_books = user_rated_books.sort_values("user_rating", ascending=False)
            
            for _, book in user_rated_books.iterrows():
                display_book_card(book, user_id=user_id, show_rating=True)
        else:
            st.info("You haven't rated any books yet. Search for books to rate them!")
            
            if st.button("Go to Search", key="goto_search_from_rated"):
                st.session_state.page = "Search Books"
                st.experimental_rerun()
    
    # Favorites tab
    with tab3:
        st.subheader("Your Favorite Books")
        
        if len(user_favorites) > 0:
            for _, book in user_favorites.iterrows():
                display_book_card(book, user_id=user_id)
        else:
            st.info("You haven't added any books to your favorites yet.")
    
    # Reading List tab
    with tab4:
        st.subheader("Your Reading List")
        
        if len(user_reading_list) > 0:
            for _, book in user_reading_list.iterrows():
                display_book_card(book, user_id=user_id)
        else:
            st.info("Your reading list is empty. Add books to read later!")

def recommendations_page(num_recommendations):
    """Display recommendations page content"""
    st.title("🔮 Your Book Recommendations")
    
    # Get user data
    user_id = st.session_state.user_id
    
    # Recommendation settings
    method = st.session_state.recommendation_method
    selected_books = st.session_state.selected_books
    
    # Display selected method
    st.subheader(f"Using {method.title()} Recommendation")
    
    # Generate recommendations based on method
    if st.button("Generate Recommendations", key="generate_recs"):
        with st.spinner("Generating personalized recommendations..."):
            if method == "content":
                if not selected_books:
                    st.warning("Please select at least one book for content-based recommendations.")
                    return
                
                recommendations = get_content_based_recommendations(
                    st.session_state.books_df,
                    st.session_state.tfidf_matrix,
                    st.session_state.indices,
                    selected_books,
                    num_recommendations
                )
                
            elif method == "collaborative":
                recommendations = get_collaborative_recommendations(
                    st.session_state.books_df,
                    st.session_state.user_ratings,
                    user_id,
                    num_recommendations
                )
                
            elif method == "popularity":
                recommendations = get_popularity_recommendations(
                    st.session_state.books_df,
                    num_recommendations=num_recommendations
                )
                
            else:  # hybrid
                recommendations = get_hybrid_recommendations(
                    st.session_state.books_df,
                    st.session_state.user_ratings,
                    st.session_state.tfidf_matrix,
                    st.session_state.indices,
                    user_id,
                    selected_books,
                    num_recommendations=num_recommendations
                )
            
            st.session_state.recommendations = recommendations
    
    # Display recommendations
    if not st.session_state.recommendations.empty:
        # Show confidence meter
        st.plotly_chart(
            plot_recommendation_confidence(st.session_state.recommendations),
            use_container_width=True
        )
        
        # Display books
        st.subheader("Books You Might Like")
        
        for _, book in st.session_state.recommendations.iterrows():
            # Generate explanation based on recommendation scores
            content_score = book.get('content_score', 0)
            collab_score = book.get('collab_score', 0)
            popularity_score = book.get('popularity_score', 0)
            
            explanation = get_recommendation_explanation(
                content_score, collab_score, popularity_score
            )
            
            display_book_card(book, user_id=user_id, explanation=explanation)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Add All to Reading List"):
                for _, book in st.session_state.recommendations.iterrows():
                    toggle_reading_list(user_id, book['book_id'])
                st.session_state.user_ratings = load_user_ratings()
                st.success("All recommended books added to your reading list!")
        
        with col2:
            csv = st.session_state.recommendations.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                "book_recommendations.csv",
                "text/csv",
                key="download_csv"
            )
    elif not st.button_pressed("generate_recs"):
        st.info("Click 'Generate Recommendations' to get personalized book suggestions.")

def explore_page():
    """Display explore books page content"""
    st.title("🌎 Explore Books")
    
    # Create tabs for exploration
    tab1, tab2, tab3 = st.tabs([
        "📊 Statistics", "📚 By Genre", "🔝 Top Rated"
    ])
    
    # Statistics tab
    with tab1:
        st.subheader("Book Collection Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution chart
            st.plotly_chart(plot_genre_distribution(st.session_state.books_df), use_container_width=True)
        
        with col2:
            # Rating distribution chart
            st.plotly_chart(plot_rating_distribution(st.session_state.books_df), use_container_width=True)
        
        # General statistics
        st.subheader("Summary Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Total Books", len(st.session_state.books_df))
        
        with col_stat2:
            st.metric("Genres", st.session_state.books_df['genre'].nunique())
        
        with col_stat3:
            st.metric("Authors", st.session_state.books_df['author'].nunique())
        
        with col_stat4:
            avg_rating = st.session_state.books_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} / 5.0")
    
    # By Genre tab
    with tab2:
        st.subheader("Explore Books by Genre")
        
        # Genre selector
        genres = get_genres(st.session_state.books_df)
        selected_genre = st.selectbox("Select a genre:", genres)
        
        # Display books in selected genre
        genre_books = st.session_state.books_df[st.session_state.books_df['genre'] == selected_genre]
        genre_books = genre_books.sort_values("rating", ascending=False)
        
        st.write(f"### Top Books in {selected_genre}")
        
        for _, book in genre_books.head(5).iterrows():
            display_book_card(book, user_id=st.session_state.user_id)
        
        # Show more button
        with st.expander("Show more books in this genre"):
            for _, book in genre_books[5:10].iterrows():
                display_book_card(book, user_id=st.session_state.user_id)
    
    # Top Rated tab
    with tab3:
        st.subheader("Top Rated Books")
        
        # Minimum number of ratings filter
        min_ratings = st.slider(
            "Minimum number of ratings:",
            min_value=100000,
            max_value=5000000,
            value=1000000,
            step=100000
        )
        
        # Get top rated books
        top_rated = st.session_state.books_df[st.session_state.books_df['num_ratings'] >= min_ratings]
        top_rated = top_rated.sort_values("rating", ascending=False)
        
        # Display top rated books
        for _, book in top_rated.head(10).iterrows():
            display_book_card(book, user_id=st.session_state.user_id)

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar content
    page, num_recommendations = sidebar_content()
    
    # Override page from session state if set
    if 'page' in st.session_state:
        page = st.session_state.page
        st.session_state.pop('page')  # Clear after use
    
    # Display page content
    if page == "Home":
        home_page()
    elif page == "Search Books":
        search_page()
    elif page == "Your Profile":
        profile_page()
    elif page == "Recommendations":
        recommendations_page(num_recommendations)
    elif page == "Explore Books":
        explore_page()

if __name__ == "__main__":
    main() 