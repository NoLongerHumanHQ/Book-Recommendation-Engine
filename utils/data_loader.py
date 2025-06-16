import pandas as pd
import json
import os
from typing import Dict, List, Union, Optional
import streamlit as st

@st.cache_data
def load_books_data() -> pd.DataFrame:
    """
    Load books data from CSV file with caching for performance
    
    Returns:
        pd.DataFrame: DataFrame containing book data
    """
    try:
        books_df = pd.read_csv('data/books.csv')
        # Convert publication_year to int, handling BCE dates (negative years)
        books_df['publication_year'] = books_df['publication_year'].astype(int)
        
        # Fill missing values
        books_df['series'] = books_df['series'].fillna('')
        books_df['description'] = books_df['description'].fillna('No description available')
        books_df['cover_url'] = books_df['cover_url'].fillna('https://via.placeholder.com/150x225?text=No+Cover')
        
        return books_df
    except Exception as e:
        st.error(f"Error loading books data: {e}")
        # Return empty dataframe with correct columns if loading fails
        return pd.DataFrame(columns=['book_id', 'title', 'author', 'genre', 'subgenre', 
                                    'rating', 'num_ratings', 'publication_year', 'description',
                                    'isbn', 'page_count', 'language', 'cover_url', 'series',
                                    'reading_level'])

def load_user_ratings() -> Dict:
    """
    Load user ratings from JSON file
    
    Returns:
        Dict: Dictionary containing user ratings
    """
    try:
        if os.path.exists('data/user_ratings.json'):
            with open('data/user_ratings.json', 'r') as file:
                return json.load(file)
        else:
            # Return default structure if file doesn't exist
            return {"users": {}}
    except Exception as e:
        st.error(f"Error loading user ratings: {e}")
        return {"users": {}}

def save_user_ratings(ratings_data: Dict) -> bool:
    """
    Save user ratings to JSON file
    
    Args:
        ratings_data (Dict): Dictionary containing user ratings
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open('data/user_ratings.json', 'w') as file:
            json.dump(ratings_data, file, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving user ratings: {e}")
        return False

def get_user_rated_books(user_id: str, books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get books rated by a specific user
    
    Args:
        user_id (str): User identifier
        books_df (pd.DataFrame): DataFrame containing all books
        
    Returns:
        pd.DataFrame: DataFrame containing books rated by the user
    """
    ratings_data = load_user_ratings()
    
    if user_id not in ratings_data['users']:
        return pd.DataFrame()
    
    user_ratings = ratings_data['users'][user_id]['ratings']
    rated_book_ids = [int(book_id) for book_id in user_ratings.keys()]
    
    rated_books = books_df[books_df['book_id'].isin(rated_book_ids)].copy()
    
    # Add user rating to the dataframe
    rated_books['user_rating'] = rated_books['book_id'].apply(
        lambda x: user_ratings[str(x)]
    )
    
    return rated_books

def get_user_favorites(user_id: str, books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get books marked as favorites by a specific user
    
    Args:
        user_id (str): User identifier
        books_df (pd.DataFrame): DataFrame containing all books
        
    Returns:
        pd.DataFrame: DataFrame containing user's favorite books
    """
    ratings_data = load_user_ratings()
    
    if user_id not in ratings_data['users'] or 'favorites' not in ratings_data['users'][user_id]:
        return pd.DataFrame()
    
    favorite_book_ids = [int(book_id) for book_id in ratings_data['users'][user_id]['favorites']]
    
    return books_df[books_df['book_id'].isin(favorite_book_ids)]

def get_user_reading_list(user_id: str, books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get books in a user's reading list
    
    Args:
        user_id (str): User identifier
        books_df (pd.DataFrame): DataFrame containing all books
        
    Returns:
        pd.DataFrame: DataFrame containing user's reading list books
    """
    ratings_data = load_user_ratings()
    
    if user_id not in ratings_data['users'] or 'reading_list' not in ratings_data['users'][user_id]:
        return pd.DataFrame()
    
    reading_list_book_ids = [int(book_id) for book_id in ratings_data['users'][user_id]['reading_list']]
    
    return books_df[books_df['book_id'].isin(reading_list_book_ids)]

def add_user_rating(user_id: str, book_id: int, rating: int) -> bool:
    """
    Add or update a user's rating for a book
    
    Args:
        user_id (str): User identifier
        book_id (int): Book identifier
        rating (int): Rating value (1-5)
        
    Returns:
        bool: True if successful, False otherwise
    """
    ratings_data = load_user_ratings()
    
    # Create user entry if it doesn't exist
    if user_id not in ratings_data['users']:
        ratings_data['users'][user_id] = {
            'ratings': {},
            'favorites': [],
            'reading_list': []
        }
    
    # Add or update rating
    ratings_data['users'][user_id]['ratings'][str(book_id)] = rating
    
    return save_user_ratings(ratings_data)

def toggle_favorite(user_id: str, book_id: int) -> bool:
    """
    Toggle a book's favorite status for a user
    
    Args:
        user_id (str): User identifier
        book_id (int): Book identifier
        
    Returns:
        bool: True if book is now a favorite, False if removed from favorites
    """
    ratings_data = load_user_ratings()
    
    # Create user entry if it doesn't exist
    if user_id not in ratings_data['users']:
        ratings_data['users'][user_id] = {
            'ratings': {},
            'favorites': [],
            'reading_list': []
        }
    
    # Initialize favorites list if it doesn't exist
    if 'favorites' not in ratings_data['users'][user_id]:
        ratings_data['users'][user_id]['favorites'] = []
    
    book_id_str = str(book_id)
    favorites = ratings_data['users'][user_id]['favorites']
    
    # Toggle favorite status
    if book_id_str in favorites:
        favorites.remove(book_id_str)
        is_favorite = False
    else:
        favorites.append(book_id_str)
        is_favorite = True
    
    save_user_ratings(ratings_data)
    return is_favorite

def toggle_reading_list(user_id: str, book_id: int) -> bool:
    """
    Toggle a book's reading list status for a user
    
    Args:
        user_id (str): User identifier
        book_id (int): Book identifier
        
    Returns:
        bool: True if book is now in reading list, False if removed
    """
    ratings_data = load_user_ratings()
    
    # Create user entry if it doesn't exist
    if user_id not in ratings_data['users']:
        ratings_data['users'][user_id] = {
            'ratings': {},
            'favorites': [],
            'reading_list': []
        }
    
    # Initialize reading list if it doesn't exist
    if 'reading_list' not in ratings_data['users'][user_id]:
        ratings_data['users'][user_id]['reading_list'] = []
    
    book_id_str = str(book_id)
    reading_list = ratings_data['users'][user_id]['reading_list']
    
    # Toggle reading list status
    if book_id_str in reading_list:
        reading_list.remove(book_id_str)
        is_in_list = False
    else:
        reading_list.append(book_id_str)
        is_in_list = True
    
    save_user_ratings(ratings_data)
    return is_in_list

def get_genres(books_df: pd.DataFrame) -> List[str]:
    """
    Get unique genres from books dataframe
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        List[str]: List of unique genres
    """
    return sorted(books_df['genre'].unique().tolist())

def get_subgenres(books_df: pd.DataFrame) -> List[str]:
    """
    Get unique subgenres from books dataframe
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        List[str]: List of unique subgenres
    """
    return sorted(books_df['subgenre'].unique().tolist())

def search_books(books_df: pd.DataFrame, query: str, search_by: str = 'title') -> pd.DataFrame:
    """
    Search books by title, author, or genre
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        query (str): Search query
        search_by (str): Field to search (title, author, genre)
        
    Returns:
        pd.DataFrame: DataFrame containing search results
    """
    query = query.lower()
    
    if search_by == 'title':
        return books_df[books_df['title'].str.lower().str.contains(query)]
    elif search_by == 'author':
        return books_df[books_df['author'].str.lower().str.contains(query)]
    elif search_by == 'genre':
        return books_df[books_df['genre'].str.lower().str.contains(query)]
    elif search_by == 'all':
        title_matches = books_df['title'].str.lower().str.contains(query)
        author_matches = books_df['author'].str.lower().str.contains(query)
        genre_matches = books_df['genre'].str.lower().str.contains(query)
        return books_df[title_matches | author_matches | genre_matches]
    else:
        return pd.DataFrame()

def filter_books(books_df: pd.DataFrame, 
                genre: Optional[str] = None,
                min_year: Optional[int] = None,
                max_year: Optional[int] = None,
                min_rating: Optional[float] = None) -> pd.DataFrame:
    """
    Filter books based on criteria
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        genre (str, optional): Genre to filter by
        min_year (int, optional): Minimum publication year
        max_year (int, optional): Maximum publication year
        min_rating (float, optional): Minimum book rating
        
    Returns:
        pd.DataFrame: DataFrame containing filtered books
    """
    filtered_df = books_df.copy()
    
    if genre:
        filtered_df = filtered_df[filtered_df['genre'] == genre]
    
    if min_year is not None:
        filtered_df = filtered_df[filtered_df['publication_year'] >= min_year]
    
    if max_year is not None:
        filtered_df = filtered_df[filtered_df['publication_year'] <= max_year]
    
    if min_rating is not None:
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
    
    return filtered_df 