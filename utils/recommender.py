import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from typing import List, Dict, Optional

@st.cache_data
def prepare_content_features(books_df: pd.DataFrame):
    """
    Prepare content features for content-based recommendation
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        tuple: (books_df, tfidf_matrix, indices)
    """
    # Create a combined feature text
    books_df['content_features'] = (
        books_df['title'] + ' ' +
        books_df['author'] + ' ' +
        books_df['genre'] + ' ' +
        books_df['subgenre'] + ' ' +
        books_df['description'].apply(lambda x: preprocess_text(x))
    )
    
    # Use TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(books_df['content_features'])
    
    # Create a reverse map of indices and book IDs
    indices = pd.Series(books_df.index, index=books_df['book_id']).drop_duplicates()
    
    return books_df, tfidf_matrix, indices

def preprocess_text(text: str) -> str:
    """Preprocess text for better feature extraction"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_content_based_recommendations(books_df: pd.DataFrame, 
                                     tfidf_matrix, 
                                     indices: pd.Series,
                                     book_ids: List[int], 
                                     num_recommendations: int = 10) -> pd.DataFrame:
    """
    Get content-based recommendations for a list of books
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        tfidf_matrix: TF-IDF matrix of book features
        indices (pd.Series): Series mapping book IDs to indices
        book_ids (List[int]): List of book IDs to recommend from
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing recommended books
    """
    # If no books are provided, return empty DataFrame
    if not book_ids:
        return pd.DataFrame()
    
    # Get the indices of the books
    book_indices = [indices.get(book_id) for book_id in book_ids if book_id in indices]
    
    # If none of the provided book_ids are found, return empty DataFrame
    if not book_indices:
        return pd.DataFrame()
    
    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Calculate the similarity scores for all books
    sim_scores = np.zeros(len(cosine_sim))
    for idx in book_indices:
        sim_scores += cosine_sim[idx]
    
    # Sort the books based on the similarity scores
    sim_scores_with_indices = list(enumerate(sim_scores))
    sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top N most similar books
    top_indices = [i[0] for i in sim_scores_with_indices if i[0] not in book_indices][:num_recommendations]
    
    # Return the top N most similar books
    recommendations = books_df.iloc[top_indices].copy()
    
    # Add similarity scores
    recommendations['similarity_score'] = [sim_scores[i] for i in top_indices]
    
    return recommendations.sort_values('similarity_score', ascending=False)

def get_collaborative_recommendations(books_df: pd.DataFrame, 
                                     user_ratings: Dict, 
                                     user_id: str,
                                     num_recommendations: int = 10) -> pd.DataFrame:
    """
    Get collaborative recommendations for a user
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        user_ratings (Dict): Dictionary containing user ratings
        user_id (str): User ID to recommend for
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing recommended books
    """
    # Extract all user ratings
    ratings_data = []
    for uid, user_data in user_ratings.get('users', {}).items():
        for book_id, rating in user_data.get('ratings', {}).items():
            ratings_data.append({
                'user_id': uid,
                'book_id': int(book_id),
                'rating': rating
            })
    
    # If no ratings, return popular books
    if not ratings_data:
        return books_df.sort_values('num_ratings', ascending=False).head(num_recommendations)
    
    # Create ratings dataframe
    ratings_df = pd.DataFrame(ratings_data)
    
    # Create the user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='book_id',
        values='rating'
    ).fillna(0)
    
    # If user doesn't exist or not enough users, return popular books
    if len(user_item_matrix) < 2 or user_id not in user_item_matrix.index:
        return books_df.sort_values('num_ratings', ascending=False).head(num_recommendations)
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    # Get similar users
    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    
    # Get books rated by similar users but not by the target user
    user_rated_books = set(ratings_df[ratings_df['user_id'] == user_id]['book_id'])
    recommended_books = pd.DataFrame()
    
    for sim_user, sim_score in similar_users.items():
        # Skip users with low similarity
        if sim_score <= 0.1:
            continue
            
        # Get books rated highly by similar user
        sim_user_ratings = ratings_df[
            (ratings_df['user_id'] == sim_user) & 
            (ratings_df['rating'] >= 4)
        ]
        
        # Filter out books already rated by the target user
        new_recommendations = sim_user_ratings[
            ~sim_user_ratings['book_id'].isin(user_rated_books)
        ]
        
        # Add weighted score based on user similarity
        new_recommendations = new_recommendations.copy()
        new_recommendations['weighted_score'] = new_recommendations['rating'] * sim_score
        
        # Append to recommendations
        recommended_books = pd.concat([recommended_books, new_recommendations])
        
        if len(recommended_books) >= num_recommendations * 2:
            break
    
    # If we still don't have recommendations, return popular books
    if len(recommended_books) == 0:
        return books_df.sort_values('num_ratings', ascending=False).head(num_recommendations)
    
    # Aggregate and rank recommendations
    recommended_books = recommended_books.groupby('book_id').agg({
        'weighted_score': 'mean'
    }).reset_index()
    
    # Sort by weighted score
    recommended_books = recommended_books.sort_values('weighted_score', ascending=False)
    
    # Get top N book IDs
    top_book_ids = recommended_books['book_id'].head(num_recommendations).tolist()
    
    # Get full book details
    final_recommendations = books_df[books_df['book_id'].isin(top_book_ids)].copy()
    
    # Add weighted scores
    score_dict = dict(zip(recommended_books['book_id'], recommended_books['weighted_score']))
    final_recommendations['weighted_score'] = final_recommendations['book_id'].map(score_dict)
    
    return final_recommendations.sort_values('weighted_score', ascending=False)

def get_popularity_recommendations(books_df: pd.DataFrame, 
                                  genre: Optional[str] = None,
                                  num_recommendations: int = 10) -> pd.DataFrame:
    """
    Get popularity-based recommendations
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        genre (str, optional): Filter by genre
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing recommended books
    """
    recommendations = books_df.copy()
    
    # Calculate popularity score
    C = recommendations['rating'].mean()  # Mean rating across all books
    m = recommendations['num_ratings'].quantile(0.90)  # Minimum ratings required
    
    recommendations['popularity_score'] = recommendations.apply(
        lambda x: (x['num_ratings'] / (x['num_ratings'] + m) * x['rating']) + 
                  (m / (x['num_ratings'] + m) * C),
        axis=1
    )
    
    # Filter by genre if specified
    if genre:
        recommendations = recommendations[recommendations['genre'] == genre]
    
    # Sort by popularity score
    recommendations = recommendations.sort_values('popularity_score', ascending=False)
    
    # Return top N recommendations
    return recommendations.head(num_recommendations)

def get_hybrid_recommendations(books_df: pd.DataFrame,
                             user_ratings: Dict,
                             tfidf_matrix,
                             indices: pd.Series,
                             user_id: str,
                             book_ids: List[int] = None,
                             content_weight: float = 0.7,
                             collab_weight: float = 0.2,
                             popularity_weight: float = 0.1,
                             num_recommendations: int = 10) -> pd.DataFrame:
    """
    Get hybrid recommendations
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        user_ratings (Dict): Dictionary containing user ratings
        tfidf_matrix: TF-IDF matrix of book features
        indices (pd.Series): Series mapping book IDs to indices
        user_id (str): User ID to recommend for
        book_ids (List[int], optional): Book IDs for content-based filtering
        content_weight (float): Weight for content-based recommendations
        collab_weight (float): Weight for collaborative recommendations
        popularity_weight (float): Weight for popularity-based recommendations
        num_recommendations (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing recommended books
    """
    # Check if user has rated any books
    user_exists = user_id in user_ratings.get('users', {})
    user_has_ratings = False
    
    if user_exists:
        user_ratings_dict = user_ratings['users'][user_id].get('ratings', {})
        user_has_ratings = len(user_ratings_dict) > 0
    
    # Adjust weights based on available data
    if not user_has_ratings:
        content_weight = 0.8
        collab_weight = 0
        popularity_weight = 0.2
    
    if not book_ids:
        content_weight = 0
        collab_weight = 0.7
        popularity_weight = 0.3
        
        if not user_has_ratings:
            collab_weight = 0
            popularity_weight = 1.0
    
    # Get more recommendations than needed for each approach
    extra_factor = 2
    content_recs = pd.DataFrame()
    collab_recs = pd.DataFrame()
    popularity_recs = pd.DataFrame()
    
    # Get recommendations from each approach
    if content_weight > 0 and book_ids:
        content_recs = get_content_based_recommendations(
            books_df, tfidf_matrix, indices, book_ids, int(num_recommendations * extra_factor)
        )
        if len(content_recs) > 0:
            content_recs['content_score'] = content_recs['similarity_score']
            content_recs = content_recs.drop('similarity_score', axis=1)
    
    if collab_weight > 0 and user_has_ratings:
        collab_recs = get_collaborative_recommendations(
            books_df, user_ratings, user_id, int(num_recommendations * extra_factor)
        )
        if len(collab_recs) > 0:
            collab_recs['collab_score'] = collab_recs['weighted_score']
            collab_recs = collab_recs.drop('weighted_score', axis=1)
    
    if popularity_weight > 0:
        popularity_recs = get_popularity_recommendations(
            books_df, None, int(num_recommendations * extra_factor)
        )
    
    # Create a set of all recommended book IDs
    all_book_ids = set()
    if len(content_recs) > 0:
        all_book_ids.update(content_recs['book_id'].tolist())
    if len(collab_recs) > 0:
        all_book_ids.update(collab_recs['book_id'].tolist())
    if len(popularity_recs) > 0:
        all_book_ids.update(popularity_recs['book_id'].tolist())
    
    # Create a dataframe with all recommended books
    all_recs = books_df[books_df['book_id'].isin(all_book_ids)].copy()
    
    # Add scores from each recommender
    all_recs['content_score'] = 0
    all_recs['collab_score'] = 0
    all_recs['popularity_score'] = 0
    
    # Map scores from each recommender
    if len(content_recs) > 0:
        content_score_dict = dict(zip(content_recs['book_id'], content_recs['content_score']))
        all_recs['content_score'] = all_recs['book_id'].map(content_score_dict).fillna(0)
    
    if len(collab_recs) > 0:
        collab_score_dict = dict(zip(collab_recs['book_id'], collab_recs['collab_score']))
        all_recs['collab_score'] = all_recs['book_id'].map(collab_score_dict).fillna(0)
    
    if len(popularity_recs) > 0:
        # Normalize popularity scores
        max_pop_score = popularity_recs['popularity_score'].max()
        min_pop_score = popularity_recs['popularity_score'].min()
        pop_range = max_pop_score - min_pop_score
        if pop_range > 0:
            popularity_recs['popularity_score'] = (popularity_recs['popularity_score'] - min_pop_score) / pop_range
        
        pop_score_dict = dict(zip(popularity_recs['book_id'], popularity_recs['popularity_score']))
        all_recs['popularity_score'] = all_recs['book_id'].map(pop_score_dict).fillna(0)
    
    # Normalize other scores
    for score_col in ['content_score', 'collab_score']:
        if all_recs[score_col].sum() > 0:
            max_score = all_recs[score_col].max()
            min_score = all_recs[score_col].min()
            score_range = max_score - min_score
            if score_range > 0:
                all_recs[score_col] = (all_recs[score_col] - min_score) / score_range
    
    # Calculate weighted hybrid score
    all_recs['hybrid_score'] = (
        content_weight * all_recs['content_score'] +
        collab_weight * all_recs['collab_score'] +
        popularity_weight * all_recs['popularity_score']
    )
    
    # Sort by hybrid score
    all_recs = all_recs.sort_values('hybrid_score', ascending=False)
    
    # Exclude books that were used for content-based filtering
    if book_ids:
        all_recs = all_recs[~all_recs['book_id'].isin(book_ids)]
    
    # Return top N recommendations
    return all_recs.head(num_recommendations)

def get_recommendation_explanation(content_score: float, collab_score: float, popularity_score: float) -> str:
    """
    Generate an explanation for why a book was recommended
    
    Args:
        content_score (float): Content-based score
        collab_score (float): Collaborative score
        popularity_score (float): Popularity score
        
    Returns:
        str: Explanation text
    """
    explanation_parts = []
    
    # Check content similarity
    if content_score > 0.7:
        explanation_parts.append("It's similar to books you selected")
    elif content_score > 0.5:
        explanation_parts.append("It has some similar themes to books you selected")
    
    # Check collaborative filtering
    if collab_score > 0.7:
        explanation_parts.append("Users with similar taste rated this book highly")
    elif collab_score > 0.5:
        explanation_parts.append("Users with similar taste liked this book")
    
    # Check popularity
    if popularity_score > 0.8:
        explanation_parts.append("It's a popular and highly rated book")
    elif popularity_score > 0.6:
        explanation_parts.append("It has good ratings from many readers")
    
    # If no strong signals, provide a general explanation
    if not explanation_parts:
        explanation_parts.append("It might match your reading preferences")
    
    return ". ".join(explanation_parts) + "." 