import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional

def plot_genre_distribution(books_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing genre distribution
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        go.Figure: Plotly figure for genre distribution
    """
    # Count number of books in each genre
    genre_counts = books_df['genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    # Create a bar chart
    fig = px.bar(
        genre_counts, 
        x='Genre', 
        y='Count',
        title='Number of Books by Genre',
        color='Count',
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def plot_subgenre_distribution(books_df: pd.DataFrame, genre: str) -> go.Figure:
    """
    Create a bar chart showing subgenre distribution for a specific genre
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        genre (str): Genre to filter by
        
    Returns:
        go.Figure: Plotly figure for subgenre distribution
    """
    # Filter by genre
    filtered_df = books_df[books_df['genre'] == genre]
    
    # Count number of books in each subgenre
    subgenre_counts = filtered_df['subgenre'].value_counts().reset_index()
    subgenre_counts.columns = ['Subgenre', 'Count']
    
    # Create a bar chart
    fig = px.bar(
        subgenre_counts, 
        x='Subgenre', 
        y='Count',
        title=f'Number of Books by Subgenre in {genre}',
        color='Count',
        color_continuous_scale='viridis',
        labels={'Count': 'Number of Books', 'Subgenre': 'Subgenre'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return fig

def plot_rating_distribution(books_df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of book ratings
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        go.Figure: Plotly figure for rating distribution
    """
    # Create a histogram
    fig = px.histogram(
        books_df, 
        x='rating',
        nbins=20,
        title='Distribution of Book Ratings',
        color_discrete_sequence=['#1f77b4']
    )
    
    return fig

def plot_publication_year_distribution(books_df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of publication years
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        go.Figure: Plotly figure for publication year distribution
    """
    # Create a histogram
    fig = px.histogram(
        books_df, 
        x='publication_year',
        nbins=50,
        title='Distribution of Publication Years',
        color_discrete_sequence=['#2ca02c'],
        labels={'publication_year': 'Publication Year', 'count': 'Number of Books'}
    )
    
    # Update layout
    fig.update_layout(
        bargap=0.1,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_rating_vs_popularity(books_df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of ratings vs. number of ratings
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        
    Returns:
        go.Figure: Plotly figure for rating vs. popularity
    """
    # Create a scatter plot
    fig = px.scatter(
        books_df, 
        x='rating', 
        y='num_ratings',
        color='genre',
        size='page_count',
        hover_name='title',
        hover_data=['author', 'publication_year'],
        title='Book Ratings vs. Popularity',
        labels={
            'rating': 'Average Rating',
            'num_ratings': 'Number of Ratings',
            'genre': 'Genre',
            'page_count': 'Page Count'
        }
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_user_genre_preferences(user_rated_books: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing user's genre preferences
    
    Args:
        user_rated_books (pd.DataFrame): DataFrame containing books rated by the user
        
    Returns:
        go.Figure: Plotly figure for user genre preferences
    """
    # If no rated books, return empty figure
    if len(user_rated_books) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No rated books yet', x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count number of books in each genre
    genre_counts = user_rated_books['genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    # Create a pie chart
    fig = px.pie(
        genre_counts, 
        values='Count', 
        names='Genre',
        title='Your Genre Preferences'
    )
    
    return fig

def plot_user_rating_distribution(user_rated_books: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of user's ratings
    
    Args:
        user_rated_books (pd.DataFrame): DataFrame containing books rated by the user
        
    Returns:
        go.Figure: Plotly figure for user rating distribution
    """
    # If no rated books, return empty figure
    if len(user_rated_books) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text='No rated books yet',
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create a histogram
    fig = px.histogram(
        user_rated_books, 
        x='user_rating',
        nbins=5,
        range_x=[0.5, 5.5],
        title='Distribution of Your Ratings',
        color_discrete_sequence=['#d62728'],
        labels={'user_rating': 'Your Rating', 'count': 'Number of Books'}
    )
    
    # Update layout
    fig.update_layout(
        bargap=0.1,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_recommendation_scores(recommendations: pd.DataFrame, 
                              score_cols: List[str] = ['content_score', 'collab_score', 'popularity_score']) -> go.Figure:
    """
    Create a bar chart showing recommendation scores for each book
    
    Args:
        recommendations (pd.DataFrame): DataFrame containing recommended books
        score_cols (List[str]): List of score columns to include
        
    Returns:
        go.Figure: Plotly figure for recommendation scores
    """
    # If no recommendations, return empty figure
    if len(recommendations) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text='No recommendations yet',
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Prepare data for plotting
    plot_data = recommendations[['title'] + score_cols].copy()
    plot_data = plot_data.set_index('title')
    
    # Create a bar chart
    fig = px.bar(
        plot_data,
        barmode='group',
        title='Recommendation Score Components',
        labels={
            'value': 'Score',
            'variable': 'Score Type',
            'title': 'Book Title'
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_recommendation_confidence(recommendations: pd.DataFrame) -> go.Figure:
    """
    Create a gauge chart showing overall confidence in recommendations
    
    Args:
        recommendations (pd.DataFrame): DataFrame containing recommended books
        
    Returns:
        go.Figure: Plotly figure for recommendation confidence
    """
    # If no recommendations, return empty figure
    if len(recommendations) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No recommendations yet', x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate confidence score (average of hybrid scores)
    confidence = recommendations['hybrid_score'].mean()
    
    # Scale to 0-100
    confidence_pct = min(100, max(0, confidence * 100))
    
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Recommendation Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 33], 'color': "#f7bbbb"},
                {'range': [33, 66], 'color': "#ffdd99"},
                {'range': [66, 100], 'color': "#c8f7c5"}
            ]
        }
    ))
    
    return fig

def create_wordcloud(books_df: pd.DataFrame, column: str = 'description') -> None:
    """
    Create and display a word cloud from book descriptions
    
    Args:
        books_df (pd.DataFrame): DataFrame containing book data
        column (str): Column to use for word cloud (default: 'description')
    """
    try:
        # Combine all text into a single string
        text = ' '.join(books_df[column].fillna('').astype(str).tolist())
        
        # Only import wordcloud if needed
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            collocations=False
        ).generate(text)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    except ImportError:
        st.warning("Please install wordcloud library: pip install wordcloud")
    except Exception as e:
        st.error(f"Error creating word cloud: {e}") 