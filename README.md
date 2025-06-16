# 📚 Book Recommendation Engine

A comprehensive book recommendation engine built with Streamlit that suggests books based on user preferences using multiple recommendation algorithms.

## ✨ Features

- **Multiple Recommendation Methods**:
  - Content-based filtering (based on book features)
  - Collaborative filtering (based on user ratings)
  - Popularity-based recommendations
  - Hybrid approach (combining all methods)

- **Interactive User Interface**:
  - Book search and filtering
  - Book rating system
  - Favorites and reading list management
  - Detailed book information with cover images
  - Recommendation explanations

- **Data Visualization**:
  - Genre distribution charts
  - Rating distributions
  - User preference visualization
  - Recommendation confidence display

- **User Management**:
  - Personal profiles for users
  - Rating history tracking
  - Favorites and reading list saving

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/book-recommendation-engine.git
   cd book-recommendation-engine
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## 📊 Data

The application uses a dataset of 100+ popular books with the following information:
- Title, Author, Genre, Subgenre
- Publication Year, ISBN, Page Count
- Description, Average Rating, Number of Ratings
- Cover Image URL, Series information, Reading Level

## 🧠 Recommendation Algorithms

### Content-Based Filtering
Recommends books based on similarity between book features (title, author, genre, description) using TF-IDF and cosine similarity.

### Collaborative Filtering
Recommends books based on ratings from users with similar preferences.

### Popularity-Based
Recommends trending and highly-rated books.

### Hybrid Approach
Combines all methods with configurable weights for each approach.

## 🛠️ Project Structure

```
book_recommendation_engine/
├── app.py                 # Main Streamlit app
├── data/
│   ├── books.csv         # Book dataset
│   └── user_ratings.json # User ratings storage
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Data loading functions
│   ├── recommender.py    # Recommendation algorithms
│   └── visualizations.py # Chart functions
├── assets/
│   └── book_covers/      # Book cover images (optional)
├── requirements.txt      # Dependencies
└── README.md
```

## 📝 Usage

1. **Home Page**: Overview of the application and popular books
2. **Search Books**: Search and filter books by various criteria
3. **Your Profile**: View and manage your ratings, favorites, and reading list
4. **Recommendations**: Get personalized book recommendations
5. **Explore Books**: Browse books by genre, ratings, and more

## 🔧 Customization

- Adjust recommendation parameters in the sidebar
- Filter books by genre, publication year, and minimum rating
- Choose between different recommendation methods

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Authors

- Your Name - Initial work

## 🙏 Acknowledgements

- Book data compiled from various public sources
- Cover images sourced from Goodreads API 