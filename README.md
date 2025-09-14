Book Recommendation Engine

This project is a comprehensive book recommendation system built using Streamlit. It suggests books to users based on their preferences using multiple recommendation algorithms.
Features

    Multiple recommendation methods including content-based filtering, collaborative filtering, popularity-based recommendations, and a hybrid approach

    Interactive user interface for book searching, filtering, and rating

    Favorites and reading list management system

    Detailed book information with cover images

    Data visualizations for genre distributions, ratings, and recommendation confidence

    User profiles with rating history tracking

Technologies Used

    Python 3.8+

    Streamlit for web interface

    Machine learning algorithms for recommendations

    Data processing libraries

Getting Started
Prerequisites

    Python 3.8 or higher

    pip package manager

Installation

    Clone the repository:

text
git clone https://github.com/NoLongerHumanHQ/Book-Recommendation-Engine.git
cd Book-Recommendation-Engine

Install required dependencies:

text
pip install -r requirements.txt

Run the Streamlit application:

text
streamlit run app.py

Open your web browser and navigate to:

    text
    http://localhost:8501

Data

The application uses a dataset of over 100 popular books containing:

    Title, author, genre, and subgenre information

    Publication year, ISBN, and page count

    Book descriptions and average ratings

    Cover image URLs and series information

    Reading level classifications

Recommendation Methods
Content-Based Filtering

Recommends books based on similarity between book features such as title, author, genre, and description using TF-IDF and cosine similarity algorithms.
Collaborative Filtering

Suggests books based on ratings from users with similar reading preferences and patterns.
Popularity-Based Recommendations

Recommends trending and highly-rated books based on overall popularity metrics.
Hybrid Approach

Combines all recommendation methods with configurable weights to provide more accurate suggestions.
Project Structure

text
book_recommendation_engine/
├── app.py                    # Main Streamlit application
├── data/
│   ├── books.csv            # Book dataset
│   └── user_ratings.json    # User ratings storage
├── utils/
│   ├── data_loader.py       # Data loading functions
│   ├── recommender.py       # Recommendation algorithms
│   └── visualizations.py    # Chart and graph functions
├── assets/
│   └── book_covers/         # Book cover images
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation

How to Use

    Home Page: Browse overview and popular books

    Search Books: Search and filter books by various criteria

    User Profile: Manage ratings, favorites, and reading lists

    Recommendations: Get personalized book suggestions

    Explore Books: Browse by genre, ratings, and publication year

Customization

    Adjust recommendation algorithm parameters through the sidebar

    Filter books by genre, publication year, and minimum rating

    Choose between different recommendation methods

    Modify weights for the hybrid recommendation approach

Contributing

Contributions are welcome. Please feel free to:

    Report bugs and issues

    Suggest new features

    Submit pull requests

    Improve documentation

License

This project is licensed under the MIT License.
Author

Created by NoLongerHumanHQ for educational and practical purposes.
Acknowledgments

    Book data compiled from various public sources

    Cover images sourced from Goodreads API
