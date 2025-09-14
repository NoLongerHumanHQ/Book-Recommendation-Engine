# Book Recommendation Engine

This project is a comprehensive book recommendation system built with **Streamlit**. It suggests books to users based on their preferences using multiple recommendation algorithms.

## Features
- Multiple recommendation methods: content-based, collaborative, popularity-based, and hybrid  
- Interactive UI for book search, filtering, and rating  
- Favorites and reading-list management  
- Detailed book pages with cover images  
- Visualizations for genre distribution, rating trends, and recommendation confidence  
- User profiles with rating history  

## Technologies Used
- Python 3.8+  
- Streamlit (web interface)  
- Scikit-learn, Pandas, NumPy (data processing & ML)  

## Getting Started

### Prerequisites
- Python 3.8 or higher  
- pip package manager  

### Installation
git clone https://github.com/NoLongerHumanHQ/Book-Recommendation-Engine.git
cd Book-Recommendation-Engine
pip install -r requirements.txt
streamlit run app.py

Open your browser at `http://localhost:8501`.

## Data
Dataset of 100+ books containing  
*Title, Author, Genre, Subgenre, Year, ISBN, Page Count, Description, Avg Rating, Ratings Count, Cover URL, Series, Reading Level.*

## Recommendation Methods
- **Content-Based:** TF-IDF + cosine similarity on book features  
- **Collaborative Filtering:** User-based rating similarity  
- **Popularity-Based:** Trending and highly rated books  
- **Hybrid:** Weighted combination of all methods  

## Project Structure
book_recommendation_engine/
├── app.py # Streamlit app
├── data/
│ ├── books.csv # Book dataset
│ └── user_ratings.json # Stored user ratings
├── utils/
│ ├── data_loader.py # Load & preprocess data
│ ├── recommender.py # Algorithms
│ └── visualizations.py # Charts
├── assets/
│ └── book_covers/ # Optional images
├── requirements.txt
└── README.md


## Usage
1. **Home:** View popular books  
2. **Search Books:** Filter by genre, author, year, rating  
3. **Profile:** Manage ratings, favorites, reading list  
4. **Recommendations:** Get personalized suggestions  
5. **Explore:** Browse by genre and popularity  

## Customization
- Adjust algorithm weights and filters in the sidebar  
- Add or replace the `books.csv` dataset to extend recommendations  

## Contributing
Pull requests are welcome. Please open an issue first to discuss significant changes.

## License
MIT License

## Author
Created by **NoLongerHumanHQ**.
