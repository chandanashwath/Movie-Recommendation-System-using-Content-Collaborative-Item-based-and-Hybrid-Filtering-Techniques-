"""
Modular Recommendation Engine
Can be used standalone or imported by FastAPI backend
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #It converts text documents into numerical feature vectors using TF-IDF so ML models can understand text.
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
'''
cosine_similarity measures how similar two vectors are based on the angle between them, not their magnitude.
linear_kernel computes the dot product of two vectors.
Both are used to measure similarity between vectors.
'''
from typing import Dict, List, Tuple, Optional #optional - A variable can either have a specific type OR be None.
import warnings
warnings.filterwarnings('ignore')


class MovieRecommendationEngine:
    """
    Encapsulated recommendation engine that can be used by any application
    """
    
    def __init__(self, movies_path: str = 'movies.csv', 
                 ratings_path: str = 'ratings.csv', 
                 users_path: str = 'users.csv'):
        """
        Initialize the recommendation engine
        
        Args:
            movies_path: Path to movies CSV file
            ratings_path: Path to ratings CSV file
            users_path: Path to users CSV file
        """
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.users_path = users_path
        
        # DataFrames
        self.movies = None
        self.ratings = None
        self.users = None
        self.user_movie_matrix = None
        
        # Similarity matrices
        self.content_similarity = None
        self.user_similarity = None
        self.item_similarity = None
        self.tfidf_matrix = None
        
        # Load data
        self._load_data()
    
    def _load_data(self) -> bool:
        """Load all datasets"""
        try:
            self.movies = pd.read_csv(self.movies_path)
            self.ratings = pd.read_csv(self.ratings_path)
            self.users = pd.read_csv(self.users_path)
            
            # Create user-movie matrix
            self.user_movie_matrix = self.ratings.pivot_table(
                index='user_id',
                columns='movie_id',
                values='rating'
            ).fillna(0)
            
            print(f"✅ Loaded {len(self.movies)} movies, {len(self.users)} users, {len(self.ratings)} ratings")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def build_content_similarity(self) -> bool:
        """Build content-based similarity matrix"""
        try:
            # Combine features
            self.movies['combined_features'] = (
                self.movies['genres'].fillna('') + ' ' +
                self.movies['director'].fillna('') + ' ' +
                self.movies['actors'].fillna('')
            )
            
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(self.movies['combined_features'])
            
            # Compute cosine similarity
            self.content_similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
            
            print(f"✅ Content similarity matrix built: {self.content_similarity.shape}")
            return True
        except Exception as e:
            print(f"❌ Error building content similarity: {e}")
            return False
    
    def build_collaborative_filtering(self) -> bool:
        """Build collaborative filtering models"""
        try:
            # User-based similarity
            user_matrix = self.user_movie_matrix.values
            self.user_similarity = cosine_similarity(user_matrix)
            
            # Item-based similarity
            item_matrix = self.user_movie_matrix.T.values
            self.item_similarity = cosine_similarity(item_matrix)
            
            print(f"✅ User similarity: {self.user_similarity.shape}")
            print(f"✅ Item similarity: {self.item_similarity.shape}")
            return True
        except Exception as e:
            print(f"❌ Error building collaborative filtering: {e}")
            return False
    
    def initialize(self) -> bool:
        """Build all models - call this once at startup"""
        print("🔧 Initializing recommendation engine...")
        success = True
        success = success and self.build_content_similarity()
        success = success and self.build_collaborative_filtering()
        if success:
            print("✅ Recommendation engine ready!")
        return success
    
    def get_content_recommendations(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get content-based recommendations for a movie
        
        Args:
            movie_id: ID of the movie to base recommendations on
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        if self.content_similarity is None:
            self.build_content_similarity()
        
        try:
            movie_idx = self.movies[self.movies['movie_id'] == movie_id].index[0]
        except IndexError:
            return pd.DataFrame()
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        # Get movie indices and scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return recommendations
        recommendations = self.movies.iloc[movie_indices].copy()
        recommendations['similarity_score'] = similarity_scores
        
        return recommendations[['movie_id', 'title', 'year', 'genres', 'avg_rating', 'similarity_score']]
    
    def get_user_based_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get user-based collaborative filtering recommendations
        
        Args:
            user_id: ID of the user to recommend for
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        if self.user_similarity is None:
            self.build_collaborative_filtering()
        
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
        
        # Get user index
        user_idx = list(self.user_movie_matrix.index).index(user_id)
        
        # Find similar users
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:11]
        
        # Get unrated movies
        user_ratings = self.user_movie_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings
        predictions = {}
        for movie_id in unrated_movies:
            similar_user_ratings = []
            similarities = []
            
            for sim_user_idx in similar_users:
                sim_user_id = self.user_movie_matrix.index[sim_user_idx]
                rating = self.user_movie_matrix.loc[sim_user_id, movie_id]
                
                if rating > 0:
                    similar_user_ratings.append(rating)
                    similarities.append(self.user_similarity[user_idx][sim_user_idx])
            
            if similar_user_ratings:
                predictions[movie_id] = np.average(similar_user_ratings, weights=similarities)
        
        # Sort and get top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_predictions:
            return pd.DataFrame()
        
        # Create response
        recommended_ids = [movie_id for movie_id, _ in sorted_predictions]
        recommendations = self.movies[self.movies['movie_id'].isin(recommended_ids)].copy()
        recommendations['predicted_rating'] = recommendations['movie_id'].map(dict(sorted_predictions))
        
        return recommendations[['movie_id', 'title', 'year', 'genres', 'avg_rating', 'predicted_rating']]
    
    def get_item_based_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get item-based collaborative filtering recommendations
        
        Args:
            user_id: ID of the user to recommend for
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        if self.item_similarity is None:
            self.build_collaborative_filtering()
        
        if user_id not in self.user_movie_matrix.index:
            return pd.DataFrame()
        
        # Get user's rated movies
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return pd.DataFrame()
        
        # Get unrated movies
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings
        predictions = {}
        for movie_id in unrated_movies:
            movie_idx = list(self.user_movie_matrix.columns).index(movie_id)
            
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_id, rating in rated_movies.items():
                rated_idx = list(self.user_movie_matrix.columns).index(rated_id)
                similarity = self.item_similarity[movie_idx][rated_idx]
                
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            if similarity_sum > 0:
                predictions[movie_id] = weighted_sum / similarity_sum
        
        # Sort and get top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_predictions:
            return pd.DataFrame()
        
        # Create response
        recommended_ids = [movie_id for movie_id, _ in sorted_predictions]
        recommendations = self.movies[self.movies['movie_id'].isin(recommended_ids)].copy()
        recommendations['predicted_rating'] = recommendations['movie_id'].map(dict(sorted_predictions))
        
        return recommendations[['movie_id', 'title', 'year', 'genres', 'avg_rating', 'predicted_rating']]
    
    def get_hybrid_recommendations(self, user_id: int, top_n: int = 10, 
                                   content_weight: float = 0.3, 
                                   collab_weight: float = 0.7) -> pd.DataFrame:
        """
        Get hybrid recommendations combining multiple approaches
        
        Args:
            user_id: ID of the user to recommend for
            top_n: Number of recommendations to return
            content_weight: Weight for content-based recommendations
            collab_weight: Weight for collaborative recommendations
            
        Returns:
            DataFrame with recommended movies
        """
        # Get item-based recommendations
        collab_recs = self.get_item_based_recommendations(user_id, top_n=top_n*2)
        
        if collab_recs.empty:
            return pd.DataFrame()
        
        # Get user's favorite genres
        user_ratings_df = self.ratings[self.ratings['user_id'] == user_id]
        highly_rated = user_ratings_df[user_ratings_df['rating'] >= 4.0]['movie_id'].values
        
        favorite_genres = set()
        for movie_id in highly_rated:
            genres = self.movies[self.movies['movie_id'] == movie_id]['genres'].values
            if len(genres) > 0:
                favorite_genres.update(genres[0].split('|'))
        
        # Genre boost
        def genre_boost(row):
            movie_genres = set(row['genres'].split('|'))
            overlap = len(movie_genres.intersection(favorite_genres))
            return overlap * 0.3
        
        collab_recs['genre_boost'] = collab_recs.apply(genre_boost, axis=1)
        
        # Calculate hybrid score
        collab_recs['hybrid_score'] = (
            collab_recs['predicted_rating'] * collab_weight +
            collab_recs['genre_boost']
        )
        
        # Sort by hybrid score
        recommendations = collab_recs.sort_values('hybrid_score', ascending=False).head(top_n)
        
        return recommendations[['movie_id', 'title', 'year', 'genres', 'avg_rating', 'predicted_rating', 'hybrid_score']]
    
    def get_popular_movies(self, top_n: int = 10, min_ratings: int = 5) -> pd.DataFrame:
        """Get most popular movies based on ratings"""
        movie_stats = self.ratings.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movie_stats.columns = ['movie_id', 'user_avg_rating', 'num_ratings']
        
        # Filter movies with minimum ratings
        popular = movie_stats[movie_stats['num_ratings'] >= min_ratings]
        popular = popular.sort_values('user_avg_rating', ascending=False).head(top_n)
        
        # Merge with movie details
        recommendations = self.movies.merge(popular, on='movie_id')
        
        return recommendations
    
    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get detailed information about a movie"""
        movie = self.movies[self.movies['movie_id'] == movie_id]
        
        if movie.empty:
            return None
        
        return movie.iloc[0].to_dict()
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Get all ratings by a user"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id].merge(
            self.movies[['movie_id', 'title', 'genres']], 
            on='movie_id'
        )
        
        return user_ratings.sort_values('rating', ascending=False)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        all_genres = set()
        for genres_str in self.movies['genres']:
            all_genres.update(genres_str.split('|'))
        
        sparsity = 1 - (len(self.ratings) / (len(self.users) * len(self.movies)))
        
        return {
            'total_movies': len(self.movies),
            'total_users': len(self.users),
            'total_ratings': len(self.ratings),
            'avg_rating': float(self.ratings['rating'].mean()),
            'sparsity': float(sparsity),
            'genres': sorted(list(all_genres))
        }


# Singleton instance for use in FastAPI
_engine_instance = None

def get_engine(movies_path: str = 'movies.csv',
               ratings_path: str = 'ratings.csv',
               users_path: str = 'users.csv') -> MovieRecommendationEngine:
    """
    Get or create singleton instance of recommendation engine
    This ensures we only load data once
    """
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = MovieRecommendationEngine(
            movies_path=movies_path,
            ratings_path=ratings_path,
            users_path=users_path
        )
        _engine_instance.initialize()
    
    return _engine_instance


# For standalone usage
if __name__ == "__main__":
    print("🎬 Initializing Movie Recommendation Engine\n")
    
    # Create engine
    engine = MovieRecommendationEngine()
    engine.initialize()
    
    print("\n" + "="*60)
    print("DEMO: Content-Based Recommendations")
    print("="*60)
    
    movie_id = 1
    movie_info = engine.get_movie_info(movie_id)
    print(f"\n🎥 Based on: {movie_info['title']} ({movie_info['year']})")
    print(f"   Genres: {movie_info['genres']}")
    
    recs = engine.get_content_recommendations(movie_id, top_n=5)
    print("\n📋 Similar Movies:")
    print(recs.to_string(index=False))
    
    print("\n" + "="*60)
    print("DEMO: Hybrid Recommendations")
    print("="*60)
    
    user_id = 10
    print(f"\n👤 Recommendations for User ID: {user_id}")
    
    recs = engine.get_hybrid_recommendations(user_id, top_n=5)
    print("\n📋 Recommended Movies:")
    print(recs.to_string(index=False))
    
    print("\n✅ Demo complete!")
