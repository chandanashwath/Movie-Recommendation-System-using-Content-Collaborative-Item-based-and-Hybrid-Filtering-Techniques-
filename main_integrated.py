"""
FastAPI Backend - Integrated with Recommendation Engine
Uses modular recommendation_engine.py for better code organization
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

# Import our recommendation engine
from recommendation_engine import get_engine, MovieRecommendationEngine

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="AI-powered movie recommendation system using ML algorithms",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Movie(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: str
    director: str
    actors: str
    runtime: int
    avg_rating: float
    budget_millions: int

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: str
    avg_rating: float
    similarity_score: Optional[float] = None
    predicted_rating: Optional[float] = None
    hybrid_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[MovieRecommendation]
    count: int
    method: str
    timestamp: str

class StatsResponse(BaseModel):
    total_movies: int
    total_users: int
    total_ratings: int
    avg_rating: float
    sparsity: float
    genres: List[str]

# Global recommendation engine instance
engine: Optional[MovieRecommendationEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize recommendation engine on startup"""
    global engine
    print("🚀 Starting Movie Recommendation API...")
    print("📦 Loading recommendation engine...")
    
    try:
        # Get singleton engine instance
        engine = get_engine(
            movies_path='movies.csv',
            ratings_path='ratings.csv',
            users_path='users.csv'
        )
        print("✅ API ready to serve recommendations!")
    except Exception as e:
        print(f"❌ Error initializing engine: {e}")
        engine = None


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy" if engine is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "engine_loaded": engine is not None
    }


# Get all movies
@app.get("/api/movies", response_model=List[Movie])
async def get_movies(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get all movies with pagination"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    movies = engine.movies.iloc[offset:offset + limit].to_dict('records')
    return movies


# Get movie by ID
@app.get("/api/movies/{movie_id}", response_model=Movie)
async def get_movie(movie_id: int):
    """Get movie details by ID"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    movie_info = engine.get_movie_info(movie_id)
    
    if movie_info is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return movie_info


# Search movies
@app.get("/api/movies/search/{query}")
async def search_movies(query: str, limit: int = Query(10, ge=1, le=50)):
    """Search movies by title or genre"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    # Search in title and genres
    mask = (
        engine.movies['title'].str.contains(query, case=False, na=False) |
        engine.movies['genres'].str.contains(query, case=False, na=False)
    )
    
    results = engine.movies[mask].head(limit).to_dict('records')
    
    return {
        "query": query,
        "count": len(results),
        "results": results
    }


# Content-based recommendations
@app.get("/api/recommendations/content/{movie_id}", response_model=RecommendationResponse)
async def get_content_recommendations(
    movie_id: int,
    top_n: int = Query(10, ge=1, le=50)
):
    """Get content-based recommendations for a movie"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Use engine method
        recommendations = engine.get_content_recommendations(movie_id, top_n=top_n)
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        # Convert to response model
        recs_list = []
        for _, row in recommendations.iterrows():
            recs_list.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=row['title'],
                year=int(row['year']),
                genres=row['genres'],
                avg_rating=float(row['avg_rating']),
                similarity_score=float(row['similarity_score'])
            ))
        
        return RecommendationResponse(
            success=True,
            recommendations=recs_list,
            count=len(recs_list),
            method="content-based",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# User-based collaborative filtering
@app.get("/api/recommendations/user/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int,
    top_n: int = Query(10, ge=1, le=50)
):
    """Get user-based collaborative filtering recommendations"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Use engine method
        recommendations = engine.get_user_based_recommendations(user_id, top_n=top_n)
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="User not found or no recommendations available")
        
        # Convert to response model
        recs_list = []
        for _, row in recommendations.iterrows():
            recs_list.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=row['title'],
                year=int(row['year']),
                genres=row['genres'],
                avg_rating=float(row['avg_rating']),
                predicted_rating=float(row['predicted_rating'])
            ))
        
        return RecommendationResponse(
            success=True,
            recommendations=recs_list,
            count=len(recs_list),
            method="user-based-collaborative",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Item-based collaborative filtering
@app.get("/api/recommendations/item/{user_id}", response_model=RecommendationResponse)
async def get_item_recommendations(
    user_id: int,
    top_n: int = Query(10, ge=1, le=50)
):
    """Get item-based collaborative filtering recommendations"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Use engine method
        recommendations = engine.get_item_based_recommendations(user_id, top_n=top_n)
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="User not found or no recommendations available")
        
        # Convert to response model
        recs_list = []
        for _, row in recommendations.iterrows():
            recs_list.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=row['title'],
                year=int(row['year']),
                genres=row['genres'],
                avg_rating=float(row['avg_rating']),
                predicted_rating=float(row['predicted_rating'])
            ))
        
        return RecommendationResponse(
            success=True,
            recommendations=recs_list,
            count=len(recs_list),
            method="item-based-collaborative",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Hybrid recommendations
@app.get("/api/recommendations/hybrid/{user_id}", response_model=RecommendationResponse)
async def get_hybrid_recommendations(
    user_id: int,
    top_n: int = Query(10, ge=1, le=50),
    content_weight: float = Query(0.3, ge=0, le=1),
    collab_weight: float = Query(0.7, ge=0, le=1)
):
    """Get hybrid recommendations combining multiple approaches"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Use engine method
        recommendations = engine.get_hybrid_recommendations(
            user_id, 
            top_n=top_n, 
            content_weight=content_weight, 
            collab_weight=collab_weight
        )
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="User not found or no recommendations available")
        
        # Convert to response model
        recs_list = []
        for _, row in recommendations.iterrows():
            recs_list.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=row['title'],
                year=int(row['year']),
                genres=row['genres'],
                avg_rating=float(row['avg_rating']),
                predicted_rating=float(row['predicted_rating']),
                hybrid_score=float(row['hybrid_score'])
            ))
        
        return RecommendationResponse(
            success=True,
            recommendations=recs_list,
            count=len(recs_list),
            method="hybrid",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get user's ratings
@app.get("/api/users/{user_id}/ratings")
async def get_user_ratings(user_id: int):
    """Get all ratings by a user"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    user_ratings = engine.get_user_ratings(user_id)
    
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail="No ratings found for this user")
    
    return {
        "user_id": user_id,
        "count": len(user_ratings),
        "ratings": user_ratings.to_dict('records')
    }


# Get popular movies
@app.get("/api/movies/popular/top")
async def get_popular_movies(
    top_n: int = Query(10, ge=1, le=50),
    min_ratings: int = Query(5, ge=1, le=20)
):
    """Get most popular movies based on ratings"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    recommendations = engine.get_popular_movies(top_n=top_n, min_ratings=min_ratings)
    
    return {
        "count": len(recommendations),
        "movies": recommendations.to_dict('records')
    }


# Get statistics
@app.get("/api/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system statistics"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    stats = engine.get_statistics()
    
    return StatsResponse(**stats)


if __name__ == "__main__":
    print("Starting FastAPI server with integrated recommendation engine...")
    print("📚 API Documentation: http://localhost:8000/api/docs")
    print("🔄 ReDoc: http://localhost:8000/api/redoc")
    
    uvicorn.run(
        "main_integrated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
