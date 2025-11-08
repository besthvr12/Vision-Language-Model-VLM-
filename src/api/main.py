"""
FastAPI REST API for Smart Visual Commerce Platform
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DatasetLoader
from src.models.embeddings import CLIPEmbedder
from src.models.search import VisualSearchEngine
from src.models.vlm_client import VLMClient
from src.models.recommender import RecommendationEngine
from src.data.preprocessor import ImagePreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Smart Visual Commerce Platform API",
    description="AI-powered e-commerce intelligence API with visual search, recommendations, and product analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models (in production, use proper state management)
models = {}


# Pydantic models for request/response
class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, str]] = None


class SearchResult(BaseModel):
    product_id: str
    similarity: float
    product_name: str
    category: str
    price: float
    attributes: Dict[str, Any]


class RecommendationRequest(BaseModel):
    product_id: str
    top_n: int = 5
    strategy: str = "hybrid"  # content, attribute, complementary, hybrid


class QualityScore(BaseModel):
    overall: float
    sharpness: float
    brightness: float
    contrast: float
    resolution: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("🚀 Initializing Smart Visual Commerce Platform API...")

    try:
        # Load dataset
        print("Loading dataset...")
        loader = DatasetLoader()
        df = loader.create_sample_dataset(num_samples=100)
        models['df'] = df
        models['loader'] = loader

        # Initialize CLIP embedder
        print("Loading CLIP model...")
        embedder = CLIPEmbedder(model_name="ViT-B/32")
        models['embedder'] = embedder

        # Generate embeddings
        print("Generating embeddings...")
        descriptions = df.apply(
            lambda x: f"{x['category']} {x['baseColour']} {x['pattern']} for {x['gender']}",
            axis=1
        ).tolist()
        embeddings = embedder.encode_text(descriptions)
        models['embeddings'] = embeddings

        # Build search index
        print("Building search index...")
        search_engine = VisualSearchEngine(embedding_dim=embeddings.shape[1])
        search_engine.build_index(
            embeddings=embeddings,
            product_ids=df['id'].tolist(),
            metadata=df
        )
        models['search_engine'] = search_engine

        # Initialize recommender
        models['recommender'] = RecommendationEngine(df)

        # Initialize preprocessor
        models['preprocessor'] = ImagePreprocessor()

        print("✓ All models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        models['error'] = str(e)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": 'embedder' in models
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models": {
            "embedder": 'embedder' in models,
            "search_engine": 'search_engine' in models,
            "recommender": 'recommender' in models,
            "dataset": 'df' in models
        },
        "dataset_size": len(models.get('df', [])) if 'df' in models else 0
    }


@app.post("/search/text", response_model=List[SearchResult])
async def search_by_text(query: SearchQuery):
    """
    Search products using text query.

    Args:
        query: Search query with text, top_k, and optional filters

    Returns:
        List of matching products with similarity scores
    """
    if 'search_engine' not in models:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        # Generate query embedding
        query_embedding = models['embedder'].encode_text(query.query)

        # Search
        if query.filters:
            results = models['search_engine'].search_with_filters(
                query_embedding,
                top_k=query.top_k,
                filters=query.filters
            )
        else:
            results = models['search_engine'].search(
                query_embedding,
                top_k=query.top_k
            )

        # Format response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "product_id": result['id'],
                "similarity": result['similarity'],
                "product_name": result['productDisplayName'],
                "category": result['category'],
                "price": result['price'],
                "attributes": {
                    "color": result.get('baseColour'),
                    "gender": result.get('gender'),
                    "pattern": result.get('pattern'),
                    "season": result.get('season')
                }
            })

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50)
):
    """
    Search products using an uploaded image.

    Args:
        file: Image file
        top_k: Number of results to return

    Returns:
        List of matching products
    """
    if 'search_engine' not in models:
        raise HTTPException(status_code=503, detail="Search engine not initialized")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Generate image embedding
        image_embedding = models['embedder'].encode_image(image)

        # Search
        results = models['search_engine'].search(image_embedding, top_k=top_k)

        # Format response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "product_id": result['id'],
                "similarity": result['similarity'],
                "product_name": result['productDisplayName'],
                "category": result['category'],
                "price": result['price'],
                "attributes": {
                    "color": result.get('baseColour'),
                    "gender": result.get('gender')
                }
            })

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Get product recommendations.

    Args:
        request: Recommendation request with product_id, top_n, and strategy

    Returns:
        List of recommended products
    """
    if 'recommender' not in models:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    try:
        recommender = models['recommender']
        embeddings = models['embeddings']
        df = models['df']

        # Get recommendations based on strategy
        if request.strategy == "content":
            recs = recommender.content_based_recommendations(
                request.product_id,
                embeddings,
                df['id'].tolist(),
                top_n=request.top_n
            )
        elif request.strategy == "attribute":
            recs = recommender.attribute_based_recommendations(
                request.product_id,
                top_n=request.top_n
            )
        elif request.strategy == "complementary":
            recs = recommender.complementary_recommendations(
                request.product_id,
                top_n=request.top_n
            )
        else:  # hybrid
            recs = recommender.hybrid_recommendations(
                request.product_id,
                embeddings,
                df['id'].tolist(),
                top_n=request.top_n
            )

        return recs

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quality/assess", response_model=QualityScore)
async def assess_image_quality(file: UploadFile = File(...)):
    """
    Assess image quality for e-commerce use.

    Args:
        file: Image file to assess

    Returns:
        Quality scores and metrics
    """
    if 'preprocessor' not in models:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Assess quality
        quality = models['preprocessor'].assess_image_quality(image)

        return quality

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attributes/extract")
async def extract_attributes(file: UploadFile = File(...)):
    """
    Extract product attributes from image using VLM.

    Note: Requires VLM API key to be configured.

    Args:
        file: Product image file

    Returns:
        Extracted attributes
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Initialize VLM client (requires API key)
        vlm_client = VLMClient(provider="openai")

        # Extract attributes
        attributes = vlm_client.extract_attributes(image)

        return attributes

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Attribute extraction failed: {str(e)}. Ensure VLM API key is configured."
        )


@app.get("/products")
async def list_products(
    category: Optional[str] = None,
    color: Optional[str] = None,
    gender: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = Query(50, ge=1, le=100)
):
    """
    List products with optional filters.

    Args:
        category: Filter by category
        color: Filter by color
        gender: Filter by gender
        min_price: Minimum price
        max_price: Maximum price
        limit: Maximum number of results

    Returns:
        List of products
    """
    if 'loader' not in models or 'df' not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        df = models['df']
        loader = models['loader']

        # Apply filters
        filtered_df = loader.filter_products(
            df,
            category=category,
            color=color,
            gender=gender,
            price_min=min_price,
            price_max=max_price
        )

        # Limit results
        filtered_df = filtered_df.head(limit)

        # Convert to dict
        products = filtered_df.to_dict('records')

        return {
            "count": len(products),
            "products": products
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """
    Get detailed product information.

    Args:
        product_id: Product ID

    Returns:
        Product details
    """
    if 'df' not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        df = models['df']
        product = df[df['id'] == product_id]

        if len(product) == 0:
            raise HTTPException(status_code=404, detail="Product not found")

        return product.iloc[0].to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get dataset statistics."""
    if 'loader' not in models or 'df' not in models:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        df = models['df']
        loader = models['loader']

        stats = loader.get_statistics(df)

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
