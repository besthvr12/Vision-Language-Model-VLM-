"""
Smart Visual Commerce Platform - Streamlit Web Interface

A comprehensive e-commerce AI platform featuring:
- Visual product search
- Attribute extraction
- Scene understanding
- Product recommendations
- Quality assessment
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from src.data.loader import DatasetLoader
from src.data.preprocessor import ImagePreprocessor
from src.models.embeddings import CLIPEmbedder
from src.models.search import VisualSearchEngine
from src.models.vlm_client import VLMClient
from src.models.recommender import RecommendationEngine


# Page configuration
st.set_page_config(
    page_title="Smart Visual Commerce Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
@st.cache_resource
def load_data():
    """Load and cache dataset."""
    loader = DatasetLoader()
    df = loader.create_sample_dataset(num_samples=100)
    return df, loader


@st.cache_resource
def initialize_models():
    """Initialize and cache models."""
    try:
        embedder = CLIPEmbedder(model_name="ViT-B/32")
        return embedder, ImagePreprocessor()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_resource
def build_search_index(_df, _embedder):
    """Build and cache search index."""
    # Generate embeddings from descriptions
    descriptions = _df.apply(
        lambda x: f"{x['category']} {x['baseColour']} {x['pattern']} for {x['gender']}",
        axis=1
    ).tolist()

    embeddings = _embedder.encode_text(descriptions)

    # Build search engine
    search_engine = VisualSearchEngine(embedding_dim=embeddings.shape[1])
    search_engine.build_index(
        embeddings=embeddings,
        product_ids=_df['id'].tolist(),
        metadata=_df
    )

    return search_engine, embeddings


def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Smart Visual Commerce Platform</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            AI-Powered E-Commerce Intelligence Platform using Vision-Language Models
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data and models
    with st.spinner("🔄 Loading models and data..."):
        df, loader = load_data()
        embedder, preprocessor = initialize_models()

        if embedder is None:
            st.error("Failed to load models. Please check installation.")
            return

        search_engine, embeddings = build_search_index(df, embedder)
        recommender = RecommendationEngine(df)

    # Sidebar
    st.sidebar.title("📊 Platform Features")
    feature = st.sidebar.radio(
        "Select Feature:",
        [
            "🏠 Overview",
            "🔍 Visual Search",
            "🏷️ Attribute Extraction",
            "🎨 Scene Understanding",
            "💡 Recommendations",
            "⭐ Quality Assessment",
            "📈 Analytics Dashboard"
        ]
    )

    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Statistics")
    st.sidebar.metric("Total Products", len(df))
    st.sidebar.metric("Categories", df['category'].nunique())
    st.sidebar.metric("Price Range", f"${df['price'].min():.0f} - ${df['price'].max():.0f}")

    # Main content
    if feature == "🏠 Overview":
        show_overview(df)

    elif feature == "🔍 Visual Search":
        show_visual_search(search_engine, embedder, df)

    elif feature == "🏷️ Attribute Extraction":
        show_attribute_extraction(df)

    elif feature == "🎨 Scene Understanding":
        show_scene_understanding()

    elif feature == "💡 Recommendations":
        show_recommendations(recommender, embeddings, df)

    elif feature == "⭐ Quality Assessment":
        show_quality_assessment(preprocessor)

    elif feature == "📈 Analytics Dashboard":
        show_analytics(df)


def show_overview(df):
    """Display platform overview."""
    st.header("🏠 Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🔍</h2>
            <h3>Visual Search</h3>
            <p>Find products using images or text</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🏷️</h2>
            <h3>Auto-Tagging</h3>
            <p>Extract product attributes with AI</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>💡</h2>
            <h3>Smart Recommendations</h3>
            <p>Personalized product suggestions</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>⭐</h2>
            <h3>Quality Check</h3>
            <p>Automated image quality assessment</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset preview
    st.subheader("📦 Product Catalog Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Category Distribution")
        fig = px.pie(df, names='category', title='Products by Category')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Price Distribution")
        fig = px.histogram(df, x='price', nbins=20, title='Price Distribution')
        st.plotly_chart(fig, use_container_width=True)


def show_visual_search(search_engine, embedder, df):
    """Display visual search interface."""
    st.header("🔍 Visual Product Search")

    st.markdown("""
    Search for products using natural language queries or apply filters to find exactly what you're looking for.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "🔎 Search Query",
            placeholder="e.g., red dress for women, blue jeans, formal shoes...",
            help="Enter a natural language description of what you're looking for"
        )

    with col2:
        top_k = st.slider("Number of Results", 1, 20, 10)

    # Filters
    with st.expander("🎛️ Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            category_filter = st.selectbox("Category", ["All"] + list(df['category'].unique()))
        with col2:
            color_filter = st.selectbox("Color", ["All"] + list(df['baseColour'].unique()))
        with col3:
            gender_filter = st.selectbox("Gender", ["All"] + list(df['gender'].unique()))

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                # Generate query embedding
                query_embedding = embedder.encode_text(query)

                # Prepare filters
                filters = {}
                if category_filter != "All":
                    filters['category'] = category_filter
                if color_filter != "All":
                    filters['baseColour'] = color_filter
                if gender_filter != "All":
                    filters['gender'] = gender_filter

                # Search
                if filters:
                    results = search_engine.search_with_filters(
                        query_embedding, top_k=top_k, filters=filters
                    )
                else:
                    results = search_engine.search(query_embedding, top_k=top_k)

                # Display results
                st.success(f"Found {len(results)} products matching your query!")

                if len(results) > 0:
                    st.subheader("🎯 Search Results")

                    # Display in grid
                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class="feature-card">
                                <h4>{result['productDisplayName']}</h4>
                                <p><strong>Category:</strong> {result['category']}</p>
                                <p><strong>Color:</strong> {result['baseColour']}</p>
                                <p><strong>Price:</strong> ${result['price']:.2f}</p>
                                <p><strong>Similarity:</strong> {result['similarity']:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No results found. Try adjusting your filters.")
        else:
            st.warning("Please enter a search query.")


def show_attribute_extraction(df):
    """Display attribute extraction interface."""
    st.header("🏷️ Product Attribute Extraction")

    st.markdown("""
    Automatically extract product attributes from images using Vision-Language Models.
    This feature demonstrates how AI can auto-tag products for better categorization.
    """)

    st.info("💡 **Note:** This feature requires VLM API keys (OpenAI or Google Gemini). Set them in your .env file.")

    # Select a product
    product_id = st.selectbox(
        "Select a Product",
        df['id'].tolist(),
        format_func=lambda x: df[df['id'] == x].iloc[0]['productDisplayName']
    )

    if product_id:
        product = df[df['id'] == product_id].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📦 Product Information")
            st.write(f"**Name:** {product['productDisplayName']}")
            st.write(f"**ID:** {product['id']}")
            st.write(f"**Description:** {product['description']}")

        with col2:
            st.subheader("🏷️ Extracted Attributes")

            attributes = {
                "Category": product['category'],
                "Sub-Category": product['subCategory'],
                "Color": product['baseColour'],
                "Pattern": product['pattern'],
                "Gender": product['gender'],
                "Season": product['season'],
                "Usage": product['usage']
            }

            for key, value in attributes.items():
                st.markdown(f"**{key}:** `{value}`")


def show_scene_understanding():
    """Display scene understanding interface."""
    st.header("🎨 Scene Understanding & Context Recommendations")

    st.markdown("""
    Upload a room or scene image to get context-aware product recommendations.
    The AI analyzes your space and suggests products that would fit perfectly.
    """)

    uploaded_file = st.file_uploader(
        "Upload a room image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of your room or space"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Your Space")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🤖 AI Analysis")
            st.info("💡 **Note:** Full scene analysis requires VLM API integration.")

            # Mock analysis for demo
            st.markdown("""
            **Room Type:** Living Room
            **Style:** Modern, Minimalist
            **Color Scheme:** Neutral tones with blue accents

            **Suggested Products:**
            - Modern sofa in grey
            - Blue decorative pillows
            - Minimalist coffee table
            - Wall art with abstract design
            """)


def show_recommendations(recommender, embeddings, df):
    """Display recommendations interface."""
    st.header("💡 Smart Product Recommendations")

    st.markdown("""
    Get personalized product recommendations using our hybrid recommendation engine.
    """)

    # Select a product
    product_id = st.selectbox(
        "Select a Product",
        df['id'].tolist(),
        format_func=lambda x: df[df['id'] == x].iloc[0]['productDisplayName']
    )

    if product_id:
        product = df[df['id'] == product_id].iloc[0]

        st.subheader("📦 Selected Product")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Category", product['category'])
        with col2:
            st.metric("Color", product['baseColour'])
        with col3:
            st.metric("Price", f"${product['price']:.2f}")

        # Recommendation types
        tab1, tab2, tab3 = st.tabs(["🔄 Similar Products", "🎯 Complete the Look", "🌟 Hybrid Recommendations"])

        with tab1:
            recs = recommender.content_based_recommendations(
                product_id, embeddings, df['id'].tolist(), top_n=5
            )
            display_recommendations(recs, "Similar products based on style and features")

        with tab2:
            recs = recommender.complementary_recommendations(product_id, top_n=5)
            display_recommendations(recs, "Complementary products to complete your look")

        with tab3:
            recs = recommender.hybrid_recommendations(
                product_id, embeddings, df['id'].tolist(), top_n=5
            )
            display_recommendations(recs, "Best recommendations from multiple strategies")


def display_recommendations(recs, description):
    """Helper function to display recommendations."""
    st.markdown(f"*{description}*")

    if len(recs) > 0:
        for i, rec in enumerate(recs, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    **{i}. {rec['productDisplayName']}**
                    Category: {rec['category']} | Color: {rec.get('baseColour', 'N/A')}
                    *{rec.get('reason', '')}*
                    """)
                with col2:
                    st.metric("Price", f"${rec['price']:.2f}")
    else:
        st.info("No recommendations available.")


def show_quality_assessment(preprocessor):
    """Display quality assessment interface."""
    st.header("⭐ Image Quality Assessment")

    st.markdown("""
    Upload product images to get automated quality assessment and suggestions.
    """)

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("📊 Quality Metrics")

            with st.spinner("Analyzing..."):
                quality = preprocessor.assess_image_quality(image)

                # Display metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Overall Quality", f"{quality['overall']:.2%}")
                    st.metric("Sharpness", f"{quality['sharpness']:.2%}")
                with col_b:
                    st.metric("Brightness", f"{quality['brightness']:.2%}")
                    st.metric("Contrast", f"{quality['contrast']:.2%}")

                # Quality score visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=quality['overall'] * 100,
                    title={'text': "Overall Quality Score"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 75], 'color': "gray"},
                               {'range': [75, 100], 'color': "lightgreen"}
                           ]}
                ))
                st.plotly_chart(fig, use_container_width=True)


def show_analytics(df):
    """Display analytics dashboard."""
    st.header("📈 Analytics Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():.2f}")
    with col3:
        st.metric("Categories", df['category'].nunique())
    with col4:
        st.metric("Colors", df['baseColour'].nunique())

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "💰 Pricing", "🎯 Categories"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(df['category'].value_counts(), title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(df['baseColour'].value_counts().head(10), title="Top 10 Colors")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.box(df, x='category', y='price', title="Price Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.sunburst(df, path=['category', 'gender'], title="Product Hierarchy")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
