"""
Data loading utilities for the Smart Visual Commerce Platform.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image
from io import BytesIO


class DatasetLoader:
    """Handles loading and managing product datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_sample_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration purposes.
        In production, this would load from Kaggle or other sources.
        """

        # Sample product data structure
        categories = ['Dress', 'Shirt', 'Jeans', 'Shoes', 'Jacket', 'Accessories']
        colors = ['Red', 'Blue', 'Black', 'White', 'Green', 'Yellow', 'Pink', 'Brown']
        patterns = ['Solid', 'Striped', 'Floral', 'Checked', 'Printed']
        genders = ['Men', 'Women', 'Unisex']

        products = []

        for i in range(num_samples):
            product = {
                'id': f'PROD_{i:04d}',
                'productDisplayName': f'Product {i}',
                'category': categories[i % len(categories)],
                'subCategory': f'Sub{categories[i % len(categories)]}',
                'baseColour': colors[i % len(colors)],
                'season': ['Summer', 'Winter', 'Fall', 'Spring'][i % 4],
                'year': 2023,
                'usage': ['Casual', 'Formal', 'Sports', 'Party'][i % 4],
                'gender': genders[i % len(genders)],
                'pattern': patterns[i % len(patterns)],
                'price': round(20 + (i * 5.7) % 200, 2),
                'image_url': f'https://via.placeholder.com/300x400?text=Product+{i}',
                'description': f'High quality {colors[i % len(colors)].lower()} {categories[i % len(categories)].lower()} for {genders[i % len(genders)].lower()}'
            }
            products.append(product)

        df = pd.DataFrame(products)

        # Save to CSV
        csv_path = self.processed_dir / "products.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Created sample dataset with {num_samples} products: {csv_path}")

        return df

    def load_dataset(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if dataset_path is None:
            dataset_path = self.processed_dir / "products.csv"

        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}. Creating sample dataset...")
            return self.create_sample_dataset()

        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded dataset with {len(df)} products")
        return df

    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
        return None

    def get_product_by_id(self, product_id: str, df: pd.DataFrame) -> Optional[Dict]:
        """Get product details by ID."""
        product = df[df['id'] == product_id]
        if len(product) > 0:
            return product.iloc[0].to_dict()
        return None

    def filter_products(
        self,
        df: pd.DataFrame,
        category: Optional[str] = None,
        color: Optional[str] = None,
        gender: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None
    ) -> pd.DataFrame:
        """Filter products based on various criteria."""
        filtered_df = df.copy()

        if category:
            filtered_df = filtered_df[filtered_df['category'] == category]

        if color:
            filtered_df = filtered_df[filtered_df['baseColour'] == color]

        if gender:
            filtered_df = filtered_df[filtered_df['gender'] == gender]

        if price_min is not None:
            filtered_df = filtered_df[filtered_df['price'] >= price_min]

        if price_max is not None:
            filtered_df = filtered_df[filtered_df['price'] <= price_max]

        return filtered_df

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_products': len(df),
            'categories': df['category'].value_counts().to_dict(),
            'colors': df['baseColour'].value_counts().to_dict(),
            'genders': df['gender'].value_counts().to_dict(),
            'price_stats': {
                'min': df['price'].min(),
                'max': df['price'].max(),
                'mean': df['price'].mean(),
                'median': df['price'].median()
            }
        }
        return stats
