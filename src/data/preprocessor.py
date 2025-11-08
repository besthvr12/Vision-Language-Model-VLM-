"""
Image preprocessing utilities for the Smart Visual Commerce Platform.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, List
import torch
from torchvision import transforms


class ImagePreprocessor:
    """Handles image preprocessing and augmentation."""

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size

        # Standard preprocessing for CLIP and similar models
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Transform without normalization (for display)
        self.transform_display = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ])

    def preprocess_image(self, image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """Preprocess a PIL Image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if normalize:
            return self.transform(image)
        else:
            return self.transform_display(image)

    def preprocess_from_path(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        """Load and preprocess image from file path."""
        image = Image.open(image_path)
        return self.preprocess_image(image, normalize)

    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess a batch of images."""
        processed = [self.preprocess_image(img) for img in images]
        return torch.stack(processed)

    def assess_image_quality(self, image: Image.Image) -> dict:
        """
        Assess image quality using various metrics.
        Returns a dictionary with quality scores.
        """
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Calculate quality metrics
        quality_scores = {}

        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_scores['sharpness'] = min(laplacian_var / 1000, 1.0)  # Normalize

        # 2. Brightness
        brightness = np.mean(gray) / 255.0
        quality_scores['brightness'] = brightness

        # 3. Contrast
        contrast = gray.std() / 128.0
        quality_scores['contrast'] = min(contrast, 1.0)

        # 4. Resolution score
        width, height = image.size
        resolution_score = min((width * height) / (1920 * 1080), 1.0)
        quality_scores['resolution'] = resolution_score

        # 5. Overall quality score (weighted average)
        overall = (
            quality_scores['sharpness'] * 0.4 +
            quality_scores['contrast'] * 0.3 +
            quality_scores['brightness'] * 0.2 +
            quality_scores['resolution'] * 0.1
        )
        quality_scores['overall'] = overall

        return quality_scores

    def detect_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Detect dominant colors in an image using k-means clustering.
        Returns list of RGB tuples.
        """
        # Resize for faster processing
        img = image.resize((150, 150))
        img_array = np.array(img)

        # Reshape to 2D array of pixels
        pixels = img_array.reshape(-1, 3)

        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)

        return [tuple(color) for color in colors]

    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (128, 128)) -> Image.Image:
        """Create a thumbnail of the image."""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

    def augment_image(self, image: Image.Image) -> List[Image.Image]:
        """
        Create augmented versions of an image for data augmentation.
        Returns list of augmented images.
        """
        augmented = [image]  # Original

        # Horizontal flip
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Slight rotations
        augmented.append(image.rotate(10, fillcolor='white'))
        augmented.append(image.rotate(-10, fillcolor='white'))

        # Brightness adjustments
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Brightness(image)
        augmented.append(enhancer.enhance(1.2))  # Brighter
        augmented.append(enhancer.enhance(0.8))  # Darker

        # Contrast adjustments
        enhancer = ImageEnhance.Contrast(image)
        augmented.append(enhancer.enhance(1.2))

        return augmented
