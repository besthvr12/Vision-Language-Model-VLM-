"""
CLIP-based embedding generation for visual and text features.
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Union, Optional
import pickle
from pathlib import Path


class CLIPEmbedder:
    """Generate embeddings using CLIP model for images and text."""

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize CLIP embedder.

        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', etc.)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model: {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"✓ CLIP model loaded successfully")

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Encode a single image to embedding vector.

        Args:
            image: PIL Image or preprocessed tensor

        Returns:
            Normalized embedding vector as numpy array
        """
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        # Get image features
        image_features = self.model.encode_image(image)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    @torch.no_grad()
    def encode_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple images in batches.

        Args:
            images: List of PIL Images
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (N x embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Preprocess batch
            batch_tensor = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)

            # Encode
            features = self.model.encode_image(batch_tensor)

            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embedding vector(s).

        Args:
            text: Single string or list of strings

        Returns:
            Normalized embedding vector(s) as numpy array
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        text_tokens = clip.tokenize(text).to(self.device)

        # Encode
        text_features = self.model.encode_text(text_tokens)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        result = text_features.cpu().numpy()

        return result[0] if len(text) == 1 else result

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0 to 1)
        """
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def compute_similarities_batch(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple embeddings.

        Args:
            query_embedding: Single query embedding
            embeddings: Array of embeddings (N x embedding_dim)

        Returns:
            Array of similarity scores
        """
        # Batch cosine similarity
        similarities = np.dot(embeddings, query_embedding)
        return similarities

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Saved embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ Loaded embeddings from {filepath}")
        return embeddings

    def create_multimodal_query(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        text_weight: float = 0.5
    ) -> np.ndarray:
        """
        Create a combined query from text and/or image.

        Args:
            text: Text query
            image: Image query
            text_weight: Weight for text (1 - text_weight will be image weight)

        Returns:
            Combined embedding vector
        """
        if text is None and image is None:
            raise ValueError("At least one of text or image must be provided")

        if text is not None and image is not None:
            # Combine both modalities
            text_emb = self.encode_text(text)
            image_emb = self.encode_image(image)

            # Weighted combination
            combined = text_weight * text_emb + (1 - text_weight) * image_emb

            # Re-normalize
            combined = combined / np.linalg.norm(combined)

            return combined

        elif text is not None:
            return self.encode_text(text)
        else:
            return self.encode_image(image)
