"""
Vision-Language Model client for attribute extraction and analysis.
Supports OpenAI GPT-4V and Google Gemini.
"""

import os
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any
from PIL import Image
import json


class VLMClient:
    """Client for interacting with Vision-Language Models."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """
        Initialize VLM client.

        Args:
            provider: 'openai' or 'gemini'
            model: Model name
            api_key: API key (will read from env if not provided)
        """
        self.provider = provider.lower()
        self.model = model

        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=api_key or os.getenv("OPENAI_API_KEY")
                )
            except ImportError:
                print("⚠ OpenAI package not installed. Run: pip install openai")
                self.client = None

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
                self.client = genai.GenerativeModel(model)
            except ImportError:
                print("⚠ Google GenAI package not installed. Run: pip install google-generativeai")
                self.client = None

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def analyze_image(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Analyze image with custom prompt.

        Args:
            image: PIL Image
            prompt: Analysis prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Model response as string
        """
        if self.client is None:
            return "VLM client not initialized. Please install required packages and set API key."

        if self.provider == "openai":
            # Convert image to base64
            base64_image = self._image_to_base64(image)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content

        elif self.provider == "gemini":
            response = self.client.generate_content([prompt, image])
            return response.text

    def extract_attributes(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract product attributes from image.

        Returns:
            Dictionary with extracted attributes
        """
        prompt = """Analyze this product image and extract the following attributes in JSON format:
{
    "category": "main product category",
    "subcategory": "specific subcategory",
    "color": "dominant color(s)",
    "pattern": "pattern type (solid, striped, floral, etc.)",
    "material": "apparent material",
    "style": "style description",
    "gender": "target gender (Men/Women/Unisex)",
    "season": "suitable season",
    "features": ["list", "of", "key", "features"]
}

Return ONLY the JSON, no additional text."""

        try:
            response = self.analyze_image(image, prompt, max_tokens=300, temperature=0.3)

            # Try to parse JSON
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            attributes = json.loads(response)
            return attributes

        except Exception as e:
            print(f"Error extracting attributes: {e}")
            return {
                "category": "Unknown",
                "error": str(e)
            }

    def assess_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Assess product image quality.

        Returns:
            Dictionary with quality assessment
        """
        prompt = """Assess the quality of this product image for e-commerce use.
Rate each aspect from 0-10 and provide brief feedback in JSON format:
{
    "overall_quality": 8,
    "image_clarity": 9,
    "lighting": 7,
    "background": 8,
    "product_visibility": 9,
    "professional_appearance": 8,
    "issues": ["list any issues"],
    "suggestions": ["improvement suggestions"]
}

Return ONLY the JSON."""

        try:
            response = self.analyze_image(image, prompt, max_tokens=300, temperature=0.3)

            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            quality = json.loads(response)
            return quality

        except Exception as e:
            print(f"Error assessing quality: {e}")
            return {
                "overall_quality": 5,
                "error": str(e)
            }

    def analyze_scene(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a room/scene image to understand context.

        Returns:
            Dictionary with scene analysis
        """
        prompt = """Analyze this room/scene image and describe it in JSON format:
{
    "room_type": "living room/bedroom/etc",
    "style": "modern/traditional/minimalist/etc",
    "color_scheme": ["dominant", "colors"],
    "items_present": ["list", "of", "visible", "items"],
    "atmosphere": "description of mood/feel",
    "suggested_products": ["products that would fit this space"]
}

Return ONLY the JSON."""

        try:
            response = self.analyze_image(image, prompt, max_tokens=400, temperature=0.5)

            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            scene = json.loads(response)
            return scene

        except Exception as e:
            print(f"Error analyzing scene: {e}")
            return {
                "room_type": "Unknown",
                "error": str(e)
            }

    def verify_review_image(
        self,
        product_description: str,
        review_image: Image.Image
    ) -> Dict[str, Any]:
        """
        Verify if review image matches product description.

        Returns:
            Dictionary with verification results
        """
        prompt = f"""Given this product description: "{product_description}"

Analyze the review image and determine:
1. Does this image show the described product?
2. What condition is the product in?
3. Are there any issues or discrepancies?

Return JSON:
{{
    "matches_description": true/false,
    "confidence": 0-100,
    "condition": "new/good/fair/poor",
    "issues": ["list any issues"],
    "authenticity_score": 0-100
}}

Return ONLY the JSON."""

        try:
            response = self.analyze_image(review_image, prompt, max_tokens=300, temperature=0.3)

            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            verification = json.loads(response)
            return verification

        except Exception as e:
            print(f"Error verifying review image: {e}")
            return {
                "matches_description": True,
                "confidence": 50,
                "error": str(e)
            }

    def generate_product_description(self, image: Image.Image) -> str:
        """
        Generate a marketing description for the product.

        Returns:
            Product description string
        """
        prompt = """Create an engaging e-commerce product description for this item.
Include:
- Main features and benefits
- Style and aesthetics
- Ideal use cases
- 2-3 concise paragraphs

Be persuasive but accurate."""

        try:
            description = self.analyze_image(image, prompt, max_tokens=300, temperature=0.7)
            return description

        except Exception as e:
            print(f"Error generating description: {e}")
            return "Product description unavailable."
