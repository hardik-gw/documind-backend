import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class EmbeddingService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY missing from .env file.")
        
        # Explicitly configure with AI Studio key
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            raise e