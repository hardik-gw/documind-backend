import os
from typing import List
from dotenv import load_dotenv
from google import genai

# Load environment variables from your secret .env file
load_dotenv()

class EmbeddingService:
    def __init__(self):
        """
        Initializes the official Google GenAI Client.
        It automatically discovers the GEMINI_API_KEY inside the .env environment.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Error: GEMINI_API_KEY missing from your .env file.")
        
        # Initialize client using standard structural syntax
        self.client = genai.Client()
        self.model_name = "gemini-embedding-2" # Standard 768-dimension vector model

    def get_embedding(self, text: str) -> List[float]:
        """
        Transforms a single text chunk into an array of floats.
        """
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            # Pull values directly out of the response matrix
            return response.embeddings[0].values
        except Exception as e:
            print(f"❌ Error compiling individual vector coordinate: {e}")
            raise e

# ==========================================
# DAY 3 SIMULATION RUNNER
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing Gemini Embedding Service...")
    try:
        embedder = EmbeddingService()
        
        test_phrase = "Owned the OCR receipt scanning module end-to-end"
        print(f"\n🔄 Sending chunk to Gemini: '{test_phrase}'")
        
        vector = embedder.get_embedding(test_phrase)
        
        print("✅ Vector calculation completely successful!")
        print(f"📐 Total Numbers in Array (Dimensions): {len(vector)}")
        print(f"🔢 First 5 numeric coordinates: {vector[:5]}")
        
    except Exception as e:
        print(f"💥 Failed: {e}")