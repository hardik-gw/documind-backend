import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY missing from environment variables.")
        
        clean_key = api_key.strip().strip('"').strip("'")
        genai.configure(api_key=clean_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are DocuMind, an elite, professional document analysis AI assistant.\n"
                "Your task is to answer the user's question using ONLY the provided Source Document Fragments below.\n"
                "CRITICAL RULES:\n"
                "1. Rely only on the clear facts directly mentioned in the context. Do not make assumptions.\n"
                "2. At the end of every sentence or claim you make, cite the source number using brackets like [1] or [2]\n"
                "   based on which Document Fragment the information came from.\n"
                "3. If the context does not contain the answer, say cleanly: "
                "'I cannot find the answer in the uploaded documents.'"
            )
        )

    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        context_text = ""
        for idx, chunk in enumerate(context_chunks):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown Document")
            page = metadata.get("page", "1")
            
            context_text += f"\n--- Source Document Fragment #{idx+1} (File: {source}, Page: {page}) ---\n"
            context_text += f"{chunk.get('text', '')}\n"

        user_message = f"Context Documents:\n{context_text}\n\nUser Question: {question}"

        try:
            print("🧠 Synthesizing answer with Gemini...")
            response = self.model.generate_content(user_message)
            return response.text
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            raise e