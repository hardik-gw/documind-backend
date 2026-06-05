import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai

# Load environment variables from your secret .env file
load_dotenv()

class LLMService:
    def __init__(self):
        """
        Initializes the official Google GenAI Client for text generation.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ Error: GEMINI_API_KEY missing from your .env file.")
        
        self.client = genai.Client()
        self.model_name = "gemini-2.5-flash" # High-speed, long-context flagship model

    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Combines the user question and retrieved document context into a professional
        system prompt, then asks Gemini 1.5 Flash to generate a synthesized answer.
        """
        # 1. Format the retrieved database chunks into a clear text block for the LLM
        context_text = ""
        for idx, chunk in enumerate(context_chunks):
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            context_text += f"\n--- Source Document Fragment #{idx+1} (File: {source}, Page: {page}) ---\n"
            context_text += f"{chunk['text']}\n"

        # 2. Build the System Instruction / Prompt Engineering boundary
        system_instruction = (
            "You are DocuMind, an elite, professional document analysis AI assistant.\n"
            "Your task is to answer the user's question using ONLY the provided Source Document Fragments below.\n"
            "CRITICAL RULES:\n"
            "1. Rely only on the clear facts directly mentioned in the context. Do not make assumptions.\n"
            "2. At the end of every sentence or claim you make, cite the source number using brackets like [1] or [2]\n"
            "   based on which Document Fragment the information came from.\n"
            "3. If the context does not contain the answer, say cleanly: 'I cannot find the answer in the uploaded documents.'"
        )

        # 3. Combine it into a structured payload
        user_message = f"Context Documents:\n{context_text}\n\nUser Question: {question}"

        try:
            print("🧠 Synthesizing answer with Gemini 1.5 Flash...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config={"system_instruction": system_instruction}
            )
            return response.text
        except Exception as e:
            print(f"❌ Error communicating with generation layer: {e}")
            raise e

# ==========================================
# DAY 5 INTEGRATED PIPELINE RUNNER
# ==========================================
if __name__ == "__main__":
    from backend.services.vector_store import VectorStoreService
    
    print("🚀 Running Full RAG Pipeline Test (Retrieval + Generation)...")
    
    # 1. Initialize the components
    db_service = VectorStoreService()
    llm_service = LLMService()
    
    # 2. Define the question
    test_question = "Tell me about Hardik's experience with FastAPI and what he built with it."
    print(f"\n1️⃣ User Question: '{test_question}'")
    
    # 3. Step 1 of RAG: Retrieve matching fragments from ChromaDB
    print("2️⃣ Querying ChromaDB for relevant information...")
    matched_fragments = db_service.query_similar_chunks(test_question, top_k=2)
    
    # 4. Step 2 of RAG: Pass fragments + question to Gemini for the finalized response
    print("3️⃣ Sending data to generation layer...")
    final_answer = llm_service.generate_answer(test_question, matched_fragments)
    
    print("\n🎯 FINAL DOCUMIND AI RESPONSE:")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)