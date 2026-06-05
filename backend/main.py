import os
import shutil
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_store import VectorStoreService
from backend.services.llm import LLMService
from backend.services.reranker import RerankerService

# 🌎 Read Environment Variables Safely
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BASE_URL = os.getenv("WEBHOOK_URL")

# Instantiate core service layers globally
processor = PDFProcessor()
db_service = VectorStoreService()
llm_service = LLMService()
reranker = RerankerService()

PDF_DIR = "./uploaded_docs"
DOWNLOAD_DIR = "./telegram_downloads"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

BOT_CHAT_HISTORY: Dict[int, List[Dict[str, str]]] = {}
SESSION_STORAGE: Dict[str, List[Dict[str, str]]] = {}

# 🤖 Create the Telegram Application instance
tg_app = Application.builder().token(TOKEN).build() if TOKEN else None

# --- TELEGRAM BOT EVENT HANDLERS ---
async def start_command(update: Update, context):
    chat_id = update.effective_chat.id
    BOT_CHAT_HISTORY[chat_id] = []
    welcome_text = (
        "🧠 *Welcome to DocuMind AI!* 🧠\n\n"
        "I am an advanced multi-tenant RAG bot running live on Render Webhooks.\n\n"
        "📥 *How to use me:*\n"
        "1. Send or forward me any *PDF document*.\n"
        "2. Type *any question* to query your files with cited source confidence rankings!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_document(update: Update, context):
    document = update.message.document
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id
    
    if not document.file_name.endswith('.pdf'):
        await update.message.reply_text("❌ Error: I currently only support reading raw PDF files.")
        return

    status_message = await update.message.reply_text("📥 *Downloading file payload...*", parse_mode="Markdown")
    try:
        tg_file = await context.bot.get_file(document.file_id)
        local_path = os.path.join(DOWNLOAD_DIR, f"{user_id}_{document.file_name}")
        await tg_file.download_to_drive(local_path)
        
        await status_message.edit_text("⚡ *File grabbed! Tokenizing layout chunks...*")
        chunks = processor.process_pdf(local_path)
        
        if not chunks:
            await status_message.edit_text("⚠️ Processing failed. No text layers could be safely extracted.")
            return
            
        await status_message.edit_text("🧱 *Running vector mappings & user isolation tags...*")
        db_service.add_chunks(chunks, user_id=user_id)
        
        BOT_CHAT_HISTORY[chat_id] = []
        await status_message.edit_text(f"✅ *Ingestion successful!*\n📄 Document: `{document.file_name}`\n\n👉 Ask me anything!")
    except Exception as e:
        await status_message.edit_text(f"❌ Ingestion layer failure: {str(e)}")

async def handle_tg_message(update: Update, context):
    user_question = update.message.text
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id
    
    if chat_id not in BOT_CHAT_HISTORY:
        BOT_CHAT_HISTORY[chat_id] = []
        
    chat_history = BOT_CHAT_HISTORY[chat_id]
    search_query = user_question
    thinking_message = await update.message.reply_text("🤔 *Analyzing context vault...*", parse_mode="Markdown")

    try:
        if len(chat_history) > 0:
            history_context = "".join([f"{t['role'].upper()}: {t['text']}\n" for t in chat_history[-4:]])
            contextual_prompt = f"Rewrite to standalone query:\n\n{history_context}\nQuestion: {user_question}"
            try:
                rewrite_response = llm_service.client.models.generate_content(model=llm_service.model_name, contents=contextual_prompt)
                search_query = rewrite_response.text.strip()
            except Exception:
                search_query = user_question

        candidates = db_service.query_similar_chunks(search_query, user_id=user_id, top_k=10)
        if not candidates:
            await thinking_message.edit_text("ℹ️ *No documents found. Please upload a PDF first!*", parse_mode="Markdown")
            return
            
        reranked_fragments = reranker.rerank(query=search_query, chunks=candidates, top_k=3)
        
        try:
            ai_response = llm_service.generate_answer(user_question, reranked_fragments)
        except Exception as api_err:
            if "503" in str(api_err) or "UNAVAILABLE" in str(api_err):
                await thinking_message.edit_text("🚦 *Google's API is currently experiencing a traffic spike. Please retry in a moment!*", parse_mode="Markdown")
            else:
                await thinking_message.edit_text("⚠️ *The AI synthesis layer hit a temporary connection timeout.*")
            return
        
        citation_text = "\n\n📌 *Sources & Relevance Metrics:*"
        for idx, chunk in enumerate(reranked_fragments):
            score = chunk.get("rerank_score", 0.0)
            page = chunk["metadata"]["page"]
            preview = chunk["text"][:60].replace('\n', ' ') + "..."
            citation_text += f"\n• *[{idx+1}]* Page {page} | Relevance: `{score:.2f}`\n   _{preview}_"
            
        final_payload = f"{ai_response}{citation_text}"
        BOT_CHAT_HISTORY[chat_id].append({"role": "user", "text": user_question})
        BOT_CHAT_HISTORY[chat_id].append({"role": "model", "text": ai_response})
        await thinking_message.edit_text(final_payload, parse_mode="Markdown")
    except Exception as e:
        await thinking_message.edit_text("❌ Failure during internal analysis loop.")

# Bind events to our Telegram app background model
if tg_app:
    tg_app.add_handler(CommandHandler("start", start_command))
    tg_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_tg_message))

# --- LIFECYCLE WEBHOOK HOOKS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles production startup/shutdown events to coordinate webhooks securely."""
    if tg_app and BASE_URL and not "your-app-name" in BASE_URL:
        webhook_route = f"{BASE_URL}/telegram-webhook"
        print(f"🌐 Activating Production Telegram Webhook pointing to: {webhook_route}")
        await tg_app.initialize()
        await tg_app.bot.set_webhook(url=webhook_route)
    else:
        print("ℹ️ Running in Local Dev Mode. Skipping webhook registration.")
        if tg_app:
            await tg_app.initialize()
    yield
    if tg_app:
        print("🛑 Dropping connections cleanly...")
        await tg_app.shutdown()

# Initialize FastAPI with the lifecycle manager bound
app = FastAPI(title="DocuMind Production RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = "guest_user"
    session_id: Optional[str] = "default_session"
    top_k: int = 3

@app.get("/")
async def root():
    return {"status": "online", "mode": "Production Webhook Layer Armed"}

@app.post("/telegram-webhook")
async def telegram_webhook(update_dict: dict):
    """The endpoint where Telegram forwards incoming traffic via POST requests."""
    if tg_app:
        update = Update.de_json(update_dict, tg_app.bot)
        await tg_app.process_update(update)
    return {"status": "processed"}

@app.post("/upload")
@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = Form("guest_user")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file format.")