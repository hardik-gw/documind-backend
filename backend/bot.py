import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_store import VectorStoreService
from backend.services.llm import LLMService
from backend.services.reranker import RerankerService

# Load environment coordinates
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Instantiate identical architectural elements from our core pipeline
processor = PDFProcessor()
db_service = VectorStoreService()
llm_service = LLMService()
reranker = RerankerService()

DOWNLOAD_DIR = "./telegram_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Shared memory chat log buffer tracking multi-turn sequences per chat session
BOT_CHAT_HISTORY = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Greets the user and initializes an empty sliding memory log window."""
    chat_id = update.effective_chat.id
    BOT_CHAT_HISTORY[chat_id] = []
    
    welcome_text = (
        "🧠 *Welcome to DocuMind AI!* 🧠\n\n"
        "I am an advanced multi-tenant RAG analysis bot optimized with Cross-Encoder reranking.\n\n"
        "📥 *How to use me:*\n"
        "1. Send or forward me any *PDF document*.\n"
        "2. Wait a brief moment for me to process your context arrays.\n"
        "3. Type *any question* to query your files with cited source confidence rankings!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intercepts uploaded PDF attachments, runs layout segmentations, and tags chunks with Telegram IDs."""
    document = update.message.document
    user_id = str(update.message.from_user.id) # 🔒 Key isolation point: Use unique Telegram numeric ID!
    chat_id = update.effective_chat.id
    
    if not document.file_name.endswith('.pdf'):
        await update.message.reply_text("❌ Error: I currently only support reading raw PDF files.")
        return

    status_message = await update.message.reply_text("📥 *Downloading file payload... Please hold.*", parse_mode="Markdown")
    
    try:
        tg_file = await context.bot.get_file(document.file_id)
        local_path = os.path.join(DOWNLOAD_DIR, f"{user_id}_{document.file_name}")
        await tg_file.download_to_drive(local_path)
        
        await status_message.edit_text("⚡ *File grabbed! Initializing text-layout segmentation...*")
        chunks = processor.process_pdf(local_path)
        
        if not chunks:
            await status_message.edit_text("⚠️ Processing failed. No text layers could be safely extracted.")
            return
            
        await status_message.edit_text(f"🧱 *Generated {len(chunks)} structural fragments. Running vector mappings...*")
        
        # Day 9 User Isolation: Commit chunks bound strictly to this sender's ID
        db_service.add_chunks(chunks, user_id=user_id)
        
        # Clear rolling conversation history whenever a fresh document context is established
        BOT_CHAT_HISTORY[chat_id] = []
        
        await status_message.edit_text(
            f"✅ *Ingestion successful!*\n"
            f"📄 Document: `{document.file_name}`\n"
            f"🔒 Isolated under your secure personal session ID.\n\n"
            f"👉 Go ahead and ask me any question about this document!"
        )
        
    except Exception as e:
        print(f"💥 Bot ingestion failure: {e}")
        await status_message.edit_text(f"❌ Critical ingestion layer failure: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles text questions, rewrites multi-turn nuances, scores context relevance, and generates answers."""
    user_question = update.message.text
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id
    
    if chat_id not in BOT_CHAT_HISTORY:
        BOT_CHAT_HISTORY[chat_id] = []
        
    chat_history = BOT_CHAT_HISTORY[chat_id]
    search_query = user_question
    
    thinking_message = await update.message.reply_text("🤔 *Analyzing context vault...*", parse_mode="Markdown")

    try:
        # Day 8: Query Contextualization Layer
        if len(chat_history) > 0:
            history_context = "".join([f"{t['role'].upper()}: {t['text']}\n" for t in chat_history[-4:]])
            contextual_prompt = (
                f"Given the following chat history and a new user question, rewrite the user question "
                f"to be a standalone query for a vector database.\n\n"
                f"Chat History:\n{history_context}\nNew Question: {user_question}\n\nStandalone Query:"
            )
            try:
                rewrite_response = llm_service.client.models.generate_content(
                    model=llm_service.model_name, contents=contextual_prompt
                )
                search_query = rewrite_response.text.strip()
            except Exception as e:
                # Fallback to the raw question if query rewriting hits an error to keep the pipeline alive
                print(f"⚠️ Contextualizer hit a snag, falling back to raw query: {e}")
                search_query = user_question

        # Day 10 Two-Stage Retrieval Phase 1: Fetch top 10 isolated candidates
        candidates = db_service.query_similar_chunks(search_query, user_id=user_id, top_k=10)
        
        if not candidates:
            await thinking_message.edit_text("ℹ️ *I couldn't find any documents bound to your user scope. Please upload a PDF file first!*", parse_mode="Markdown")
            return
            
        # Day 10 Two-Stage Retrieval Phase 2: Local Cross-Encoder Reranking
        reranked_fragments = reranker.rerank(query=search_query, chunks=candidates, top_k=3)
        
        # Day 5: Synthesize final cited text summary block with explicit API error trapping
        try:
            ai_response = llm_service.generate_answer(user_question, reranked_fragments)
        except Exception as api_err:
            print(f"💥 Gemini API Network Error: {api_err}")
            if "503" in str(api_err) or "UNAVAILABLE" in str(api_err):
                await thinking_message.edit_text(
                    "🚦 *Google's API is currently experiencing a massive traffic spike (503 Unavailable).* \n\n"
                    "Your documents are perfectly safe and vectorized. Please wait a minute and send your question again!",
                    parse_mode="Markdown"
                )
            else:
                await thinking_message.edit_text("⚠️ *The AI synthesis layer encountered a temporary connection issue. Please try your question again shortly.*")
            return
        
        # Build clean structural citations footer block for the chat layout
        citation_text = "\n\n📌 *Sources & Relevance Metrics:*"
        for idx, chunk in enumerate(reranked_fragments):
            score = chunk.get("rerank_score", 0.0)
            page = chunk["metadata"]["page"]
            preview = chunk["text"][:60].replace('\n', ' ') + "..."
            citation_text += f"\n• *[{idx+1}]* Page {page} | Relevance Score: `{score:.2f}`\n   _{preview}_"
            
        final_payload = f"{ai_response}{citation_text}"
        
        # Update memory state arrays
        BOT_CHAT_HISTORY[chat_id].append({"role": "user", "text": user_question})
        BOT_CHAT_HISTORY[chat_id].append({"role": "model", "text": ai_response})
        
        await thinking_message.edit_text(final_payload, parse_mode="Markdown")
        
    except Exception as e:
        print(f"💥 Generic Bot interaction error: {e}")
        await thinking_message.edit_text(f"❌ Failure during analysis loop: Simple processing exception occurred.")

def main():
    if not TOKEN:
        print("❌ Critical Error: TELEGRAM_BOT_TOKEN missing from environment configurations.")
        return
        
    print("🤖 Launching Integrated DocuMind Telegram Engine Core...")
    app = Application.builder().token(TOKEN).build()
    
    # Map event routing boundaries
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Bot is live and actively listening for incoming stream data...")
    app.run_polling()

if __name__ == "__main__":
    main()