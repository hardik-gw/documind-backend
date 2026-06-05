import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_store import VectorStoreService
from backend.services.llm import LLMService
from backend.services.reranker import RerankerService

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

processor = PDFProcessor()
db_service = VectorStoreService()
llm_service = LLMService()
reranker = RerankerService()

DOWNLOAD_DIR = "./telegram_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

BOT_CHAT_HISTORY = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    document = update.message.document
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id
    
    if not document.file_name.endswith('.pdf'):
        await update.message.reply_text("❌ Error: I currently only support reading raw PDF files.")
        return

    status_message = await update.message.reply_text("📥 *Downloading file payload... Please hold.*", parse_mode="Markdown")
    
    try:
        tg_file = await context.bot.get_file(document.file_id)
        local_path = os.path.join(DOWNLOAD_DIR, f"{user_id}_{document.file_name}")
        await tg_file.download_to_drive(local_path)
        
        await status_message.edit_text("⚡ *File grabbed! Initializing text-layout segmentation...*", parse_mode="Markdown")
        chunks = processor.process_pdf(local_path)
        
        if not chunks:
            await status_message.edit_text("⚠️ Processing failed. No text layers could be safely extracted.")
            return
            
        await status_message.edit_text(
            f"🧱 *Generated {len(chunks)} structural fragments. Running vector mappings...*",
            parse_mode="Markdown"
        )
        db_service.add_chunks(chunks, user_id=user_id)
        BOT_CHAT_HISTORY[chat_id] = []
        
        await status_message.edit_text(
            f"✅ *Ingestion successful!*\n"
            f"📄 Document: `{document.file_name}`\n"
            f"🔒 Isolated under your secure personal session ID.\n\n"
            f"👉 Go ahead and ask me any question about this document!",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        print(f"💥 Bot ingestion failure: {e}")
        await status_message.edit_text(f"❌ Critical ingestion layer failure: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id
    
    if chat_id not in BOT_CHAT_HISTORY:
        BOT_CHAT_HISTORY[chat_id] = []
        
    chat_history = BOT_CHAT_HISTORY[chat_id]
    search_query = user_question
    
    thinking_message = await update.message.reply_text("🤔 *Analyzing context vault...*", parse_mode="Markdown")

    try:
        # --- FIX: use llm_service.model directly ---
        if len(chat_history) > 0:
            history_context = "".join([f"{t['role'].upper()}: {t['text']}\n" for t in chat_history[-4:]])
            contextual_prompt = (
                f"Given the following chat history and a new user question, rewrite the user question "
                f"to be a standalone query for a vector database.\n\n"
                f"Chat History:\n{history_context}\nNew Question: {user_question}\n\nStandalone Query:"
            )
            try:
                rewrite_response = llm_service.model.generate_content(contextual_prompt)
                search_query = rewrite_response.text.strip()
            except Exception as e:
                print(f"⚠️ Contextualizer fallback to raw query: {e}")
                search_query = user_question

        candidates = db_service.query_similar_chunks(search_query, user_id=user_id, top_k=10)
        
        if not candidates:
            await thinking_message.edit_text(
                "ℹ️ *I couldn't find any documents bound to your user scope. Please upload a PDF file first!*",
                parse_mode="Markdown"
            )
            return
            
        reranked_fragments = reranker.rerank(query=search_query, chunks=candidates, top_k=3)
        
        try:
            ai_response = llm_service.generate_answer(user_question, reranked_fragments)
        except Exception as api_err:
            print(f"💥 Gemini API Error: {api_err}")
            if "503" in str(api_err) or "UNAVAILABLE" in str(api_err):
                await thinking_message.edit_text(
                    "🚦 *Google's API is currently experiencing a traffic spike.*\n\n"
                    "Your documents are safe. Please retry in a moment!",
                    parse_mode="Markdown"
                )
            else:
                await thinking_message.edit_text(
                    "⚠️ *The AI synthesis layer encountered a temporary issue. Please try again shortly.*"
                )
            return
        
        citation_text = "\n\n📌 *Sources & Relevance Metrics:*"
        for idx, chunk in enumerate(reranked_fragments):
            score = chunk.get("rerank_score", 0.0)
            page = chunk.get("metadata", {}).get("page", "Unknown")
            preview = chunk.get("text", "")[:60].replace('\n', ' ') + "..."
            citation_text += f"\n• *[{idx+1}]* Page {page} | Relevance Score: `{score:.2f}`\n   _{preview}_"
            
        final_payload = f"{ai_response}{citation_text}"
        
        BOT_CHAT_HISTORY[chat_id].append({"role": "user", "text": user_question})
        BOT_CHAT_HISTORY[chat_id].append({"role": "model", "text": ai_response})
        
        await thinking_message.edit_text(final_payload, parse_mode="Markdown")
        
    except Exception as e:
        print(f"💥 Generic bot error: {e}")
        await thinking_message.edit_text("❌ Failure during analysis loop. Please try again.")

def main():
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN missing from environment.")
        return
        
    print("🤖 Launching DocuMind Telegram Bot...")
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Bot live and polling...")
    app.run_polling()

if __name__ == "__main__":
    main()