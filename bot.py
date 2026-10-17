"""
Stream Telegram Bot — streams LLM responses word-by-word into a single Telegram message.

Inspired by Mira's streaming UX: one placeholder message, then the same message
is edited in real time as the model generates tokens.

Requirements:
    pip install python-telegram-bot[ext] openai python-dotenv
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. 用简体中文回答。",
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Streaming tuning
MIN_EDIT_INTERVAL = 1.0        # Telegram rate-limit: ~1 edit/sec per message
MAX_MESSAGE_LENGTH = 4000     # Telegram hard limit is 4096
TYPING_CARET = " ▌"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("stream-bot")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Per-user conversation history (in-memory; swap for Redis/DB to persist)
conversation: dict[int, list[dict[str, str]]] = {}
HISTORY_LIMIT = 20  # messages per user


def get_history(user_id: int) -> list[dict[str, str]]:
    """Return a bounded history for the user, creating it on first use."""
    if user_id not in conversation:
        conversation[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversation[user_id]


# ---------------------------------------------------------------------------
# Streaming core
# ---------------------------------------------------------------------------
async def stream_to_message(
    bot,
    chat_id: int,
    text_stream: AsyncIterator[str],
) -> None:
    """
    Stream text chunks into a single Telegram message.

    1. Send a placeholder.
    2. As chunks arrive, edit the same message with the growing text + a caret.
    3. Throttle edits to respect Telegram rate limits.
    4. Trim if the text exceeds Telegram's per-message limit.
    5. Final clean edit removes the caret.
    """
    placeholder = await bot.send_message(chat_id=chat_id, text="⏳ ...")
    full_text = ""
    last_edit = 0.0

    try:
        async for chunk in text_stream:
            if not chunk:
                continue
            full_text += chunk
            now = time.monotonic()

            if now - last_edit < MIN_EDIT_INTERVAL:
                continue
            if not full_text.strip():
                continue

            display = full_text + TYPING_CARET
            if len(display) > MAX_MESSAGE_LENGTH:
                display = "... " + display[-MAX_MESSAGE_LENGTH:]

            try:
                await placeholder.edit_text(display)
                last_edit = now
            except Exception as e:  # network blip, "message is not modified", etc.
                log.debug("edit suppressed: %s", e)

        # Final clean edit
        if full_text.strip():
            try:
                final = full_text
                if len(final) > MAX_MESSAGE_LENGTH:
                    final = "... " + final[-MAX_MESSAGE_LENGTH:]
                await placeholder.edit_text(final)
            except Exception as e:
                log.debug("final edit failed: %s", e)
        else:
            await placeholder.edit_text("(empty response)")
    except Exception as e:
        log.exception("Streaming failed")
        try:
            await placeholder.edit_text(f"⚠️ Error while streaming: `{e}`", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass


async def stream_from_openai(
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """Yield text chunks from an OpenAI chat-completion stream."""
    stream = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
        temperature=0.7,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! Send me any message and I'll stream my reply in real time.\n"
        "Tip: each new message continues the conversation; /reset clears the history."
    )


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation.pop(user_id, None)
    await update.message.reply_text("🧹 Conversation history cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_text = update.message.text

    # Show "typing..." status in the chat header
    await update.message.chat.send_action(ChatAction.TYPING)

    history = get_history(user_id)
    history.append({"role": "user", "content": user_text})
    # Trim to keep memory bounded
    if len(history) > HISTORY_LIMIT:
        history[:] = [history[0]] + history[-(HISTORY_LIMIT - 1):]

    try:
        await stream_to_message(
            context.bot,
            update.effective_chat.id,
            stream_from_openai(history),
        )
        # Note: we don't add the assistant reply to history here because the
        # final text is in the edited message, not in this handler. For full
        # memory continuity, capture the streamed text and append it below.
    except Exception as e:
        log.exception("Handler failed")
        await update.message.reply_text(f"⚠️ Something went wrong: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot is starting… (model=%s)", OPENAI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
