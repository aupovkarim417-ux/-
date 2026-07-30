import asyncio
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import aiohttp
import discord
from discord.ext import commands

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_CONTEXT_MESSAGES = int(os.getenv("GROQ_CONTEXT_MESSAGES", "8"))
GROQ_CANDIDATE_POOL = int(os.getenv("GROQ_CANDIDATE_POOL", "40"))
POOL_FILE = os.getenv("POOL_FILE", "pool.json")
SETTINGS_FILE = os.getenv("SETTINGS_FILE", "settings.json")
MIN_REPLY_INTERVAL = 10
REPLY_EVERY_N = max(MIN_REPLY_INTERVAL, int(os.getenv("REPLY_EVERY_N", str(MIN_REPLY_INTERVAL))))
BACKFILL_HISTORY_LIMIT = int(os.getenv("BACKFILL_HISTORY_LIMIT", "500"))

# Do not hardcode API keys in git. Use one of these environment variables:
# GROQ_API_KEYS=gsk_key1,gsk_key2
# GROQ_API_KEY_1=gsk_key1, GROQ_API_KEY_2=gsk_key2, ...
# GROQ_API_KEY=gsk_single_key


def _load_groq_keys() -> list[str]:
    multi = os.environ.get("GROQ_API_KEYS", "")
    if multi:
        keys = [key.strip() for key in multi.split(",") if key.strip()]
        if keys:
            return keys

    numbered = [
        key.strip()
        for index in range(1, 11)
        if (key := os.environ.get(f"GROQ_API_KEY_{index}")) and key.strip()
    ]
    if numbered:
        return numbered

    single = os.environ.get("GROQ_API_KEY")
    return [single.strip()] if single and single.strip() else []


GROQ_API_KEYS = _load_groq_keys()
_groq_current_idx = 0
_groq_fail_streak = 0
_groq_key_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
GROQ_MAX_FAIL_STREAK = 2


class ContentKind:
    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    VIDEO = "video"
    GIF = "gif"
    EMOJI = "emoji"
    STICKER = "sticker"
    MIXED = "mixed"


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
UNICODE_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u27BF"
    "]+"
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi")
GIF_EXTENSIONS = (".gif",)
VIDEO_HOSTS = ("youtube.com", "youtu.be", "tiktok.com", "twitch.tv", "vimeo.com", "streamable.com")
GIF_HOSTS = ("tenor.com", "giphy.com", "media.tenor.com")


@dataclass
class SmartCandidate:
    index: int
    content: str
    author: str
    kinds: list[str]
    attachments: list[dict[str, str]]
    stickers: list[dict[str, str]]
    created_at: str | None = None

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "id": self.index,
            "author": self.author,
            "kinds": self.kinds,
            "text": self.content[:500],
            "attachments": self.attachments[:4],
            "stickers": self.stickers[:4],
            "created_at": self.created_at,
        }


intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None,
    allowed_mentions=discord.AllowedMentions(everyone=False, users=True, roles=False),
)

all_messages: list[dict[str, Any]] = []
sent_indices: set[int] = set()
message_counters: dict[int, int] = defaultdict(int)
reply_settings: dict[str, Any] = {"mode": "fixed", "n": REPLY_EVERY_N, "min": MIN_REPLY_INTERVAL, "max": 20, "gen_mode": "smart"}
_known_message_keys: set[str] = set()


def normalize_reply_settings() -> None:
    """Keep classic reply interval enabled and never below 10 messages."""
    reply_settings["mode"] = "fixed"
    reply_settings["n"] = max(MIN_REPLY_INTERVAL, int(reply_settings.get("n", MIN_REPLY_INTERVAL)))
    reply_settings["min"] = max(MIN_REPLY_INTERVAL, int(reply_settings.get("min", MIN_REPLY_INTERVAL)))
    reply_settings["max"] = max(reply_settings["min"], int(reply_settings.get("max", reply_settings["min"])))


def message_key(record: dict[str, Any]) -> str:
    return f"{record.get('guild_id')}:{record.get('channel_id')}:{record.get('message_id')}"


def rebuild_known_message_keys() -> None:
    _known_message_keys.clear()
    for record in all_messages:
        key = message_key(record)
        if not key.endswith(':None'):
            _known_message_keys.add(key)


def save_pool() -> None:
    with open(POOL_FILE, "w", encoding="utf-8") as file:
        json.dump(all_messages, file, ensure_ascii=False, indent=2)


def load_pool() -> None:
    global all_messages
    if os.path.exists(POOL_FILE):
        with open(POOL_FILE, "r", encoding="utf-8") as file:
            all_messages = json.load(file)
    rebuild_known_message_keys()


def save_settings() -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(reply_settings, file, ensure_ascii=False, indent=2)


def load_settings() -> None:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            reply_settings.update(json.load(file))
    normalize_reply_settings()


def _groq_active_key() -> str | None:
    if not GROQ_API_KEYS:
        return None
    return GROQ_API_KEYS[_groq_current_idx % len(GROQ_API_KEYS)]


def _groq_rotate(reason: str) -> None:
    global _groq_current_idx, _groq_fail_streak
    if len(GROQ_API_KEYS) <= 1:
        return
    old_idx = _groq_current_idx % len(GROQ_API_KEYS)
    _groq_key_stats[old_idx]["fail"] += 1
    _groq_current_idx += 1
    _groq_fail_streak = 0
    new_idx = _groq_current_idx % len(GROQ_API_KEYS)
    print(f"Groq key #{old_idx + 1} -> #{new_idx + 1}: {reason}")


def _groq_on_success() -> None:
    global _groq_fail_streak
    _groq_fail_streak = 0
    if GROQ_API_KEYS:
        _groq_key_stats[_groq_current_idx % len(GROQ_API_KEYS)]["ok"] += 1


def _groq_on_fail(reason: str) -> None:
    global _groq_fail_streak
    _groq_fail_streak += 1
    if _groq_fail_streak >= GROQ_MAX_FAIL_STREAK:
        _groq_rotate(reason)


def is_news_channel(channel: discord.abc.Messageable) -> bool:
    channel_type = getattr(channel, "type", None)
    if channel_type == discord.ChannelType.news:
        return True
    parent = getattr(channel, "parent", None)
    return getattr(parent, "type", None) == discord.ChannelType.news


def can_collect_message(message: discord.Message) -> bool:
    if message.author.bot or is_news_channel(message.channel):
        return False
    return message.type in {discord.MessageType.default, discord.MessageType.reply}


def add_record_to_pool(record: dict[str, Any]) -> bool:
    key = message_key(record)
    if key in _known_message_keys:
        return False
    all_messages.append(record)
    if not key.endswith(':None'):
        _known_message_keys.add(key)
    return True


async def backfill_guild_history() -> None:
    """Load existing server messages so the bot can choose from more than live messages."""
    added = 0
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if is_news_channel(channel):
                continue
            permissions = channel.permissions_for(guild.me) if guild.me else None
            if permissions and (not permissions.read_message_history or not permissions.view_channel):
                continue
            try:
                async for message in channel.history(limit=BACKFILL_HISTORY_LIMIT):
                    if can_collect_message(message):
                        added += int(add_record_to_pool(message_to_record(message)))
            except (discord.Forbidden, discord.HTTPException):
                continue
    if added:
        save_pool()
    print(f"Backfill complete: added {added} messages; pool size {len(all_messages)}")


def classify_url(url: str) -> str:
    clean_url = url.lower().split("?", 1)[0]
    if clean_url.endswith(GIF_EXTENSIONS) or any(host in clean_url for host in GIF_HOSTS):
        return ContentKind.GIF
    if clean_url.endswith(IMAGE_EXTENSIONS):
        return ContentKind.IMAGE
    if clean_url.endswith(VIDEO_EXTENSIONS) or any(host in clean_url for host in VIDEO_HOSTS):
        return ContentKind.VIDEO
    return ContentKind.LINK


def classify_attachment(attachment: discord.Attachment | dict[str, Any]) -> str:
    content_type = ""
    filename = ""
    url = ""
    if isinstance(attachment, dict):
        content_type = str(attachment.get("content_type") or "").lower()
        filename = str(attachment.get("filename") or "").lower()
        url = str(attachment.get("url") or "").lower()
    else:
        content_type = str(attachment.content_type or "").lower()
        filename = attachment.filename.lower()
        url = attachment.url.lower()

    if "gif" in content_type or filename.endswith(GIF_EXTENSIONS) or url.endswith(GIF_EXTENSIONS):
        return ContentKind.GIF
    if content_type.startswith("image/") or filename.endswith(IMAGE_EXTENSIONS):
        return ContentKind.IMAGE
    if content_type.startswith("video/") or filename.endswith(VIDEO_EXTENSIONS):
        return ContentKind.VIDEO
    return ContentKind.LINK


def detect_content_kinds(content: str, attachments: list[Any] | None = None, stickers: list[Any] | None = None) -> list[str]:
    kinds: set[str] = set()
    if content.strip():
        kinds.add(ContentKind.TEXT)
    if CUSTOM_EMOJI_RE.search(content) or UNICODE_EMOJI_RE.search(content):
        kinds.add(ContentKind.EMOJI)
    for url in URL_RE.findall(content):
        kinds.add(classify_url(url))
    for attachment in attachments or []:
        kinds.add(classify_attachment(attachment))
    if stickers:
        kinds.add(ContentKind.STICKER)
    if len(kinds) > 1:
        kinds.add(ContentKind.MIXED)
    return sorted(kinds) or [ContentKind.TEXT]


def message_to_record(message: discord.Message) -> dict[str, Any]:
    attachments = [
        {
            "url": attachment.url,
            "proxy_url": attachment.proxy_url,
            "filename": attachment.filename,
            "content_type": attachment.content_type or "",
        }
        for attachment in message.attachments
    ]
    stickers = [
        {
            "id": str(sticker.id),
            "name": sticker.name,
            "url": str(sticker.url) if sticker.url else "",
            "format": str(getattr(sticker, "format", "")),
        }
        for sticker in message.stickers
    ]
    return {
        "content": message.content,
        "author": message.author.display_name,
        "author_id": str(message.author.id),
        "message_id": message.id,
        "channel_id": message.channel.id,
        "channel_type": str(getattr(message.channel, "type", "")),
        "guild_id": message.guild.id if message.guild else None,
        "created_at": message.created_at.isoformat(),
        "attachments": attachments,
        "stickers": stickers,
        "kinds": detect_content_kinds(message.content, attachments, stickers),
    }


def make_candidate(index: int, record: dict[str, Any]) -> SmartCandidate:
    attachments = record.get("attachments") or []
    stickers = record.get("stickers") or []
    return SmartCandidate(
        index=index,
        content=record.get("content", ""),
        author=record.get("author", "unknown"),
        kinds=record.get("kinds") or detect_content_kinds(record.get("content", ""), attachments, stickers),
        attachments=attachments,
        stickers=stickers,
        created_at=record.get("created_at"),
    )


def candidate_pool(exclude_content: str | None = None, kinds: list[str] | None = None) -> list[SmartCandidate]:
    if len(sent_indices) >= len(all_messages):
        sent_indices.clear()

    preferred_kinds = set(kinds or [])
    candidates: list[SmartCandidate] = []
    for index, record in enumerate(all_messages):
        if index in sent_indices or record.get("content") == exclude_content:
            continue
        if record.get("channel_type") == str(discord.ChannelType.news):
            continue
        candidate = make_candidate(index, record)
        if preferred_kinds and not preferred_kinds.intersection(candidate.kinds):
            continue
        candidates.append(candidate)

    if not candidates and preferred_kinds:
        return candidate_pool(exclude_content=exclude_content, kinds=None)
    random.shuffle(candidates)
    return candidates[:GROQ_CANDIDATE_POOL]


async def groq_pick_candidate(
    trigger_message: discord.Message,
    candidates: list[SmartCandidate],
    recent_messages: list[discord.Message],
) -> SmartCandidate | None:
    api_key = _groq_active_key()
    if not api_key or not candidates:
        return None

    context = [
        {
            "author": msg.author.display_name,
            "text": msg.content[:300],
            "kinds": detect_content_kinds(msg.content, list(msg.attachments), list(msg.stickers)),
        }
        for msg in reversed(recent_messages[-GROQ_CONTEXT_MESSAGES:])
        if not msg.author.bot
    ]
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты выбираешь лучший готовый ответ для Discord-бота из списка кандидатов. "
                    "Ответ может быть текстом, ссылкой, картинкой, видео, GIF, эмодзи или стикером. "
                    "Не придумывай новый текст. Верни JSON строго вида {\"id\": number, \"reason\": string}. "
                    "Выбирай по смыслу текущего чата, юмору и формату, а не случайно."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "trigger": {
                            "author": trigger_message.author.display_name,
                            "text": trigger_message.content[:500],
                            "kinds": detect_content_kinds(
                                trigger_message.content,
                                list(trigger_message.attachments),
                                list(trigger_message.stickers),
                            ),
                        },
                        "recent_context": context,
                        "candidates": [candidate.as_prompt_dict() for candidate in candidates],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as response:
                if response.status >= 400:
                    _groq_on_fail(f"HTTP {response.status}")
                    return None
                data = await response.json()
        raw = data["choices"][0]["message"]["content"]
        selected_id = int(json.loads(raw).get("id"))
        by_id = {candidate.index: candidate for candidate in candidates}
        selected = by_id.get(selected_id)
        if selected:
            _groq_on_success()
        return selected
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _groq_on_fail(type(exc).__name__)
        return None


def fallback_pick(candidates: list[SmartCandidate]) -> SmartCandidate | None:
    if not candidates:
        return None
    # Fallback is only used when Groq is unavailable; prefer richer content over plain text.
    weights = []
    for candidate in candidates:
        weights.append(4 if ContentKind.MIXED in candidate.kinds else 3 if any(k in candidate.kinds for k in [ContentKind.GIF, ContentKind.IMAGE, ContentKind.VIDEO, ContentKind.STICKER]) else 1)
    return random.choices(candidates, weights=weights, k=1)[0]


async def pick_message_smart(message: discord.Message) -> SmartCandidate | None:
    trigger_kinds = detect_content_kinds(message.content, list(message.attachments), list(message.stickers))
    candidates = candidate_pool(exclude_content=message.content, kinds=trigger_kinds)
    recent = [msg async for msg in message.channel.history(limit=GROQ_CONTEXT_MESSAGES + 1)]
    selected = await groq_pick_candidate(message, candidates, recent)
    return selected or fallback_pick(candidates)


async def send_candidate(channel: discord.abc.Messageable, candidate: SmartCandidate, guild: discord.Guild | None) -> None:
    content = candidate.content
    if guild:
        content = discord.utils.escape_mentions(content)

    sticker_urls = [item.get("url") for item in candidate.stickers if item.get("url")]
    attachment_urls = [item.get("url") for item in candidate.attachments if item.get("url")]
    parts = [content.strip(), *attachment_urls, *sticker_urls]
    text = "\n".join(part for part in parts if part)
    if not text:
        return
    await channel.send(text)
    sent_indices.add(candidate.index)


def next_reply_threshold() -> int:
    normalize_reply_settings()
    return max(MIN_REPLY_INTERVAL, int(reply_settings.get("n", REPLY_EVERY_N)))


channel_thresholds: dict[int, int] = defaultdict(next_reply_threshold)


@bot.event
async def on_ready() -> None:
    load_pool()
    load_settings()
    print(f"Logged in as {bot.user}; loaded {len(all_messages)} saved messages")
    await backfill_guild_history()


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    if is_news_channel(message.channel):
        await bot.process_commands(message)
        return

    if (message.content or message.attachments or message.stickers) and can_collect_message(message):
        if add_record_to_pool(message_to_record(message)):
            save_pool()

    await bot.process_commands(message)

    message_counters[message.channel.id] += 1
    if message_counters[message.channel.id] < channel_thresholds[message.channel.id]:
        return

    message_counters[message.channel.id] = 0
    channel_thresholds[message.channel.id] = next_reply_threshold()
    if reply_settings.get("gen_mode") != "smart":
        return

    candidate = await pick_message_smart(message)
    if candidate:
        await send_candidate(message.channel, candidate, message.guild)


@bot.command(name="режим")
async def set_generation_mode(ctx: commands.Context, mode: str) -> None:
    if mode not in {"smart", "off"}:
        await ctx.reply("Режимы: smart, off")
        return
    reply_settings["gen_mode"] = mode
    save_settings()
    await ctx.reply(f"Ок, режим генерации: {mode}")


@bot.command(name="интервал")
async def set_reply_interval(ctx: commands.Context, value: int | None = None) -> None:
    if value is None:
        await ctx.reply(f"Текущий интервал: {reply_settings['n']} сообщений. Минимум — {MIN_REPLY_INTERVAL}.")
        return
    if value < MIN_REPLY_INTERVAL:
        await ctx.reply(f"Интервал нельзя убрать или поставить меньше {MIN_REPLY_INTERVAL} сообщений.")
        return
    reply_settings["mode"] = "fixed"
    reply_settings["n"] = value
    normalize_reply_settings()
    save_settings()
    channel_thresholds[ctx.channel.id] = next_reply_threshold()
    message_counters[ctx.channel.id] = 0
    await ctx.reply(f"Ок, бот будет отвечать раз в {reply_settings['n']} сообщений.")


@bot.command(name="пул")
async def pool_stats(ctx: commands.Context) -> None:
    counts = defaultdict(int)
    for index, record in enumerate(all_messages):
        for kind in make_candidate(index, record).kinds:
            counts[kind] += 1
    stats = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items())) or "пусто"
    await ctx.reply(f"В пуле {len(all_messages)} сообщений. Типы: {stats}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Set DISCORD_TOKEN environment variable")
    bot.run(DISCORD_TOKEN)
