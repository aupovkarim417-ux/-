import asyncio
import datetime
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
GROQ_KEYS_FILE = os.getenv("GROQ_KEYS_FILE", "groq_keys.json")
POOL_FILE = os.getenv("POOL_FILE", "pool.json")
SETTINGS_FILE = os.getenv("SETTINGS_FILE", "settings.json")
MIN_REPLY_INTERVAL = 10
REPLY_EVERY_N = MIN_REPLY_INTERVAL
BACKFILL_HISTORY_LIMIT = int(os.getenv("BACKFILL_HISTORY_LIMIT", "500"))

# Do not hardcode API keys in git. Use one of these environment variables:
# GROQ_API_KEYS=gsk_key1,gsk_key2
# GROQ_API_KEY_1=gsk_key1, GROQ_API_KEY_2=gsk_key2, ...
# GROQ_API_KEY=gsk_single_key
# Or put keys in an untracked GROQ_KEYS_FILE (default: groq_keys.json) as JSON list or newline text.


def _load_keys_from_file(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        raw = file.read().strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(parsed, list):
        return [str(key).strip() for key in parsed if str(key).strip()]
    if isinstance(parsed, dict):
        keys = parsed.get("keys") or parsed.get("GROQ_API_KEYS") or []
        if isinstance(keys, str):
            return [key.strip() for key in keys.split(",") if key.strip()]
        if isinstance(keys, list):
            return [str(key).strip() for key in keys if str(key).strip()]
    return []


def _load_groq_keys() -> list[str]:
    file_keys = _load_keys_from_file(GROQ_KEYS_FILE)
    if file_keys:
        return file_keys

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
    """Keep classic reply interval locked to exactly 10 messages."""
    reply_settings["mode"] = "fixed"
    reply_settings["n"] = MIN_REPLY_INTERVAL
    reply_settings["min"] = MIN_REPLY_INTERVAL
    reply_settings["max"] = MIN_REPLY_INTERVAL


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


async def send_random_extra_command(channel: discord.abc.Messageable, user_name: str) -> None:
    """Send the rendered result of a random extra command for automatic 10-message replies."""
    command_name, _description, template = random.choice(EXTRA_COMMANDS)
    latency_ms = round(bot.latency * 1000)
    response = render_extra_response(template, user_name, "", latency_ms)
    await channel.send(discord.utils.escape_mentions(f"🎲 !{command_name}\n{response}"))


def next_reply_threshold() -> int:
    normalize_reply_settings()
    return MIN_REPLY_INTERVAL


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
    await send_random_extra_command(message.channel, message.author.display_name)


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
    normalize_reply_settings()
    save_settings()
    channel_thresholds[ctx.channel.id] = next_reply_threshold()
    message_counters[ctx.channel.id] = 0
    if value is not None and value != MIN_REPLY_INTERVAL:
        await ctx.reply(f"Интервал закреплён: только {MIN_REPLY_INTERVAL} сообщений. Убрать или изменить нельзя.")
        return
    await ctx.reply(f"Интервал закреплён: бот отвечает каждые {MIN_REPLY_INTERVAL} сообщений.")


@bot.command(name="пул")
async def pool_stats(ctx: commands.Context) -> None:
    counts = defaultdict(int)
    for index, record in enumerate(all_messages):
        for kind in make_candidate(index, record).kinds:
            counts[kind] += 1
    stats = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items())) or "пусто"
    await ctx.reply(f"В пуле {len(all_messages)} сообщений. Типы: {stats}")


EXTRA_COMMANDS: list[tuple[str, str, str]] = [
    ("шар", "магический шар", "🎱 {user}, ответ: {choice:да|нет|возможно|точно|лучше позже|без шансов}"),
    ("монетка", "орёл или решка", "🪙 {user}: {choice:орёл|решка|ребро}"),
    ("кубик", "бросок кубика", "🎲 {user} выбросил {rand:1:6}."),
    ("д20", "бросок d20", "🎲 d20 для {user}: {rand:1:20}."),
    ("процент", "случайный процент", "📊 {user}: {rand:0:100}%"),
    ("рейтинг", "рандомный рейтинг", "⭐ Рейтинг {user}: {rand:1:10}/10"),
    ("удача", "проверка удачи", "🍀 Удача {user} сегодня: {rand:0:100}%"),
    ("вайб", "вайб дня", "🌈 Вайб {user}: {choice:спокойный|хаос|мемный|легендарный|сонный|боевой}"),
    ("настроение", "настроение", "🙂 Настроение {user}: {choice:имба|норм|сомнительно|заряжено|мутно|легендарно}"),
    ("энергия", "уровень энергии", "⚡ Энергия {user}: {rand:0:100}%"),
    ("сон", "оценка сна", "😴 Сонливость {user}: {rand:0:100}%"),
    ("чай", "чайный совет", "🍵 {user}, чай дня: {choice:чёрный|зелёный|мятный|с лимоном|без сахара|с печенькой}"),
    ("кофе", "кофейный совет", "☕ {user}, кофе дня: {choice:латте|эспрессо|капучино|американо|раф|без кофеина}"),
    ("пицца", "пицца дня", "🍕 Пицца для {user}: {choice:пепперони|сырная|гавайская|маргарита|грибная|четыре сыра}"),
    ("мем", "мемный статус", "🗿 Мемный статус {user}: {choice:база|кринж|имба|легенда|архив|ультрамем}"),
    ("обнять", "виртуально обнять", "🤗 {user} обнял(а) чат."),
    ("погладить", "погладить чат", "🫳 {user} погладил(а) чат по голове."),
    ("пять", "дать пять", "✋ {user} дал(а) пять!"),
    ("буп", "boop", "👉 boop, {user}!"),
    ("танец", "танец", "💃 {user} устроил(а) танец: {choice:шафл|робот|казачок|диско|хаос|тихий степ}"),
    ("песня", "песня дня", "🎵 Песня дня для {user}: {choice:что-то бодрое|что-то грустное|фонк|рок|лоуфай|поп}"),
    ("фильм", "жанр фильма", "🎬 {user}, жанр на вечер: {choice:комедия|ужасы|фантастика|боевик|драма|аниме}"),
    ("игра", "игра дня", "🎮 {user}, игра дня: {choice:выживание|шутер|стратегия|гонки|песочница|кооп}"),
    ("квест", "мини-квест", "🧭 Квест для {user}: {choice:найти мем|сказать спасибо|выпить воды|сделать скрин|позвать друга|отдохнуть}"),
    ("миссия", "миссия дня", "📌 Миссия {user}: {choice:не сгореть|победить лень|накинуть мем|помочь кому-то|сохранить вайб|дожить до вечера}"),
    ("прогноз", "прогноз", "🔮 Прогноз для {user}: {choice:будет движ|будет тихо|ожидается мем|возможен спор|чат оживёт|нужен чай}"),
    ("совет", "случайный совет", "💡 Совет {user}: {choice:не спеши|проверь дважды|скинь мем|сделай паузу|пей воду|держи баланс}"),
    ("антисовет", "плохой совет", "⚠️ Антисовет {user}: {choice:спорь до утра|не читай правила|открой 100 вкладок|пингуй всех|забудь сохранить|сиди без воды}"),
    ("вода", "напоминание про воду", "💧 {user}, выпей воды."),
    ("перерыв", "напоминание про перерыв", "🧘 {user}, пора сделать маленький перерыв."),
    ("фокус", "фокус режим", "🎯 Фокус {user}: {rand:0:100}%"),
    ("лень", "уровень лени", "🛋️ Лень {user}: {rand:0:100}%"),
    ("имба", "имбовость", "🏆 Имбовость {user}: {rand:0:100}%"),
    ("кринж", "кринжометр", "📉 Кринжометр {user}: {rand:0:100}%"),
    ("база", "базированность", "🧱 Базированность {user}: {rand:0:100}%"),
    ("шанс", "шанс события", "🎯 Шанс: {rand:0:100}%"),
    ("реши", "выбор да/нет", "🤔 Решение: {choice:да|нет|позже|спроси ещё раз|точно да|точно нет}"),
    ("выбери", "выбор варианта", "👉 Я выбираю: {arg_or_choice:первый|второй|третий|рандомный}"),
    ("число", "рандомное число", "🔢 Число для {user}: {rand:1:1000}"),
    ("пароль", "смешной пароль", "🔐 Пароль дня: {choice:banan|kotik|memlord|чай123|discord777|groqboom}{rand:10:99}"),
    ("ник", "ник дня", "🏷️ Ник для {user}: {choice:Мемный Архив|Чайный Маг|Кот Серверный|Гига Вайб|Сонный Босс|Лорд GIF}"),
    ("титул", "титул дня", "👑 Титул {user}: {choice:хранитель мемов|герой чата|маг ссылок|повелитель GIF|рыцарь эмодзи|архивариус}"),
    ("роль", "роль в пати", "🧩 Роль {user}: {choice:танк|хил|дд|саппорт|мемолог|наблюдатель}"),
    ("класс", "класс персонажа", "⚔️ Класс {user}: {choice:воин|маг|вор|бард|инженер|хаосит}"),
    ("лут", "случайный лут", "🎁 {user} получает: {choice:палку|легендарный мем|чай|камень|GIF|секретную ссылку}"),
    ("сундук", "открыть сундук", "🧰 В сундуке: {choice:ничего|монетка|картинка|стикер|редкий вайб|эпичный мем}"),
    ("босс", "мини-босс", "🐲 Босс дня: {choice:Лень|Кринж|Дедлайн|Пинг|Спам|Сон}"),
    ("урон", "урон", "💥 {user} наносит {rand:1:999} урона."),
    ("хил", "лечение", "💚 {user} восстанавливает {rand:1:100} HP."),
    ("щит", "щит", "🛡️ Щит {user}: {rand:1:100}%"),
    ("магия", "магический эффект", "✨ Магия {user}: {choice:искры|дым|конфетти|тишина|хаос|телепорт}"),
    ("погода", "погода в чате", "🌦️ Погода в чате: {choice:мемный дождь|солнечный вайб|туман|шторм сообщений|снег эмодзи|ясно}"),
    ("температура", "температура чата", "🌡️ Температура чата: {rand:-10:40}°C"),
    ("ветер", "ветер", "🌬️ Ветер мемов: {rand:0:30} м/с"),
    ("час", "часовой вайб", "🕒 Сейчас час: {choice:мемов|тишины|чаепития|GIF|ссылок|стикеров}"),
    ("дата", "текущая дата UTC", "📅 Сегодня UTC: {date}."),
    ("время", "текущее время UTC", "⏰ Сейчас UTC: {time}."),
    ("таймер", "мини-таймер", "⏳ {user}, таймер настроения: {rand:1:60} секунд в воображении."),
    ("скорость", "скорость", "🚀 Скорость {user}: {rand:1:300} км/ч"),
    ("пинг", "пинг бота", "🏓 Pong! Задержка: {latency} ms"),
    ("эхо", "повторить текст", "📣 {arg_or:скажи что-нибудь после команды}"),
    ("капс", "текст капсом", "🔊 {arg_upper_or:НЕЧЕГО КРИЧАТЬ}"),
    ("тихо", "текст тихо", "🤫 {arg_lower_or:тихо...}"),
    ("переверни", "перевернуть текст", "🔁 {arg_reverse_or:текст не найден}"),
    ("длина", "длина текста", "📏 Длина текста: {arg_len} символов."),
    ("слова", "количество слов", "🧮 Слов: {arg_words}."),
    ("буквы", "количество букв", "🔤 Букв: {arg_letters}."),
    ("эмодзи", "случайный эмодзи", "{choice:😀|😂|😎|🤔|🔥|💀|✨|🍀|🎲|🫡}"),
    ("гиф", "GIF-подсказка", "🖼️ GIF-вариант: {choice:https://tenor.com/search/cat-gifs|https://tenor.com/search/meme-gifs|https://tenor.com/search/dance-gifs}"),
    ("картинка", "картинка-подсказка", "🖼️ Картинка дня: {choice:https://picsum.photos/400|https://picsum.photos/500|https://picsum.photos/600}"),
    ("видео", "видео-подсказка", "🎥 Видео-вайб: {choice:https://www.youtube.com/|https://www.tiktok.com/|https://vimeo.com/}"),
    ("ссылка", "ссылка-подсказка", "🔗 Ссылка дня: {choice:https://discord.com|https://groq.com|https://python.org}"),
    ("стикер", "стикер-подсказка", "🏷️ Стикерный вайб: {choice:кот|мем|шок|смех|сон|огонь}"),
    ("цвет", "цвет дня", "🎨 Цвет {user}: {choice:красный|синий|зелёный|фиолетовый|чёрный|золотой}"),
    ("число2", "число 1-2", "🔢 {rand:1:2}"),
    ("число3", "число 1-3", "🔢 {rand:1:3}"),
    ("число4", "число 1-4", "🔢 {rand:1:4}"),
    ("число5", "число 1-5", "🔢 {rand:1:5}"),
    ("число10", "число 1-10", "🔢 {rand:1:10}"),
    ("число100", "число 1-100", "🔢 {rand:1:100}"),
    ("дуэль", "дуэль", "⚔️ {user} вызывает чат на дуэль. Победитель: {choice:чат|бот|никто|рандом|мем}"),
    ("арена", "арена", "🏟️ Арена выбрала: {choice:бой|мир|танцы|хаос|тишину}"),
    ("комбо", "комбо", "💫 Комбо {user}: x{rand:1:50}"),
    ("серия", "серия", "🔥 Серия {user}: {rand:1:100}"),
    ("ранг", "ранг", "🏅 Ранг {user}: {choice:бронза|серебро|золото|платина|алмаз|легенда}"),
    ("ачивка", "ачивка", "🏆 {user} получил(а) ачивку: {choice:Первый мем|Сила чая|GIF-мастер|Без пинга|Архиватор|Ночной страж}"),
    ("штраф", "штраф", "🚨 Штраф {user}: {rand:0:999} мем-коинов."),
    ("награда", "награда", "🎖️ Награда {user}: {rand:1:999} мем-коинов."),
    ("банк", "банк", "🏦 Баланс {user} в воображаемом банке: {rand:0:10000}."),
    ("рынок", "рынок", "📈 Рынок мемов: {choice:растёт|падает|флэт|хаос|ракета|сон}"),
    ("курс", "курс", "💱 1 мем = {rand:1:999} вайбов."),
    ("кот", "кот", "🐱 Кот говорит: {choice:мяу|дай еды|скинь GIF|спать|уважаю|мур}"),
    ("пёс", "пёс", "🐶 Пёс говорит: {choice:гав|гулять|мем?|ура|дай косточку|вах}"),
    ("утка", "утка", "🦆 Утка: {choice:кря|важно кря|мем кря|кря?}"),
    ("лягушка", "лягушка", "🐸 Лягушка оценивает вайб на {rand:1:10}/10."),
    ("робот", "робот", "🤖 Робот считает: {choice:бип|буп|мем принят|ошибка вайба|чай нужен}"),
    ("пират", "пират", "🏴‍☠️ Пиратский ответ: {choice:йо-хо-хо|где ром|мем на борт|сокровище найдено}"),
    ("космос", "космос", "🌌 Космос отправил {user}: {choice:звезду|комету|чёрную дыру|спутник|сигнал|тишину}"),
    ("ракета", "ракета", "🚀 Ракета {user} взлетела на {rand:1:1000} км."),
    ("финал", "финальный вердикт", "✅ Финальный вердикт для {user}: {choice:можно|нельзя|надо подумать|имба|лучше мем}"),
    ("фан101", "авто-функция 101", "🎉 Функция #101 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан102", "авто-функция 102", "🎉 Функция #102 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан103", "авто-функция 103", "🎉 Функция #103 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан104", "авто-функция 104", "🎉 Функция #104 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан105", "авто-функция 105", "🎉 Функция #105 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан106", "авто-функция 106", "🎉 Функция #106 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан107", "авто-функция 107", "🎉 Функция #107 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан108", "авто-функция 108", "🎉 Функция #108 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан109", "авто-функция 109", "🎉 Функция #109 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан110", "авто-функция 110", "🎉 Функция #110 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан111", "авто-функция 111", "🎉 Функция #111 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан112", "авто-функция 112", "🎉 Функция #112 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан113", "авто-функция 113", "🎉 Функция #113 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан114", "авто-функция 114", "🎉 Функция #114 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан115", "авто-функция 115", "🎉 Функция #115 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан116", "авто-функция 116", "🎉 Функция #116 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан117", "авто-функция 117", "🎉 Функция #117 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан118", "авто-функция 118", "🎉 Функция #118 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан119", "авто-функция 119", "🎉 Функция #119 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан120", "авто-функция 120", "🎉 Функция #120 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан121", "авто-функция 121", "🎉 Функция #121 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан122", "авто-функция 122", "🎉 Функция #122 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан123", "авто-функция 123", "🎉 Функция #123 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан124", "авто-функция 124", "🎉 Функция #124 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан125", "авто-функция 125", "🎉 Функция #125 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан126", "авто-функция 126", "🎉 Функция #126 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан127", "авто-функция 127", "🎉 Функция #127 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан128", "авто-функция 128", "🎉 Функция #128 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан129", "авто-функция 129", "🎉 Функция #129 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан130", "авто-функция 130", "🎉 Функция #130 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан131", "авто-функция 131", "🎉 Функция #131 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан132", "авто-функция 132", "🎉 Функция #132 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан133", "авто-функция 133", "🎉 Функция #133 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан134", "авто-функция 134", "🎉 Функция #134 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан135", "авто-функция 135", "🎉 Функция #135 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан136", "авто-функция 136", "🎉 Функция #136 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан137", "авто-функция 137", "🎉 Функция #137 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан138", "авто-функция 138", "🎉 Функция #138 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан139", "авто-функция 139", "🎉 Функция #139 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан140", "авто-функция 140", "🎉 Функция #140 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан141", "авто-функция 141", "🎉 Функция #141 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан142", "авто-функция 142", "🎉 Функция #142 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан143", "авто-функция 143", "🎉 Функция #143 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан144", "авто-функция 144", "🎉 Функция #144 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан145", "авто-функция 145", "🎉 Функция #145 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан146", "авто-функция 146", "🎉 Функция #146 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан147", "авто-функция 147", "🎉 Функция #147 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан148", "авто-функция 148", "🎉 Функция #148 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан149", "авто-функция 149", "🎉 Функция #149 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан150", "авто-функция 150", "🎉 Функция #150 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан151", "авто-функция 151", "🎉 Функция #151 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан152", "авто-функция 152", "🎉 Функция #152 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан153", "авто-функция 153", "🎉 Функция #153 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан154", "авто-функция 154", "🎉 Функция #154 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан155", "авто-функция 155", "🎉 Функция #155 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан156", "авто-функция 156", "🎉 Функция #156 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан157", "авто-функция 157", "🎉 Функция #157 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан158", "авто-функция 158", "🎉 Функция #158 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан159", "авто-функция 159", "🎉 Функция #159 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан160", "авто-функция 160", "🎉 Функция #160 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан161", "авто-функция 161", "🎉 Функция #161 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан162", "авто-функция 162", "🎉 Функция #162 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан163", "авто-функция 163", "🎉 Функция #163 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан164", "авто-функция 164", "🎉 Функция #164 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан165", "авто-функция 165", "🎉 Функция #165 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан166", "авто-функция 166", "🎉 Функция #166 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан167", "авто-функция 167", "🎉 Функция #167 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан168", "авто-функция 168", "🎉 Функция #168 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан169", "авто-функция 169", "🎉 Функция #169 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан170", "авто-функция 170", "🎉 Функция #170 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан171", "авто-функция 171", "🎉 Функция #171 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан172", "авто-функция 172", "🎉 Функция #172 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан173", "авто-функция 173", "🎉 Функция #173 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан174", "авто-функция 174", "🎉 Функция #174 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан175", "авто-функция 175", "🎉 Функция #175 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан176", "авто-функция 176", "🎉 Функция #176 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан177", "авто-функция 177", "🎉 Функция #177 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан178", "авто-функция 178", "🎉 Функция #178 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан179", "авто-функция 179", "🎉 Функция #179 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан180", "авто-функция 180", "🎉 Функция #180 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан181", "авто-функция 181", "🎉 Функция #181 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан182", "авто-функция 182", "🎉 Функция #182 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан183", "авто-функция 183", "🎉 Функция #183 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан184", "авто-функция 184", "🎉 Функция #184 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан185", "авто-функция 185", "🎉 Функция #185 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан186", "авто-функция 186", "🎉 Функция #186 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан187", "авто-функция 187", "🎉 Функция #187 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан188", "авто-функция 188", "🎉 Функция #188 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан189", "авто-функция 189", "🎉 Функция #189 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан190", "авто-функция 190", "🎉 Функция #190 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан191", "авто-функция 191", "🎉 Функция #191 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан192", "авто-функция 192", "🎉 Функция #192 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан193", "авто-функция 193", "🎉 Функция #193 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан194", "авто-функция 194", "🎉 Функция #194 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан195", "авто-функция 195", "🎉 Функция #195 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан196", "авто-функция 196", "🎉 Функция #196 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан197", "авто-функция 197", "🎉 Функция #197 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан198", "авто-функция 198", "🎉 Функция #198 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан199", "авто-функция 199", "🎉 Функция #199 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
    ("фан200", "авто-функция 200", "🎉 Функция #200 для {user}: {choice:мем|вайб|GIF|картинка|ссылка|стикер|хаос} +{rand:1:100}%"),
]


def render_extra_response(template: str, user: str, argument: str, latency_ms: int) -> str:
    now = datetime.datetime.utcnow()

    def replace_choice(match: re.Match[str]) -> str:
        return random.choice(match.group(1).split("|"))

    def replace_arg_or_choice(match: re.Match[str]) -> str:
        return argument.strip() or random.choice(match.group(1).split("|"))

    def replace_rand(match: re.Match[str]) -> str:
        low, high = int(match.group(1)), int(match.group(2))
        return str(random.randint(low, high))

    response = template
    response = re.sub(r"\{choice:([^{}]+)\}", replace_choice, response)
    response = re.sub(r"\{arg_or_choice:([^{}]+)\}", replace_arg_or_choice, response)
    response = re.sub(r"\{rand:(-?\d+):(-?\d+)\}", replace_rand, response)
    response = response.replace("{user}", user)
    response = response.replace("{latency}", str(latency_ms))
    response = response.replace("{date}", now.strftime("%Y-%m-%d"))
    response = response.replace("{time}", now.strftime("%H:%M:%S"))
    response = response.replace("{arg_or:скажи что-нибудь после команды}", argument or "скажи что-нибудь после команды")
    response = response.replace("{arg_upper_or:НЕЧЕГО КРИЧАТЬ}", argument.upper() if argument else "НЕЧЕГО КРИЧАТЬ")
    response = response.replace("{arg_lower_or:тихо...}", argument.lower() if argument else "тихо...")
    response = response.replace("{arg_reverse_or:текст не найден}", argument[::-1] if argument else "текст не найден")
    response = response.replace("{arg_len}", str(len(argument)))
    response = response.replace("{arg_words}", str(len(argument.split())))
    response = response.replace("{arg_letters}", str(sum(char.isalpha() for char in argument)))
    return response


def make_extra_command(command_name: str, template: str):
    async def extra_command(ctx: commands.Context, *, argument: str = "") -> None:
        latency_ms = round(bot.latency * 1000)
        response = render_extra_response(template, ctx.author.display_name, argument, latency_ms)
        await ctx.reply(discord.utils.escape_mentions(response))

    extra_command.__name__ = f"extra_{command_name}"
    return commands.command(name=command_name)(extra_command)


for _command_name, _description, _template in EXTRA_COMMANDS:
    bot.add_command(make_extra_command(_command_name, _template))


@bot.command(name="функции")
async def list_extra_commands(ctx: commands.Context) -> None:
    names = ", ".join(f"!{name}" for name, _, _ in EXTRA_COMMANDS)
    await ctx.reply(f"Добавлено {len(EXTRA_COMMANDS)} функций:\n{names}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Set DISCORD_TOKEN environment variable")
    bot.run(DISCORD_TOKEN)
