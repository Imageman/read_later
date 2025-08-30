#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Локальный офлайн-пайплайн:
1) Читает RSS/Atom.
2) Для каждой записи тянет веб-страницу, извлекает "чистый" HTML (newspaper4k)
   и конвертирует в Markdown (html2text / pypandoc / MarkItDown).
3) Фильтрует по ключевым словам/доменам/минимальной длине.
4) Сохраняет в Markdown с YAML front matter в output_dir.
5) Ведёт базу "seen" (SQLite) — без дублей.
6) Авто-удаляет заметки старше retention_days.
7) Опционально скачивает изображения и переписывает ссылки на локальные.

Запуск:
    python news_grabber.py            # использует config.yaml по умолчанию
    python news_grabber.py --config custom.yaml
Добавьте в Планировщик задач Windows для периодического запуска.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from loguru import logger
import os
import re
import shutil
import sqlite3
import sys
import time
import traceback
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse, urljoin

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz
from dateutil.parser import parse as date_parse

# newspaper4k
from newspaper import Article as NPArticle

# HTML → Markdown converters
import html2text
import pypandoc
from markitdown import MarkItDown

# -------------------- CONSTANTS --------------------
FEED_DIR_NAME_LIMIT = 20
DEFAULT_FEED_DIR_NAME = "feed"

# -------------------- ЛОГИ --------------------
def setup_logger(log_file: Path):
    """Configure logging using Loguru."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Логирование в файлы
    logger.add(log_file, rotation="10 MB",  retention=3,  encoding="utf-8", backtrace=True,
               level='DEBUG', diagnose=True)

    try:
        # Логирование в консоль
        logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>",
                   level='DEBUG')
    except Exception as error:
        # Эта ошибка может возникнуть, если нет доступной консоли (например, при запуске с pythonw.exe)
        # Логируем это в файл для отладки, но не прерываем работу.
        logger.info(f"Could not add console logger: {error}\n{traceback.format_exc()}")


    logger.info("Logging is configured.")


# -------------------- МОДЕЛИ --------------------
@dataclass
class FeedItem:
    feed_name: str
    title: str
    link: str
    published: Optional[datetime]
    summary: str

@dataclass
class Article:
    title: str
    authors: List[str]
    content_markdown: str
    content_text: str
    top_image: Optional[str]
    url: str
    published: Optional[datetime]
    site: str

# -------------------- УТИЛИТЫ --------------------
def safe_filename(name: str, limit: int = 120) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[:limit].rstrip()
    return s or "untitled"


def safe_feed_dir(name: str, limit: int = FEED_DIR_NAME_LIMIT) -> str:
    """Return sanitized feed name suitable for directory creation."""
    sanitized = safe_filename(name, limit=limit).replace(" ", "_")
    return sanitized or DEFAULT_FEED_DIR_NAME

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def now_local() -> datetime:
    return datetime.now(tz.tzlocal())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# -------------------- БД ПРОСМОТРЕННОГО --------------------
class SeenDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_dir(db_path.parent)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, url TEXT, created_at INTEGER)"
            )
            conn.commit()

    def exists(self, uid: str) -> bool:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute("SELECT 1 FROM seen WHERE id = ?", (uid,))
            return cur.fetchone() is not None

    def add(self, uid: str, url: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen (id, url, created_at) VALUES (?,?,?)",
                (uid, url, int(time.time())),
            )
            conn.commit()

# -------------------- ИЗВЛЕЧЕНИЕ КОНТЕНТА --------------------
def build_http_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    s.timeout = 30  # type: ignore[attr-defined]
    return s

def html_to_markdown(html_str: str, base_url: str, cfg: Dict) -> str:
    """Convert HTML to Markdown using selected engine."""
    engine = (cfg.get("engine") or cfg.get("name") or "html2text").lower()
    if engine == "pypandoc":
        extra = cfg.get("pypandoc", {}).get("extra_args", ["--wrap=none"])
        return pypandoc.convert_text(html_str, to="md", format="html", extra_args=extra)
    if engine == "markitdown":
        mk = MarkItDown()
        if base_url:
            html_str = f'<base href="{base_url}">{html_str}'
        res = mk.convert(BytesIO(html_str.encode("utf-8")), "text/html")
        if hasattr(res, "text_content"):
            return res.text_content
        if hasattr(res, "markdown"):
            return res.markdown
        return str(res)
    # default html2text
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.baseurl = base_url
    return h.handle(html_str)


def extract_article(session: requests.Session, url: str, conv_cfg: Dict) -> Article:
    resp = session.get(url, allow_redirects=True, timeout=30)
    resp.raise_for_status()

    art = NPArticle(url)
    # newspaper4k <=0.9 had `set_html`, newer versions expect `download(input_html=...)`
    if hasattr(art, "set_html"):
        art.set_html(resp.text)
    else:
        # fall back to the new API while avoiding a second network request
        art.download(input_html=resp.text)
    art.parse()

    content_html = art.article_html or art.html or resp.text
    content_md = html_to_markdown(content_html, url, conv_cfg)
    content_text = art.text or BeautifulSoup(content_html, "lxml").get_text("\n", strip=True)

    title = art.title or ""
    authors = art.authors or []
    published = art.publish_date
    top_image = art.top_image if getattr(art, "top_image", None) else None
    site = domain_of(url)

    if not title:
        soup = BeautifulSoup(resp.text, "lxml")
        t = soup.title.string if soup.title and soup.title.string else ""
        title = (t or url).strip()

    return Article(
        title=title.strip(),
        authors=authors,
        content_markdown=content_md.strip(),
        content_text=content_text.strip(),
        top_image=top_image,
        url=url,
        published=published,
        site=site,
    )

# -------------------- ФИЛЬТРЫ --------------------
def match_keywords(text: str, inc: List[str], exc: List[str]) -> bool:
    T = text.lower()
    if inc:
        ok = any(k.lower() in T for k in inc)
        if not ok:
            return False
    if exc and any(k.lower() in T for k in exc):
            return False
    return True

def is_domain_allowed(url: str, allow: List[str], deny: List[str]) -> bool:
    d = domain_of(url)
    if deny and any(d.endswith(x) for x in deny):
        return False
    if allow and not any(d.endswith(x) for x in allow):
        return False
    return True

# -------------------- СОХРАНЕНИЕ --------------------
def build_front_matter(a: Article, feed_name: str, tags: List[str]) -> str:
    published_iso = a.published.isoformat() if a.published else None
    fm = {
        "title": a.title,
        "source": a.url,
        "site": a.site,
        "authors": a.authors or [],
        "feed": feed_name,
        "created": now_local().isoformat(),
        "published": published_iso,
        "tags": tags,
    }
    # YAML вручную для предсказуемости
    def yaml_escape(s: str) -> str:
        s = s.replace('"', '\\"')
        if ":" in s or "-" in s:
            return f'"{s}"'
        return s
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(yaml_escape(x) for x in v)}]")
        elif v is None:
            lines.append(f"{k}: ")
        else:
            lines.append(f"{k}: {yaml_escape(str(v))}")
    lines.append("---")
    return "\n".join(lines)

def rewrite_and_download_images(a: Article, out_dir: Path, session: requests.Session) -> str:
    """Скачивает <img src> и переписывает на локальные пути. Возвращает изменённый Markdown."""
    md = a.content_markdown
    # Наивный разбор Markdown-картинок ![alt](url)
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    def repl(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2)
        if url.startswith("data:"):
            return m.group(0)
        try:
            r = session.get(urljoin(a.url, url), timeout=20)
            r.raise_for_status()
            ext = os.path.splitext(urlparse(url).path)[1] or ".img"
            fname = f"img_{hash_id(url)}{ext}"
            img_dir = out_dir / "assets"
            ensure_dir(img_dir)
            with open(img_dir / fname, "wb") as f:
                f.write(r.content)
            rel = f"./assets/{fname}"
            return f"![{alt}]({rel})"
        except Exception as e:
            logger.debug(f"Image fetch failed: {url} -> {e}\n{traceback.format_exc()}")
            return m.group(0)
    return pattern.sub(repl, md)

def save_markdown(
    a: Article,
    feed_name: str,
    out_root: Path,
    tags: List[str],
    download_images: bool,
    session: requests.Session,
) -> Path:
    """Save article as Markdown file inside a feed-named directory."""
    out_dir = out_root / safe_feed_dir(feed_name)
    ensure_dir(out_dir)

    fname = f"{safe_filename(a.title)}_{hash_id(a.url)}.md"
    path = out_dir / fname

    body_md = a.content_markdown
    if download_images:
        body_md = rewrite_and_download_images(a, out_dir, session)

    front = build_front_matter(a, feed_name, tags)
    with open(path, "w", encoding="utf-8") as f:
        f.write(front + "\n\n" + f"> Источник: [{a.site}]({a.url})\n\n" + body_md + "\n")
    return path

# -------------------- СБОРОЩИК --------------------
def fetch_feed_items(session: requests.Session, url: str, feed_name: str) -> List[FeedItem]:
    logger.info(f"Fetching feed: {feed_name} | {url}")
    d = feedparser.parse(url)
    items: List[FeedItem] = []
    for e in d.entries:
        link = getattr(e, "link", None)
        title = getattr(e, "title", "") or ""
        summary = getattr(e, "summary", "") or ""
        published = None
        # published_parsed, updated_parsed…
        for attr in ("published", "updated"):
            val = getattr(e, attr, None)
            if val:
                try:
                    published = date_parse(val)
                    break
                except Exception:
                    pass
        if not link:
            # иногда ссылка в "id"
            link = getattr(e, "id", None)
        if not link:
            continue
        items.append(
            FeedItem(
                feed_name=feed_name,
                title=title,
                link=link,
                published=published,
                summary=BeautifulSoup(summary, "lxml").get_text(" ", strip=True),
            )
        )
    return items

def clean_old_notes(out_root: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 86400
    deleted = 0
    for p in out_root.rglob("*.md"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted += 1
        except Exception as e:
            logger.debug(f"Failed to delete {p}: {e}\n{traceback.format_exc()}")
    if deleted:
        logger.info(f"Deleted old notes: {deleted}")

# -------------------- MAIN --------------------
def main() -> int:
    log_file = "news_grabber.log"
    setup_logger(log_file)
    ap = argparse.ArgumentParser(description="Local RSS → Markdown pipeline for Obsidian")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--max", type=int, default=None, help="Global max articles per run")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    converter_cfg = cfg.get("markdown_converter", {})

    output_dir = Path(cfg["output_dir"]).expanduser()
    ensure_dir(output_dir)


    retention_days = int(cfg.get("retention_days", 30))
    min_chars = int(cfg.get("min_chars", 600))
    download_images = bool(cfg.get("download_images", False))
    max_articles_per_feed = int(cfg.get("max_articles_per_feed", 15))
    user_agent = cfg.get("user_agent", "NewsGrabber/1.0")

    g_inc = cfg.get("include_keywords", []) or []
    g_exc = cfg.get("exclude_keywords", []) or []
    g_allow = cfg.get("allow_domains", []) or []
    g_deny = cfg.get("deny_domains", []) or []

    feeds = cfg.get("feeds", [])
    if not feeds:
        logger.error("No feeds configured.")
        return 2

    seen = SeenDB(output_dir / "seen.db")
    session = build_http_session(user_agent)

    total_saved = 0

    for feed in feeds:
        name = feed.get("name", "Feed")
        url = feed["url"]
        f_inc = feed.get("include_keywords", [])
        f_exc = feed.get("exclude_keywords", [])
        f_allow = feed.get("allow_domains", [])
        f_deny = feed.get("deny_domains", [])
        tags = feed.get("tags", []) or []

        items = fetch_feed_items(session, url, name)
        count = 0

        for it in items:
            if args.max is not None and total_saved >= args.max:
                break
            if count >= max_articles_per_feed:
                break

            uid = hash_id(it.link)
            if seen.exists(uid):
                continue

            # Домены
            if not is_domain_allowed(it.link, f_allow or g_allow, f_deny or g_deny):
                continue

            # Грубая фильтрация по заголовку/анонсу заранее
            pre_text = f"{it.title}\n{it.summary}"
            if not match_keywords(pre_text, f_inc or g_inc, f_exc or g_exc):
                continue

            try:
                art = extract_article(session, url=it.link, conv_cfg=converter_cfg)
            except Exception as e:
                logger.warning(f"Extract failed: {it.link} -> {e}\n{traceback.format_exc()}")
                continue

            # Повторная фильтрация по полному тексту
            full_text = f"{it.title}\n{art.content_text}"
            if not match_keywords(full_text, f_inc or g_inc, f_exc or g_exc):
                continue

            if len(art.content_text) < min_chars:
                continue

            save_path = save_markdown(
                art,
                feed_name=name,
                out_root=output_dir,
                tags=tags,
                download_images=download_images,
                session=session,
            )
            seen.add(uid, it.link)
            count += 1
            total_saved += 1
            logger.info(f"Saved: {save_path}")

        logger.info(f"Feed done: {name} (saved {count})")
        if args.max is not None and total_saved >= args.max:
            break

    clean_old_notes(output_dir, retention_days)
    logger.info(f"All done. Total saved: {total_saved}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
