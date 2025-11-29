#!/usr/bin/env python3
# bot_python_only_v4.py (Full Flask + Webhook for Render.com)
# Python-only Universal File Host v4 with Flask webhook support
# Run on Render: Set env vars BOT_TOKEN and OWNER_ID
# Requirements: pip install pyTelegramBotAPI psutil flask gunicorn gitpython

import os
import sys
import time
import json
import zipfile
import shutil
import subprocess
import logging
import threading
import ast
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import telebot
from telebot import types
import psutil
import tempfile

# Flask for webhook
from flask import Flask, request, abort

# ---------------- CONFIG (Load from ENV - NO HARDCODE) ----------------
TOKEN = os.environ.get('BOT_TOKEN')
OWNER_ID = int(os.environ.get('OWNER_ID', '0'))
ADMIN_IDS = [OWNER_ID]  # Add more admins via env if needed, e.g., os.environ.get('ADMIN_IDS', '').split(',')

if not TOKEN:
    raise ValueError("BOT_TOKEN env var required!")

# Limits
FREE_USER_FILE_LIMIT = 10
FREE_USER_RUN_LIMIT = 10
MAX_FILE_SIZE = 100 * 1024 * 1024
TELEGRAM_MAX_MSG = 4000

# Webhook path (for Render)
WEBHOOK_PATH = '/webhook'
WEBHOOK_URL = os.environ.get('WEBHOOK_URL', f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost')}{WEBHOOK_PATH}")

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_BOTS_DIR = os.path.join(BASE_DIR, "upload_bots")
LOGS_DIR = os.path.join(BASE_DIR, "execution_logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
BANNED_FILE = os.path.join(DATA_DIR, "banned.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

os.makedirs(UPLOAD_BOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MAIN_PRIORITY = ["main.py", "bot.py", "app.py", "rs.py"]
EXECUTABLE_EXTS = {".py"}

STANDARD_LIBS: Set[str] = {
    'os', 'sys', 'time', 'json', 'threading', 'datetime', 'pathlib', 'typing',
    'subprocess', 'logging', 'tempfile', 'zipfile', 'shutil', 'collections',
    'itertools', 're', 'math', 'random', 'urllib', 'http', 'socket', 'io',
    'functools', 'operator', 'enum', 'abc', 'contextlib', 'warnings', 'traceback',
    'inspect', 'gc', 'weakref', 'copy', 'pprint', 'gettext', 'locale', 'codecs',
    'unicodedata', 'string', 'configparser', 'argparse', 'csv', 'sqlite3',
    'hashlib', 'hmac', 'secrets', 'base64', 'binascii', 'struct', 'calendar',
    'email', 'smtplib', 'imaplib', 'poplib', 'ftplib', 'http', 'urllib', 'xml',
    'html', 'json', 'tarfile', 'gzip', 'bz2', 'lzma', 'zipfile', 'codecs',
    'asyncio', 'concurrent', 'multiprocessing', 'signal', 'mmap', 'readline',
    'rlcompleter', 'pdb', 'profile', 'cProfile', 'timeit', 'trace', 'tracemalloc'
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "bot_v4.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bot_v4")

# Bot initialization
bot = telebot.TeleBot(TOKEN)

# In-memory state
user_files = {}       # user_id -> list of (display_name, full_path, type) where type='main'|'other'
hidden_deps = {}      # user_id -> list of dependency file paths (not shown in Check Files)
running_scripts = {}  # key "userid_basename" -> info dict {proc, start_time, log, path}
pending_prompts = {}  # user_id -> dict for pending states
banned_users = set()  # loaded from disk

# ---------------- Persistence for banned users and full state ----------------
def load_banned():
    global banned_users
    try:
        if os.path.exists(BANNED_FILE):
            with open(BANNED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            banned_users = set(int(x) for x in data.get("banned", []))
        else:
            banned_users = set()
    except Exception as e:
        logger.warning("Failed to load banned.json: %s", e)
        banned_users = set()


def save_banned():
    try:
        with open(BANNED_FILE, "w", encoding="utf-8") as f:
            json.dump({"banned": list(banned_users)}, f)
    except Exception as e:
        logger.error("Failed saving banned.json: %s", e)


def load_users_state():
    """Load persisted user_files, hidden_deps, running_scripts (paths & metadata only)."""
    global user_files, hidden_deps, running_scripts
    if not os.path.exists(USERS_FILE):
        logger.info("No users.json found, starting fresh.")
        return
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure integer keys for user_files and hidden_deps
        uf = {}
        for k, v in data.get("user_files", {}).items():
            try:
                uf[int(k)] = v
            except:
                uf[k] = v
        hd = {}
        for k, v in data.get("hidden_deps", {}).items():
            try:
                hd[int(k)] = v
            except:
                hd[k] = v
        user_files = uf
        hidden_deps = hd
        # running_scripts persisted as dict key-> {path, start_time, log}
        persisted_running = data.get("running_scripts", {})
        running_scripts = {}
        for k, v in persisted_running.items():
            # keep metadata; 'proc' will be created on auto-restore
            running_scripts[k] = {"proc": None, "path": v.get("path"), "start_time": v.get("start_time"), "log": v.get("log")}
        logger.info("Loaded persistent user state: users=%d, hidden_deps=%d, running=%d",
                    len(user_files), len(hidden_deps), len(running_scripts))
    except Exception as e:
        logger.exception("Failed to load users.json: %s", e)
        user_files = {}
        hidden_deps = {}
        running_scripts = {}


def save_users_state():
    """Persist only serializable parts (no subprocess objects)."""
    try:
        # prepare serializable dicts
        user_files_serial = {str(k): v for k, v in user_files.items()}
        hidden_deps_serial = {str(k): v for k, v in hidden_deps.items()}
        running_serial = {}
        for k, v in running_scripts.items():
            running_serial[k] = {"path": v.get("path"), "start_time": v.get("start_time"), "log": v.get("log")}
        data = {"user_files": user_files_serial, "hidden_deps": hidden_deps_serial, "running_scripts": running_serial}
        tmp = USERS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, USERS_FILE)
    except Exception as e:
        logger.exception("Failed saving users.json: %s", e)


# initialize persisted state
load_banned()
load_users_state()

# ---------------- Helpers ----------------
def get_user_folder(user_id: int) -> str:
    folder = os.path.join(UPLOAD_BOTS_DIR, str(user_id))
    os.makedirs(folder, exist_ok=True)
    return folder


def add_user_file(user_id: int, display_name: str, full_path: str, ftype: str):
    """Add file to visible list (ftype: 'main' or 'other') and persist."""
    lst = user_files.setdefault(user_id, [])
    # avoid duplicates by path
    for d, p, t in lst:
        if os.path.abspath(p) == os.path.abspath(full_path):
            return
    lst.append((display_name, full_path, ftype))
    save_users_state()


def add_hidden_dep(user_id: int, path: str):
    lst = hidden_deps.setdefault(user_id, [])
    norm = os.path.abspath(path)
    if norm not in [os.path.abspath(x) for x in lst]:
        lst.append(path)
        save_users_state()


def find_py_files(folder: str):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(".py"):
                files.append(os.path.join(root, fn))
    return files


def find_main_candidates(folder: str):
    """Return list of candidate main files in order of priority."""
    py_files = find_py_files(folder)
    def score(fp):
        name = os.path.basename(fp).lower()
        for idx, p in enumerate(MAIN_PRIORITY):
            if name == p:
                return idx
        return len(MAIN_PRIORITY) + 1
    py_files.sort(key=lambda p: (score(p), os.path.basename(p)))
    return py_files


def extract_imports(file_path: str) -> Set[str]:
    """Extract top-level import module names from a Python file using AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        return imports
    except Exception as e:
        logger.warning("Failed to parse imports from %s: %s", file_path, e)
        return set()


def is_standard_lib(pkg: str) -> bool:
    """Check if a package is a standard Python library module."""
    return pkg in STANDARD_LIBS


def detect_and_install_requirements(folder: str) -> tuple[bool, str]:
    """Detect requirements from imports if no requirements.txt exists, then install."""
    req_path = os.path.join(folder, "requirements.txt")
    if os.path.exists(req_path):
        return pip_install_requirements(req_path)  # Use original for existing file

    # Detect imports from all .py files
    py_files = find_py_files(folder)
    if not py_files:
        return True, "No Python files found for requirement detection"

    all_imports = set()
    for py_file in py_files:
        imports = extract_imports(py_file)
        # Filter non-standard libs (top-level only)
        non_std = {imp for imp in imports if not is_standard_lib(imp) and not imp.startswith('_')}
        all_imports.update(non_std)

    if not all_imports:
        return True, "No external requirements detected"

    # Create temporary requirements.txt
    temp_req = os.path.join(folder, "detected_requirements.txt")
    try:
        with open(temp_req, 'w', encoding='utf-8') as f:
            for pkg in sorted(all_imports):
                f.write(f"{pkg}\n")
        
        # Install
        return pip_install_requirements(temp_req)
    except Exception as e:
        return False, f"Detection/install failed: {e}"
    finally:
        # Clean up temp file
        if os.path.exists(temp_req):
            os.remove(temp_req)


def pip_install_requirements(req_file: str) -> tuple[bool, str]:
    """Install from a requirements.txt file."""
    folder = os.path.dirname(req_file)
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file],
                             capture_output=True, text=True, timeout=600, cwd=folder)
        if res.returncode == 0:
            return True, f"Requirements installed: {len(open(req_file).readlines())} packages"
        else:
            return False, f"Pip failed: {res.stderr[:1000]}"
    except Exception as e:
        return False, f"Exception pip install: {e}"


def forward_upload_to_admins(uploader_msg, saved_path: str):
    """Forward uploaded file to all admins with uploader info and file."""
    sender = uploader_msg.from_user
    info_text = (
        f"ðŸ“¥ New upload received\n\n"
        f"ðŸ‘¤ From: {sender.first_name or ''} {getattr(sender, 'last_name', '') or ''}\n"
        f"ðŸ”— Username: @{sender.username if getattr(sender, 'username', None) else 'N/A'}\n"
        f"ðŸ†” UserID: {sender.id}\n"
        f"ðŸ•’ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"ðŸ“„ File: {os.path.basename(saved_path)}\n"
        f"ðŸ“ Path: {saved_path}\n"
    )
    for admin in ADMIN_IDS:
        try:
            bot.send_message(admin, info_text)
            with open(saved_path, "rb") as df:
                bot.send_document(admin, df, caption=f"Forwarded upload from {sender.id}")
        except Exception as e:
            logger.warning("Forwarding failed to admin %s: %s", admin, e)


def start_python_script(user_id: int, script_path: str):
    """Start a .py script as background process and log output.
       ALWAYS install requirements FIRST (detect if needed)."""
    if not os.path.exists(script_path):
        return False, "File not found"

    if not script_path.lower().endswith(".py"):
        return False, "Only .py files can be started"

    # ALWAYS detect and install requirements before starting
    folder = os.path.dirname(script_path)
    req_ok, req_msg = detect_and_install_requirements(folder)
    if not req_ok:
        return False, f"Requirements failed: {req_msg}"

    logfn = os.path.join(LOGS_DIR, f"log_{user_id}_{int(time.time())}_{os.path.basename(script_path)}.txt")
    try:
        f = open(logfn, "w", encoding="utf-8")
    except Exception as e:
        return False, f"Log creation failed: {e}"

    try:
        proc = subprocess.Popen([sys.executable, script_path],
                                stdout=f, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(script_path),
                                env=os.environ.copy())
    except Exception as e:
        f.close()
        return False, f"Failed to start: {e}"

    key = f"{user_id}_{os.path.basename(script_path)}"
    running_scripts[key] = {"proc": proc, "start_time": datetime.now().isoformat(), "log": logfn, "path": script_path}
    save_users_state()
    logger.info("Started script %s for user %s (pid %s)", script_path, user_id, proc.pid)
    return True, f"Started PID {proc.pid} (logs: {logfn}). Requirements: {req_msg}"


def stop_python_script(user_id: int, basename: str):
    key = f"{user_id}_{basename}"
    info = running_scripts.get(key)
    if not info:
        return False, "Not running"
    proc = info.get("proc")
    try:
        if proc:
            proc.terminate()
            proc.wait(timeout=8)
    except Exception:
        try:
            if proc:
                proc.kill()
        except Exception:
            pass
    running_scripts.pop(key, None)
    save_users_state()
    return True, "Stopped"


def read_log_tail(path: str, lines=50):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read().splitlines()
            return "\n".join(data[-lines:])
    except Exception as e:
        return f"Could not read log: {e}"


# ---------------- safe send helpers for long messages / logs ----------------
def send_long_message(chat_id: int, text: str, prefix: Optional[str] = None):
    """Chunk text into TELEGRAM_MAX_MSG pieces and send sequentially."""
    try:
        if prefix:
            bot.send_message(chat_id, prefix)
        for i in range(0, len(text), TELEGRAM_MAX_MSG):
            part = text[i:i + TELEGRAM_MAX_MSG]
            bot.send_message(chat_id, part)
    except Exception as e:
        logger.warning("send_long_message failed: %s", e)
        # best-effort: try smaller chunks
        try:
            for i in range(0, len(text), TELEGRAM_MAX_MSG // 2):
                bot.send_message(chat_id, text[i:i + TELEGRAM_MAX_MSG // 2])
        except Exception as e2:
            logger.exception("send_long_message fallback failed: %s", e2)


def send_log_or_file(chat_id: int, text: str, filename_hint: str = "log.txt"):
    """
    If text is small => send as message.
    If large => try to attach as file. If attach fails, fall back to chunked messages.
    """
    try:
        if not text:
            bot.send_message(chat_id, "ðŸ“œ No log available.")
            return
        if len(text) <= TELEGRAM_MAX_MSG:
            bot.send_message(chat_id, f"ðŸ“œ Log tail:\n\n{text}")
            return
        # try sending as file
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix="_" + filename_hint) as tf:
            tf.write(text)
            tmp_path = tf.name
        try:
            with open(tmp_path, "rb") as doc:
                bot.send_document(chat_id, doc, caption="ðŸ“œ Full log attached")
        except Exception as e:
            logger.warning("Sending log as document failed: %s", e)
            # fallback to chunked messages
            send_long_message(chat_id, text, prefix="ðŸ“œ Log is large; sending in parts:")
        finally:
            try:
                os.remove(tmp_path)
            except:
                pass
    except Exception as e:
        logger.exception("send_log_or_file failed: %s", e)


# ---------------- Telegram UI / Handlers ----------------

def build_main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    rows = [
        ["ðŸ“¢ Updates Channel"],
        ["ðŸ“¤ Upload File", "ðŸ“‚ Check Files"],
        ["âš¡ Bot Speed", "ðŸ“Š Statistics"],
        ["ðŸ¤– Git Clone", "ðŸ“ž Contact Owner"]
    ]
    for r in rows:
        markup.add(*[types.KeyboardButton(x) for x in r])
    return markup


@bot.message_handler(commands=["start"])
def cmd_start(m):
    uid = m.from_user.id
    if uid in banned_users:
        bot.reply_to(m, "âŒ You are banned from using this bot.")
        return
    markup = build_main_keyboard()
    bot.send_message(m.chat.id, f"ðŸ‘‹ Hello {m.from_user.first_name or ''}!\nPython-only Universal File Host (v4)\nUpload any file; .py and .zip preferred.", reply_markup=markup)


@bot.message_handler(func=lambda m: m.text == "ðŸ“¢ Updates Channel")
def updates(m):
    bot.reply_to(m, "ðŸ“¢ Channel: @CARTOONFUNNY03")


@bot.message_handler(func=lambda m: m.text == "âš¡ Bot Speed")
def speed(m):
    t0 = time.time()
    msg = bot.reply_to(m, "Checking...")
    elapsed = int((time.time() - t0) * 1000)
    bot.edit_message_text(f"âš¡ Response {elapsed} ms", msg.chat.id, msg.message_id)


@bot.message_handler(func=lambda m: m.text == "ðŸ“Š Statistics")
def stats(m):
    total_users = len(user_files)
    total_files = sum(len(v) for v in user_files.values())
    running = len([k for k, v in running_scripts.items() if v.get("proc")])
    bot.reply_to(m, f"ðŸ“Š Users: {total_users}\nFiles tracked: {total_files}\nRunning scripts (live): {running}")


@bot.message_handler(func=lambda m: m.text == "ðŸ“ž Contact Owner")
def contact(m):
    bot.reply_to(m, f"Owner ID: {OWNER_ID}")


@bot.message_handler(func=lambda m: m.text == "ðŸ“¤ Upload File")
def upload_hint(m):
    bot.reply_to(m, "ðŸ“¤ Send me a file: any type. .zip files containing .py are supported. Max size per file: {} MB".format(MAX_FILE_SIZE // (1024 * 1024)))


@bot.message_handler(func=lambda m: m.text == "ðŸ¤– Git Clone")
def git_clone_prompt(m):
    uid = m.from_user.id
    if uid in banned_users:
        bot.reply_to(m, "âŒ You are banned.")
        return
    bot.reply_to(m, "ðŸ”— Send the repo URL (https://github.com/owner/repo.git) or 'cancel' to abort.")
    pending_prompts[uid] = {"expecting_clone_link": True}


@bot.message_handler(func=lambda m: m.text == "ðŸ“‚ Check Files")
def check_files_ui(m):
    uid = m.from_user.id
    if uid in banned_users:
        bot.reply_to(m, "âŒ You are banned.")
        return
    lst = user_files.get(uid, [])
    if not lst:
        bot.reply_to(m, "ðŸ“‚ You have no visible files. Upload .py or a .zip with a main file.")
        return
    markup = types.InlineKeyboardMarkup(row_width=1)
    for idx, (display, full, ftype) in enumerate(lst):
        status = "ðŸŸ¢" if f"{uid}_{os.path.basename(full)}" in running_scripts and running_scripts.get(f"{uid}_{os.path.basename(full)}", {}).get("proc") else "â¯ï¸"
        markup.add(types.InlineKeyboardButton(f"{status} {display}", callback_data=f"file_{idx}"))
    bot.reply_to(m, "ðŸ“‚ Your Files (only main files are shown):", reply_markup=markup)


# Generic text handler to manage pending prompts (clone link, env edit, choose main)
@bot.message_handler(func=lambda m: True, content_types=['text'])
def generic_text(m):
    uid = m.from_user.id
    txt = m.text.strip()
    if uid in banned_users:
        bot.reply_to(m, "âŒ You are banned.")
        return

    state = pending_prompts.get(uid, {})
    # Handle clone link expected
    if state.get("expecting_clone_link"):
        if txt.lower() in ("cancel", "stop"):
            pending_prompts.pop(uid, None)
            bot.reply_to(m, "Cancelled clone.")
            return
        repo = txt
        bot.reply_to(m, "â³ Cloning repository... please wait (this can take a while).")
        success, result = clone_repo_and_process(uid, repo)
        if not success:
            bot.reply_to(m, f"âŒ Clone failed: {result}")
            pending_prompts.pop(uid, None)
            return
        # result is clone_folder
        clone_folder = result
        # find main candidates
        mains = find_main_candidates(clone_folder)
        if mains:
            chosen = mains[0]
            add_user_file(uid, os.path.relpath(chosen, get_user_folder(uid)), chosen, "main")
            ok, msg = start_python_script(uid, chosen)
            bot.reply_to(m, f"{'âœ… Cloned and started' if ok else 'âš ï¸ Cloned but failed to start'}: {msg}")
            pending_prompts.pop(uid, None)
            return
        else:
            # ask user to provide main filename
            pending_prompts[uid] = {"expecting_main_from_clone": clone_folder}
            bot.reply_to(m, "âš ï¸ No prioritized main file found automatically. Please reply with the filename to run (relative path within repo, e.g. app/main.py or main.py).")
            return

    # Handle expecting main from clone
    if state.get("expecting_main_from_clone"):
        chosen = txt
        clone_folder = state["expecting_main_from_clone"]
        candidate = resolve_candidate_in_folder(clone_folder, chosen)
        if not candidate:
            bot.reply_to(m, f"âŒ File '{chosen}' not found in cloned repo.")
            return
        add_user_file(uid, os.path.relpath(candidate, get_user_folder(uid)), candidate, "main")
        ok, msg = start_python_script(uid, candidate)
        bot.reply_to(m, f"{'âœ… Started' if ok else 'âŒ Failed to start'}: {msg}")
        pending_prompts.pop(uid, None)
        return

    # Handle expecting main from extract
    if state.get("expecting_main_from_extract"):
        chosen = txt
        extract_folder = state["expecting_main_from_extract"]
        candidate = resolve_candidate_in_folder(extract_folder, chosen)
        if not candidate:
            bot.reply_to(m, f"âŒ File '{chosen}' not found in extracted folder.")
            return
        add_user_file(uid, os.path.relpath(candidate, get_user_folder(uid)), candidate, "main")
        ok, msg = start_python_script(uid, candidate)
        bot.reply_to(m, f"{'âœ… Started' if ok else 'âŒ Failed to start'}: {msg}")
        pending_prompts.pop(uid, None)
        return

    # Handle expecting .env edit: user will reply "yes" to start edit, then send content lines
    if state.get("expecting_env_edit"):
        env_path = state["expecting_env_edit"]
        if txt.lower() in ("no", "cancel", "skip"):
            bot.reply_to(m, "Skipped editing .env. You can edit manually later.")
            pending_prompts.pop(uid, None)
            return
        if txt.lower() in ("yes", "y"):
            pending_prompts[uid] = {"expecting_env_content": env_path}
            bot.reply_to(m, "Send the .env content now in format KEY=VALUE per line. When done send /done")
            return

    if state.get("expecting_env_content"):
        if txt.strip() == "/done":
            bot.reply_to(m, "Finished editing .env.")
            pending_prompts.pop(uid, None)
            return
        env_path = state["expecting_env_content"]
        # append the sent text lines to env file
        try:
            with open(env_path, "a", encoding="utf-8") as ef:
                ef.write(txt + "\n")
            bot.reply_to(m, "Appended to .env.")
        except Exception as e:
            bot.reply_to(m, f"Failed to write to .env: {e}")
        return

    # fallback
    if txt.startswith("/"):
        # some admin commands like /ban handled below
        return

    bot.reply_to(m, "I didn't understand that. Use buttons: Upload File, Check Files, Git Clone, etc.")


# ---------- helpers used in message handlers ----------
def resolve_candidate_in_folder(base_folder: str, relative_or_name: str) -> Optional[str]:
    """Find file by exact name or relative path inside base_folder."""
    # exact match by name
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f == relative_or_name or os.path.relpath(os.path.join(root, f), base_folder) == relative_or_name:
                return os.path.join(root, f)
    return None


def clone_repo_and_process(user_id: int, repo_url: str) -> tuple[bool, str]:
    """Clone repo into user folder, install requirements (detect if needed), detect .env, register py files."""
    user_folder = get_user_folder(user_id)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target = os.path.join(user_folder, f"cloned_{repo_name}_{int(time.time())}")
    try:
        res = subprocess.run(["git", "clone", repo_url, target], capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            return False, f"git clone failed: {res.stderr[:800]}"
        # register .py files as hidden by default, main candidate detection will add main visible
        for root, _, files in os.walk(target):
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    add_hidden_dep(user_id, full)
        # check requirements (now detects if no file)
        ok, msg = detect_and_install_requirements(target)
        if not ok:
            logger.warning("Requirements failed during clone: %s", msg)
        # check .env or config.json
        env_path = None
        for candidate in (".env", "config.json"):
            p = os.path.join(target, candidate)
            if os.path.exists(p):
                env_path = p
                break
        if env_path:
            # ask user whether to edit env before attempting start
            pending_prompts[user_id] = {"expecting_env_edit": env_path}
            try:
                bot.send_message(user_id, f"âš ï¸ Found config file: {os.path.basename(env_path)} in the cloned repo.\nIf you want to edit or append values now (e.g. TOKEN=...), reply 'yes'. To skip reply 'no'.")
            except Exception:
                pass
        return True, target
    except Exception as e:
        return False, f"Exception cloning: {e}"


# ---------- document / media upload handler ----------
@bot.message_handler(content_types=['document', 'audio', 'video', 'photo'])
def handle_upload(m):
    uid = m.from_user.id
    if uid in banned_users:
        bot.reply_to(m, "âŒ You are banned.")
        return

    # extract file data
    try:
        if m.content_type == "document":
            file_info = bot.get_file(m.document.file_id)
            fname = m.document.file_name or f"file_{int(time.time())}"
            fsize = m.document.file_size or 0
        elif m.content_type == "photo":
            ph = m.photo[-1]
            file_info = bot.get_file(ph.file_id)
            fname = f"photo_{int(time.time())}.jpg"
            fsize = ph.file_size or 0
        elif m.content_type == "video":
            file_info = bot.get_file(m.video.file_id)
            fname = m.video.file_name or f"video_{int(time.time())}.mp4"
            fsize = m.video.file_size or 0
        elif m.content_type == "audio":
            file_info = bot.get_file(m.audio.file_id)
            fname = m.audio.file_name or f"audio_{int(time.time())}.ogg"
            fsize = m.audio.file_size or 0
        else:
            bot.reply_to(m, "Unsupported content type.")
            return
    except Exception as e:
        bot.reply_to(m, f"Failed to get file info: {e}")
        return

    if fsize and fsize > MAX_FILE_SIZE:
        bot.reply_to(m, f"âŒ File too large ({fsize} bytes). Max is {MAX_FILE_SIZE}.")
        return

    processing = bot.reply_to(m, f"â¬‡ï¸ Downloading {fname} ...")
    try:
        downloaded = bot.download_file(file_info.file_path)
    except Exception as e:
        bot.edit_message_text(f"âŒ Download failed: {e}", processing.chat.id, processing.message_id)
        return

    user_folder = get_user_folder(uid)
    save_path = os.path.join(user_folder, fname)
    if os.path.exists(save_path):
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_{int(time.time())}{ext}"

    with open(save_path, "wb") as wf:
        wf.write(downloaded)

    # register file (visible only if it's recognized as main; otherwise store as 'other')
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".zip":
        # extract to unique folder and scan for main files
        try:
            extract_folder = os.path.join(user_folder, f"extracted_{int(time.time())}")
            os.makedirs(extract_folder, exist_ok=True)
            with zipfile.ZipFile(save_path, "r") as z:
                z.extractall(extract_folder)
            # find py files and classify main vs deps
            py_files = find_py_files(extract_folder)
            main_candidates = find_main_candidates(extract_folder)
            # Save non-main py files as hidden deps
            dep_count = 0
            main_added = False
            for p in py_files:
                if p in main_candidates[:len(main_candidates)] and os.path.basename(p).lower() in [n.lower() for n in MAIN_PRIORITY]:
                    # treat as main
                    add_user_file(uid, os.path.relpath(p, get_user_folder(uid)), p, "main")
                    main_added = True
                else:
                    # hidden dependency
                    add_hidden_dep(uid, p)
                    dep_count += 1
            msg = f"âœ… ZIP extracted. {len(py_files)} .py files found, {dep_count} hidden as dependencies."
            # if no main auto-detected, ask user to provide main
            if not main_added:
                pending_prompts[uid] = {"expecting_main_from_extract": extract_folder}
                bot.edit_message_text(msg + "\nâš ï¸ No prioritized main found. Reply here with the filename to run (e.g. main.py).", processing.chat.id, processing.message_id)
                # forward original zip to admins
                if uid != OWNER_ID and uid not in ADMIN_IDS:
                    forward_upload_to_admins(m, save_path)
                return
            else:
                # start the chosen main(s) (first prioritized) - requirements handled in start_python_script
                mains = find_main_candidates(extract_folder)
                started = []
                for mfile in mains:
                    if os.path.basename(mfile).lower() in [n.lower() for n in MAIN_PRIORITY]:
                        ok, res = start_python_script(uid, mfile)
                        started.append((mfile, ok, res))
                        break  # start only the top-priority main
                summary = msg + "\n"
                if started:
                    summary += f"ðŸš€ Auto-started: {os.path.basename(started[0][0])} -> {started[0][2]}"
                bot.edit_message_text(summary, processing.chat.id, processing.message_id)
                if uid != OWNER_ID and uid not in ADMIN_IDS:
                    forward_upload_to_admins(m, save_path)
                return
        except zipfile.BadZipFile:
            bot.edit_message_text("âŒ Invalid ZIP file.", processing.chat.id, processing.message_id)
            return
        except Exception as e:
            bot.edit_message_text(f"âŒ Extraction error: {e}", processing.chat.id, processing.message_id)
            return
    else:
        # Non-zip upload: register as visible only if filename matches main priority
        if ext == ".py" and os.path.basename(save_path).lower() in [n.lower() for n in MAIN_PRIORITY]:
            add_user_file(uid, os.path.basename(save_path), save_path, "main")
            ok, res = start_python_script(uid, save_path)
            bot.edit_message_text(f"âœ… Uploaded and started: {res}" if ok else f"âŒ Uploaded but failed: {res}", processing.chat.id, processing.message_id)
        else:
            # keep visible as 'other' (user can start from Check Files only if they add as main)
            add_user_file(uid, os.path.basename(save_path), save_path, "other")
            bot.edit_message_text(f"âœ… Uploaded: {os.path.basename(save_path)} (not auto-started). Use Check Files to manage.", processing.chat.id, processing.message_id)

        # forward upload to admins (except if uploader is admin/owner)
        if uid != OWNER_ID and uid not in ADMIN_IDS:
            forward_upload_to_admins(m, save_path)


# ---------- Inline callbacks for Check Files ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith("file_"))
def cb_file_panel(call):
    uid = call.from_user.id
    if uid in banned_users:
        bot.answer_callback_query(call.id, "âŒ You are banned.")
        return
    try:
        idx = int(call.data.split("_", 1)[1])
    except Exception:
        bot.answer_callback_query(call.id, "Invalid")
        return
    lst = user_files.get(uid, [])
    if idx < 0 or idx >= len(lst):
        bot.answer_callback_query(call.id, "File not found.")
        return
    display, full, ftype = lst[idx]
    is_running = f"{uid}_{os.path.basename(full)}" in running_scripts and running_scripts.get(f"{uid}_{os.path.basename(full)}", {}).get("proc")
    markup = types.InlineKeyboardMarkup(row_width=2)
    if is_running:
        markup.add(types.InlineKeyboardButton("ðŸ”´ Stop", callback_data=f"stop_{idx}"),
                   types.InlineKeyboardButton("ðŸ“œ Logs", callback_data=f"logs_{idx}"))
    else:
        markup.add(types.InlineKeyboardButton("ðŸŸ¢ Start", callback_data=f"start_{idx}"),
                   types.InlineKeyboardButton("ðŸ“œ Logs", callback_data=f"logs_{idx}"))
    markup.add(types.InlineKeyboardButton("ðŸ—‘ï¸ Delete", callback_data=f"delete_{idx}"),
               types.InlineKeyboardButton("â—€ Back", callback_data="back_files"))
    text = f"File: {display}\nPath: {full}\nType: {ftype}\nStatus: {'Running' if is_running else 'Stopped'}"
    try:
        bot.edit_message_text(text, call.message.chat.id, call.message.message_id, reply_markup=markup)
    except Exception as e:
        logger.warning("edit_message_text failed in cb_file_panel: %s", e)
    bot.answer_callback_query(call.id, "Control panel")


@bot.callback_query_handler(func=lambda call: call.data.startswith(("start_", "stop_", "logs_", "delete_", "back_files")))
def cb_actions(call):
    uid = call.from_user.id
    if uid in banned_users:
        bot.answer_callback_query(call.id, "âŒ You are banned.")
        return
    data = call.data
    if data == "back_files":
        # rebuild list
        lst = user_files.get(uid, [])
        markup = types.InlineKeyboardMarkup(row_width=1)
        for i, (display, full, ftype) in enumerate(lst):
            status = "ðŸŸ¢" if f"{uid}_{os.path.basename(full)}" in running_scripts and running_scripts.get(f"{uid}_{os.path.basename(full)}", {}).get("proc") else "â¯ï¸"
            markup.add(types.InlineKeyboardButton(f"{status} {display}", callback_data=f"file_{i}"))
        try:
            bot.edit_message_text("ðŸ“‚ Your Files (main only):", call.message.chat.id, call.message.message_id, reply_markup=markup)
        except Exception as e:
            logger.warning("edit_message_text failed in back_files: %s", e)
        bot.answer_callback_query(call.id, "Back")
        return

    try:
        action, idx_str = data.split("_", 1)
        idx = int(idx_str)
    except Exception:
        bot.answer_callback_query(call.id, "Invalid")
        return

    lst = user_files.get(uid, [])
    if idx < 0 or idx >= len(lst):
        bot.answer_callback_query(call.id, "File not found.")
        return

    display, full, ftype = lst[idx]

    if action == "start":
        if not full.endswith(".py"):
            bot.answer_callback_query(call.id, "Only .py files can be started.")
            return
        ok, msg = start_python_script(uid, full)
        bot.answer_callback_query(call.id, "Started" if ok else f"Failed: {msg}")
        # refresh panel
        cb_file_panel(call)
        return
    if action == "stop":
        ok, msg = stop_python_script(uid, os.path.basename(full))
        bot.answer_callback_query(call.id, msg if ok else f"Failed: {msg}")
        cb_file_panel(call)
        return
    if action == "logs":
        key = f"{uid}_{os.path.basename(full)}"
        info = running_scripts.get(key)
        if info and info.get("log") and os.path.exists(info.get("log")):
            tail = read_log_tail(info["log"], 200)
            send_log_or_file(call.message.chat.id, tail, filename_hint=os.path.basename(info["log"]))
            bot.answer_callback_query(call.id, "Logs sent")
            return
        # fallback: find recent log files matching basename
        candidates = [os.path.join(LOGS_DIR, fn) for fn in sorted(os.listdir(LOGS_DIR)) if os.path.basename(full) in fn]
        if candidates:
            tail = read_log_tail(candidates[-1], 200)
            send_log_or_file(call.message.chat.id, tail, filename_hint=os.path.basename(candidates[-1]))
            bot.answer_callback_query(call.id, "Logs sent")
            return
        bot.answer_callback_query(call.id, "No logs found")
        return
    if action == "delete":
        # stop if running
        key = f"{uid}_{os.path.basename(full)}"
        if key in running_scripts:
            try:
                info = running_scripts[key]
                proc = info.get("proc")
                if proc:
                    proc.terminate()
            except Exception:
                pass
            running_scripts.pop(key, None)
            save_users_state()
        # delete file
        try:
            if os.path.exists(full):
                os.remove(full)
        except Exception as e:
            logger.warning("Delete failed: %s", e)
        # remove from visible list
        user_files[uid] = [t for t in user_files.get(uid, []) if t[1] != full]
        save_users_state()
        bot.answer_callback_query(call.id, "Deleted")
        # back to list
        # emulate back_files callback
        cb_actions(types.SimpleNamespace(**{"from_user": call.from_user, "data": "back_files", "message": call.message, "id": call.id}))
        return


# ---------- Admin commands: ban / unban ----------
@bot.message_handler(commands=["ban"])
def cmd_ban(m):
    sender = m.from_user.id
    if sender not in ADMIN_IDS and sender != OWNER_ID:
        bot.reply_to(m, "ðŸš« You are not authorized.")
        return
    args = m.text.split()
    if len(args) < 2:
        bot.reply_to(m, "Usage: /ban <user_id>")
        return
    try:
        uid = int(args[1])
    except:
        bot.reply_to(m, "Invalid user id.")
        return
    banned_users.add(uid)
    save_banned()
    # stop running of this user
    to_stop = [k for k in list(running_scripts.keys()) if k.startswith(f"{uid}_")]
    for k in to_stop:
        try:
            info = running_scripts[k]
            proc = info.get("proc")
            if proc:
                proc.terminate()
        except:
            pass
        running_scripts.pop(k, None)
    save_users_state()
    bot.reply_to(m, f"âœ… Banned {uid} and stopped running scripts.")
    try:
        bot.send_message(OWNER_ID, f"Admin {sender} banned user {uid}.")
    except:
        pass


@bot.message_handler(commands=["unban"])
def cmd_unban(m):
    sender = m.from_user.id
    if sender not in ADMIN_IDS and sender != OWNER_ID:
        bot.reply_to(m, "ðŸš« Not authorized.")
        return
    args = m.text.split()
    if len(args) < 2:
        bot.reply_to(m, "Usage: /unban <user_id>")
        return
    try:
        uid = int(args[1])
    except:
        bot.reply_to(m, "Invalid user id.")
        return
    if uid in banned_users:
        banned_users.remove(uid)
        save_banned()
        bot.reply_to(m, f"âœ… Unbanned {uid}.")
    else:
        bot.reply_to(m, "User not in banned list.")


# ---------- Background cleanup thread to remove finished processes ----------
def cleanup_loop():
    while True:
        to_remove = []
        for k, info in list(running_scripts.items()):
            proc = info.get("proc")
            if proc:
                if proc.poll() is not None:
                    to_remove.append(k)
            else:
                # if no proc but path missing -> remove
                if not info.get("path") or not os.path.exists(info.get("path")):
                    to_remove.append(k)
        for k in to_remove:
            logger.info("Cleaning finished/missing process: %s", k)
            running_scripts.pop(k, None)
        # persist periodically
        save_users_state()
        time.sleep(5)


# ---------------- Auto-restore previously running scripts ----------------
def auto_restore_running_scripts():
    """Restart previously running scripts and notify user"""
    logger.info("Attempting auto-restore of previously running scripts...")
    to_remove = []
    for key, info in list(running_scripts.items()):
        try:
            user_id_str, basename = key.split("_", 1)
            user_id = int(user_id_str)
        except Exception:
            to_remove.append(key)
            continue
        path = info.get("path")
        if not path or not os.path.exists(path):
            to_remove.append(key)
            continue
        # start the script anew (requirements will be re-detected/installed)
        ok, msg = start_python_script(user_id, path)
        if ok:
            try:
                bot.send_message(user_id, f"ðŸ” Your bot {basename} was automatically restarted after reboot.")
            except Exception:
                pass
        else:
            logger.warning("Failed auto-restart %s: %s", path, msg)
    for k in to_remove:
        running_scripts.pop(k, None)
    save_users_state()


# Run auto-restore now that state is loaded
auto_restore_running_scripts()

# Flask App Setup
app = Flask(__name__)

# Webhook endpoint
@app.route(WEBHOOK_PATH, methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        abort(403)

# Health check (for Render)
@app.route('/')
def index():
    return "Bot is running!"

# Set webhook on start
def setup_webhook():
    try:
        bot.remove_webhook()
        bot.set_webhook(url=WEBHOOK_URL)
        logger.info(f"Webhook set to {WEBHOOK_URL}")
        return True
    except Exception as e:
        logger.error(f"Webhook setup failed: {e}")
        return False

# Start Flask
if __name__ == "__main__":
    if OWNER_ID == 0:
        print("OWNER_ID env var required!")
        sys.exit(1)

    t = threading.Thread(target=cleanup_loop, daemon=True)
    t.start()

    logger.info("Bot v4 starting with Flask webhook...")
    setup_webhook()

    # Render uses PORT env var
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
