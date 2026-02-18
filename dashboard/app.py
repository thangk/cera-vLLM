"""cera-vLLM Dashboard — Model management and monitoring for local LLM serving."""

import asyncio
import json
import os
import secrets
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import docker
import httpx
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import snapshot_download, HfApi
from itsdangerous import URLSafeTimedSerializer

from models_registry import RECOMMENDED_MODELS

# --- Configuration ---

DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")
VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", "cera-vllm")
DOWNLOAD_SPEED_LIMIT = os.environ.get("DOWNLOAD_SPEED_LIMIT", "")  # KB/s, e.g. "50000" for ~50 MB/s
if DOWNLOAD_SPEED_LIMIT:
    # Disable hf_transfer (Rust fast downloader) so urllib3 throttle works
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
CONFIG_DIR = Path("/config")
DATA_DIR = Path("/data")
DB_PATH = DATA_DIR / "dashboard.db"
SECRET_KEY = secrets.token_hex(32)
COOKIE_NAME = "cera_vllm_session"
SESSION_MAX_AGE = 86400  # 24 hours

serializer = URLSafeTimedSerializer(SECRET_KEY)

# Track active downloads: {model_id: {"progress": float, "status": str}}
download_tasks: dict[str, dict] = {}


# --- Database ---

async def init_db():
    """Initialize SQLite database with schema and seed recommended models."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                vram_gb INTEGER NOT NULL,
                speed TEXT NOT NULL DEFAULT '',
                quality TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                requires_hf_token INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'not_downloaded',
                download_progress REAL NOT NULL DEFAULT 0.0,
                is_recommended INTEGER NOT NULL DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        await db.commit()

        # Seed recommended models if not present
        for model in RECOMMENDED_MODELS:
            existing = await db.execute(
                "SELECT 1 FROM models WHERE model_id = ?", (model["model_id"],)
            )
            if not await existing.fetchone():
                await db.execute(
                    """INSERT INTO models
                       (model_id, display_name, vram_gb, speed, quality, description,
                        requires_hf_token, is_recommended)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                    (model["model_id"], model["display_name"], model["vram_gb"],
                     model["speed"], model["quality"], model["description"],
                     1 if model["requires_hf_token"] else 0),
                )
        await db.commit()

        # Generate API key on first boot
        existing_key = await db.execute(
            "SELECT value FROM config WHERE key = 'api_key'"
        )
        if not await existing_key.fetchone():
            api_key = f"cvllm-{secrets.token_hex(24)}"
            await db.execute(
                "INSERT INTO config (key, value) VALUES ('api_key', ?)", (api_key,)
            )
            await db.commit()


async def get_db():
    """Get database connection."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def get_config(db: aiosqlite.Connection, key: str, default: str = "") -> str:
    """Get a config value."""
    cursor = await db.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return row["value"] if row else default


async def set_config(db: aiosqlite.Connection, key: str, value: str):
    """Set a config value."""
    await db.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
    )
    await db.commit()


# --- Auth ---

def create_session_cookie(response: RedirectResponse) -> RedirectResponse:
    """Create a signed session cookie."""
    token = serializer.dumps({"authenticated": True})
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, max_age=SESSION_MAX_AGE, samesite="lax"
    )
    return response


def verify_session(request: Request) -> bool:
    """Verify the session cookie."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("authenticated", False)
    except Exception:
        return False


async def require_auth(request: Request):
    """Dependency that requires authentication."""
    if not verify_session(request):
        raise HTTPException(status_code=303, headers={"Location": "/login"})


# --- vLLM Control ---

def get_docker_client():
    """Get Docker client for container management."""
    return docker.from_env()


def restart_vllm_container():
    """Restart the vLLM container to pick up new model config."""
    try:
        client = get_docker_client()
        container = client.containers.get(VLLM_CONTAINER_NAME)
        container.restart(timeout=30)
        return True
    except Exception as e:
        print(f"[Dashboard] Failed to restart vLLM container: {e}")
        return False


def write_model_config(model_id: str, api_key: str):
    """Write the active model config for vLLM to read."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_id,
        "api_key": api_key,
        "extra_args": VLLM_EXTRA_ARGS,
    }
    config_path = CONFIG_DIR / "active_model.json"
    config_path.write_text(json.dumps(config, indent=2))


async def _get_vllm_api_key() -> str:
    """Read the API key from DB for authenticating with vLLM."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            return await get_config(db, "api_key")
    except Exception:
        return ""


async def get_vllm_status() -> dict:
    """Check vLLM server health and get loaded models."""
    try:
        api_key = await _get_vllm_api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("http://cera-vllm:8000/v1/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                return {"status": "running", "models": models}
            return {"status": "error", "models": [], "error": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"status": "waiting", "models": [], "error": "Waiting for vLLM to start..."}
    except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
        # Timeout under heavy load — vLLM is still running, just busy
        return {"status": "running", "models": [], "error": "Server busy (timeout)"}
    except Exception as e:
        return {"status": "error", "models": [], "error": str(e)}


# State for computing tokens/sec from counter deltas
_prev_gen_tokens: float = 0.0
_prev_gen_time: float = 0.0


async def get_vllm_metrics() -> dict:
    """Fetch Prometheus metrics from vLLM and compute throughput."""
    global _prev_gen_tokens, _prev_gen_time
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # /metrics endpoint doesn't require auth in vLLM
            resp = await client.get("http://cera-vllm:8000/metrics")
            if resp.status_code != 200:
                return {}
            text = resp.text
            metrics = {}
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                # Parse simple Prometheus format: metric_name{labels} value
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split("{")[0]
                    try:
                        metrics[name] = float(parts[-1])
                    except ValueError:
                        pass

            # Compute tokens/sec from generation_tokens_total delta
            now = time.time()
            gen_tokens = metrics.get("vllm:generation_tokens_total", 0)
            if _prev_gen_time > 0 and now > _prev_gen_time:
                dt = now - _prev_gen_time
                delta = gen_tokens - _prev_gen_tokens
                metrics["_computed_throughput_tps"] = max(0, delta / dt)
            else:
                metrics["_computed_throughput_tps"] = 0
            _prev_gen_tokens = gen_tokens
            _prev_gen_time = now

            return metrics
    except Exception:
        return {}


def get_gpu_info() -> list[dict]:
    """Get GPU utilization and VRAM usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                })
        return gpus
    except Exception:
        return []


# --- Model Download ---

def get_dir_size(path: Path) -> int:
    """Get total size of all files in a directory tree."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def download_model_sync(model_id: str, db_path: str):
    """Download a model from HuggingFace (runs in background thread)."""
    import sqlite3
    import time

    def update_status(status: str, progress: float = 0.0, speed_mbps: float = 0.0):
        download_tasks[model_id] = {
            "status": status, "progress": progress, "speed_mbps": speed_mbps,
        }
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE models SET status = ?, download_progress = ? WHERE model_id = ?",
            (status, progress, model_id),
        )
        conn.commit()
        conn.close()

    try:
        update_status("downloading", 0.0)

        # Fetch total expected size from model metadata
        hf = HfApi()
        try:
            info = hf.model_info(model_id, files_metadata=True, token=HF_TOKEN)
            total_bytes = sum(s.size for s in (info.siblings or []) if s.size) or 0
        except Exception:
            total_bytes = 0
        print(f"[Dashboard] Model {model_id}: total size ~{total_bytes / 1e9:.1f} GB")

        # HF cache path: ~/.cache/huggingface/hub/models--org--model/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"

        # Apply speed throttle if configured
        _orig_read = None
        if DOWNLOAD_SPEED_LIMIT:
            import urllib3.response
            _orig_read = urllib3.response.HTTPResponse.read
            speed_bytes_per_sec = int(DOWNLOAD_SPEED_LIMIT) * 1024

            def _throttled_read(self, amt=None, **kwargs):
                chunk_size = min(amt or 65536, 65536)
                start = time.monotonic()
                data = _orig_read(self, amt=chunk_size, **kwargs)
                if data and speed_bytes_per_sec > 0:
                    expected = len(data) / speed_bytes_per_sec
                    elapsed = time.monotonic() - start
                    if elapsed < expected:
                        time.sleep(expected - elapsed)
                return data

            urllib3.response.HTTPResponse.read = _throttled_read
            print(f"[Dashboard] Speed limit: {DOWNLOAD_SPEED_LIMIT} KB/s")

        # Run snapshot_download in a sub-thread so we can monitor progress
        download_error = [None]
        download_done = threading.Event()

        def _do_download():
            try:
                snapshot_download(
                    model_id,
                    token=HF_TOKEN,
                    max_workers=1 if DOWNLOAD_SPEED_LIMIT else 4,
                )
            except Exception as e:
                download_error[0] = e
            finally:
                download_done.set()

        dl_thread = threading.Thread(target=_do_download, daemon=True)
        dl_thread.start()

        # Monitor progress by watching cache directory size
        last_bytes = get_dir_size(cache_dir) if cache_dir.exists() else 0
        last_time = time.monotonic()

        while not download_done.is_set():
            download_done.wait(timeout=3)
            if download_done.is_set():
                break

            current_bytes = get_dir_size(cache_dir) if cache_dir.exists() else 0
            now = time.monotonic()

            # Progress percentage
            if total_bytes > 0:
                progress = min(current_bytes / total_bytes * 100, 99.9)
            else:
                progress = 0.0

            # Speed (MB/s)
            dt = now - last_time
            speed_mbps = ((current_bytes - last_bytes) / dt / (1024 * 1024)) if dt > 0 else 0.0

            last_bytes = current_bytes
            last_time = now

            update_status("downloading", round(progress, 1), round(speed_mbps, 1))

        # Restore urllib3 if patched
        if _orig_read is not None:
            import urllib3.response
            urllib3.response.HTTPResponse.read = _orig_read

        if download_error[0]:
            raise download_error[0]

        update_status("downloaded", 100.0)
        print(f"[Dashboard] Download complete: {model_id}")

    except Exception as e:
        print(f"[Dashboard] Download failed for {model_id}: {e}")
        if _orig_read is not None:
            try:
                import urllib3.response
                urllib3.response.HTTPResponse.read = _orig_read
            except Exception:
                pass
        update_status("not_downloaded", 0.0)
        download_tasks[model_id] = {
            "status": "error", "progress": 0.0, "speed_mbps": 0.0, "error": str(e),
        }


# --- App Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield


# --- FastAPI App ---

app = FastAPI(title="cera-vLLM Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    if verify_session(request):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    """Handle login form submission."""
    if password == DASHBOARD_PASSWORD:
        response = RedirectResponse("/", status_code=303)
        return create_session_cookie(response)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid password"}
    )


@app.get("/logout")
async def logout():
    """Log out and clear session."""
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    if not verify_session(request):
        return RedirectResponse("/login", status_code=303)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        api_key = await get_config(db, "api_key")
        active_model = await get_config(db, "active_model")
        cursor = await db.execute("SELECT * FROM models ORDER BY is_recommended DESC, display_name")
        models = [dict(row) for row in await cursor.fetchall()]

    # Merge in-memory download progress
    for model in models:
        if model["model_id"] in download_tasks:
            task = download_tasks[model["model_id"]]
            model["status"] = task["status"]
            model["download_progress"] = task.get("progress", 0.0)
            if "error" in task:
                model["download_error"] = task["error"]

    vllm_status = await get_vllm_status()
    gpus = get_gpu_info()

    # Get external IP for CERA config
    external_port = os.environ.get("VLLM_PORT", "8100")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_key": api_key,
        "active_model": active_model,
        "models": models,
        "vllm_status": vllm_status,
        "gpus": gpus,
        "external_port": external_port,
        "hf_token_set": bool(HF_TOKEN),
    })


# --- API Endpoints ---

@app.post("/api/download/{model_id:path}")
async def start_download(model_id: str, request: Request):
    """Start downloading a model."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    if model_id in download_tasks and download_tasks[model_id]["status"] == "downloading":
        return JSONResponse({"error": "Download already in progress"}, status_code=409)

    # Start download in background thread
    thread = threading.Thread(
        target=download_model_sync,
        args=(model_id, str(DB_PATH)),
        daemon=True,
    )
    thread.start()
    return JSONResponse({"status": "started"})


@app.post("/api/activate/{model_id:path}")
async def activate_model(model_id: str, request: Request):
    """Activate a model (write config and restart vLLM)."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # Check model is downloaded
        cursor = await db.execute(
            "SELECT status FROM models WHERE model_id = ?", (model_id,)
        )
        row = await cursor.fetchone()
        if not row or row["status"] not in ("downloaded", "active", "loading"):
            return JSONResponse({"error": "Model not downloaded"}, status_code=400)

        api_key = await get_config(db, "api_key")

        # Clear previous active model
        await db.execute(
            "UPDATE models SET status = 'downloaded' WHERE status IN ('active', 'loading')"
        )
        # Set new model to loading (promoted to active once vLLM confirms)
        await db.execute(
            "UPDATE models SET status = 'loading' WHERE model_id = ?", (model_id,)
        )
        await set_config(db, "active_model", model_id)

    # Write config for vLLM and restart
    write_model_config(model_id, api_key)
    success = restart_vllm_container()

    return JSONResponse({
        "status": "activating" if success else "error",
        "message": "vLLM restarting with new model..." if success else "Failed to restart vLLM container",
    })


@app.post("/api/add-custom")
async def add_custom_model(request: Request):
    """Add a custom model by HuggingFace ID."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    data = await request.json()
    model_id = data.get("model_id", "").strip()
    if not model_id or "/" not in model_id:
        return JSONResponse({"error": "Invalid model ID (expected format: org/model)"}, status_code=400)

    async with aiosqlite.connect(DB_PATH) as db:
        existing = await db.execute(
            "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
        )
        if await existing.fetchone():
            return JSONResponse({"error": "Model already exists"}, status_code=409)

        display_name = model_id.split("/")[-1]
        await db.execute(
            """INSERT INTO models (model_id, display_name, vram_gb, speed, quality, description, is_recommended)
               VALUES (?, ?, 0, 'Unknown', 'Unknown', 'Custom model', 0)""",
            (model_id, display_name),
        )
        await db.commit()

    return JSONResponse({"status": "added"})


@app.get("/api/search-models")
async def search_hf_models(request: Request, q: str = ""):
    """Search HuggingFace Hub for text-generation models."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    if not q or len(q) < 2:
        return JSONResponse({"results": []})

    try:
        hf = HfApi()
        models = hf.list_models(
            search=q,
            pipeline_tag="text-generation",
            sort="downloads",
            direction=-1,
            limit=10,
        )
        results = []
        for m in models:
            # Estimate size from safetensors parameter count (2 bytes/param for BF16)
            size_bytes = None
            params = None
            if getattr(m, "safetensors", None) and isinstance(m.safetensors, dict):
                params = m.safetensors.get("total", None)
                if params:
                    size_bytes = params * 2  # BF16/FP16 = 2 bytes per param
            results.append({
                "id": m.id,
                "downloads": m.downloads or 0,
                "likes": m.likes or 0,
                "size_bytes": size_bytes,
                "params": params,
            })
        return JSONResponse({"results": results})
    except Exception as e:
        return JSONResponse({"results": [], "error": str(e)})


@app.post("/api/regenerate-key")
async def regenerate_api_key(request: Request):
    """Regenerate the API key and restart vLLM."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    new_key = f"cvllm-{secrets.token_hex(24)}"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await set_config(db, "api_key", new_key)
        active_model = await get_config(db, "active_model")

    # Rewrite config with new key and restart if model is active
    if active_model:
        write_model_config(active_model, new_key)
        restart_vllm_container()

    return JSONResponse({"api_key": new_key})


@app.get("/api/status")
async def api_status(request: Request):
    """Get live status (polled by dashboard JS)."""
    if not verify_session(request):
        raise HTTPException(status_code=401)

    vllm_status = await get_vllm_status()
    metrics = await get_vllm_metrics()
    gpus = get_gpu_info()

    # Get model download statuses
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT model_id, status, download_progress FROM models")
        model_statuses = {row["model_id"]: dict(row) for row in await cursor.fetchall()}

        # Auto-promote loading → active when vLLM confirms the model is serving
        served_models = vllm_status.get("models", [])
        for model_id, info in model_statuses.items():
            if info["status"] == "loading" and model_id in served_models:
                await db.execute(
                    "UPDATE models SET status = 'active' WHERE model_id = ?",
                    (model_id,),
                )
                await db.commit()
                model_statuses[model_id]["status"] = "active"

    # Merge in-memory download progress (includes speed)
    for model_id, task in download_tasks.items():
        if model_id in model_statuses:
            model_statuses[model_id]["status"] = task["status"]
            model_statuses[model_id]["download_progress"] = task.get("progress", 0.0)
            model_statuses[model_id]["speed_mbps"] = task.get("speed_mbps", 0.0)

    return JSONResponse({
        "vllm": vllm_status,
        "metrics": {
            "throughput_tps": metrics.get("_computed_throughput_tps", 0),
            "running_requests": metrics.get("vllm:num_requests_running", 0),
            "waiting_requests": metrics.get("vllm:num_requests_waiting", 0),
            "prompt_tokens_total": metrics.get("vllm:prompt_tokens_total", 0),
            "generation_tokens_total": metrics.get("vllm:generation_tokens_total", 0),
        },
        "gpus": gpus,
        "models": model_statuses,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
