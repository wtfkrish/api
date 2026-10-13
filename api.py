# api.py
import time
import logging
import threading
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from html import escape

from config import (
    API_HOST,
    API_PORT,
    APP_VERSION,
    APP_MIN_VERSION,
    APP_BUILD,
    MASTER_USER_IDS,
    API_DOCS_ENABLED,
    API_CORS_ORIGINS,
    LOGIN_RATE_LIMIT_COUNT,
    LOGIN_RATE_LIMIT_WINDOW_SECONDS,
    ANDROID_CLIENT_ID,
    ANDROID_REQUIRED_BUILD,
)
from auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    get_optional_user,
    validate_key_format,
    require_supported_app,
)
from database import (
    init_db,
    save_user,
    get_user,
    get_user_key_info,
    get_key_info,
    get_user_subscription_full,
    refresh_key_status,
    redeem_key,
    get_binding_by_key,
    get_binding_by_user,
    get_device_id_for_user,
    bind_key_to_device,
    register_device,
    verify_device,
    is_device_verified,
    revoke_device,
    get_app_version_config,
    log_audit,
    save_refresh_token,
    get_refresh_token,
    delete_refresh_token,
    is_user_banned,
    is_admin,
    get_new_user_id,
    delete_user_if_empty,
    get_session,
    update_session_status,
    get_session_by_slot,
    get_user_auth_version,
    bump_user_auth_version,
    redeem_key_and_bind_device,
    get_all_non_terminal_sessions,
    get_max_concurrency,
)
from session_manager import (
    get_instance,
    SessionError,
    SessionUserActive,
    SessionCapacityError,
    SessionGroupLimit,
    SessionInternalError,
    SessionNotFound,
    SessionAlreadyEnded,
    SessionPermissionError,
    SessionTransportError,
)
from helpers import validate_ip, validate_port
from state import state
from display_identity import get_display_identity
from utils import format_ist

logger = logging.getLogger(__name__)
_LOGIN_ATTEMPTS: dict = {}

_BOT_APP = None
_BOT_LOOP = None
_BOT_SEND_TIMEOUT = 15


def configure_bot_bridge(application, loop) -> None:
    global _BOT_APP, _BOT_LOOP
    _BOT_APP = application
    _BOT_LOOP = loop


def _send_owner_message_from_api(text: str) -> None:
    """Send an owner notification through the already-running Telegram bot."""
    if _BOT_APP is None or _BOT_LOOP is None or _BOT_LOOP.is_closed():
        raise RuntimeError("Telegram bot bridge is not ready")

    for owner_id in MASTER_USER_IDS:
        future = asyncio.run_coroutine_threadsafe(
            _BOT_APP.bot.send_message(
                chat_id=int(owner_id),
                text=text,
                parse_mode="HTML",
            ),
            _BOT_LOOP,
        )
        future.result(timeout=_BOT_SEND_TIMEOUT)


async def _notify_owner_of_apk_redemption(
    user_id: int,
    key: str,
    expires_at: int,
    plan: str | None,
    device_name: str,
    device_id: str,
) -> None:
    """Build and deliver the same-style redemption alert used by /redeem."""
    identity = await get_display_identity(int(user_id))
    readable_name = str(identity.get("readable_name") or device_name or "Unknown device")
    resolved_device_id = str(identity.get("device_id") or device_id or "Not registered")
    plan_text = str(plan or "NORMAL").upper()
    identity_lines = (
        f"📱 Device · {escape(readable_name)}\n\n"
        f"🪪 Device ID · <code>{escape(resolved_device_id)}</code>"
    )
    owner_notification = (
        "🎉 NEW REDEMPTION!\n\n"
        "💎 KEY REDEEMED SUCCESSFULLY\n\n"
        f"{identity_lines}\n\n"
        f"🔐 Key · <code>{escape(key)}</code>\n\n"
        "🟢 Status · ACTIVATED\n\n"
        f"💎 Plan · {escape(plan_text)}\n\n"
        f"📅 Expires · {escape(format_ist(expires_at))}"
    )

    try:
        await asyncio.to_thread(_send_owner_message_from_api, owner_notification)
    except Exception:
        logger.exception("Could not notify owner about APK key redemption: %s", key)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_login_rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.monotonic()
    hits = [t for t in _LOGIN_ATTEMPTS.get(key, []) if now - t < LOGIN_RATE_LIMIT_WINDOW_SECONDS]
    stale_keys = [
        k for k, values in _LOGIN_ATTEMPTS.items()
        if not any(now - t < LOGIN_RATE_LIMIT_WINDOW_SECONDS for t in values)
    ]
    for stale in stale_keys:
        _LOGIN_ATTEMPTS.pop(stale, None)
    if len(hits) >= LOGIN_RATE_LIMIT_COUNT:
        raise HTTPException(
            status_code=429,
            detail=error_response("RATE_LIMITED", "Too many login attempts. Try again later.")["error"],
        )
    hits.append(now)
    _LOGIN_ATTEMPTS[key] = hits


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ZEUS API",
    version=str(APP_VERSION),
    docs_url="/docs" if API_DOCS_ENABLED else None,
    redoc_url="/redoc" if API_DOCS_ENABLED else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-ZEUS-APP", "X-ZEUS-BUILD"],
)


def _safe_body_preview(body) -> str:
    """Return a short, safe preview of a validation-failed request body."""
    try:
        if body is None:
            return "<empty>"
        if isinstance(body, (bytes, bytearray)):
            text = body.decode("utf-8", errors="replace")
        else:
            text = str(body)
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return text[:500] if text else "<empty>"
    except Exception:
        return "<unreadable>"


@app.exception_handler(RequestValidationError)
async def api_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Emit the project's standard error envelope for every 422.

    FastAPI's default body is ``{"detail": [...]}``, which the mobile client
    cannot parse with the same contract as every other endpoint. This handler
    returns the same shape the app already understands, includes the raw
    pydantic errors under ``errors`` for debugging, and logs enough context
    that the next 422 tells us exactly which field the client failed to send
    correctly.
    """
    try:
        raw_errors = exc.errors()
    except Exception:
        raw_errors = []

    serializable_errors = []
    for err in raw_errors:
        if not isinstance(err, dict):
            continue
        item = {
            "loc": [str(part) for part in (err.get("loc") or [])],
            "type": str(err.get("type") or ""),
            "msg": str(err.get("msg") or ""),
        }
        if "input" in err:
            try:
                item["input"] = str(err.get("input"))[:200]
            except Exception:
                item["input"] = "<unrepresentable>"
        serializable_errors.append(item)

    body_preview = _safe_body_preview(getattr(exc, "body", None))
    logger.warning(
        "Request validation failed method=%s path=%s errors=%s body=%r",
        request.method,
        request.url.path,
        serializable_errors,
        body_preview,
    )

    message = "Request validation failed."
    if serializable_errors:
        first = serializable_errors[0]
        loc = ".".join(p for p in first["loc"] if p not in {"body", "__root__"}) or "body"
        message = f"Invalid request: {loc} — {first['msg']}"

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "code": "VALIDATION_ERROR",
            "message": message,
            "error": {"code": "VALIDATION_ERROR", "message": message},
            "detail": {"code": "VALIDATION_ERROR", "message": message},
            "errors": serializable_errors,
        },
    )


@app.exception_handler(HTTPException)
async def api_http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail

    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        code = str(detail["code"])
        message = str(detail["message"])
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "code": code,
                "message": message,
                "error": {"code": code, "message": message},
                "detail": {"code": code, "message": message},
            },
        )

    message = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "code": "HTTP_ERROR",
            "message": message,
            "error": {"code": "HTTP_ERROR", "message": message},
            "detail": message,
        },
    )


# ============================================================
# HELPERS
# ============================================================

def now_ts() -> int:
    return int(time.time())


def iso_timestamp(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    return (
        datetime.fromtimestamp(int(ts), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def success_response(data=None, message="Success"):
    return {
        "success": True,
        "message": message,
        "data": data or {},
    }


def error_response(code: str, message: str):
    return {
        "success": False,
        "error": {"code": code, "message": message},
    }


def http_error(status: int, code: str, message: str):
    raise HTTPException(
        status_code=status,
        detail=error_response(code, message)["error"],
    )


def is_master(user_id: int) -> bool:
    return int(user_id) in MASTER_USER_IDS


async def require_subscription(user_id: int):
    if is_master(user_id):
        return {
            "active": True,
            "status": "ACTIVE",
            "key": None,
            "plan": "SUPREME",
            "duration_minutes": None,
            "duration_label": "Unlimited",
            "price": None,
            "expires_at": None,
            "expires_at_iso": None,
            "remaining_seconds": 999999999,
            "remaining_minutes": 999999999,
            "timer_seconds": 999999999,
            "cooldown_seconds": 0,
        }

    if await is_user_banned(user_id):
        http_error(403, "USER_BANNED", "User is banned.")

    sub = await get_user_subscription_full(user_id)
    if not sub:
        http_error(403, "NO_ACTIVE_SUBSCRIPTION", "No active subscription found.")

    expiry = int(sub["expires_at"])
    remaining = max(0, expiry - now_ts())
    if remaining <= 0:
        http_error(403, "SUBSCRIPTION_EXPIRED", "Subscription has expired.")

    return {
        "active": True,
        "status": "ACTIVE",
        "key": sub.get("key"),
        "plan": sub.get("plan"),
        "duration_minutes": int(sub.get("duration_minutes") or 0),
        "duration_label": sub.get("duration_label"),
        "price": sub.get("price"),
        "expires_at": expiry,
        "expires_at_iso": iso_timestamp(expiry),
        "remaining_seconds": remaining,
        "remaining_minutes": remaining // 60,
        "timer_seconds": int(sub.get("timer_seconds") or 0),
        "cooldown_seconds": int(sub.get("cooldown_seconds") or 0),
    }


# ============================================================
# MODELS
# ============================================================

class LoginRequest(BaseModel):
    key: str = Field(..., min_length=6, max_length=30)
    device_id: str = Field(..., min_length=1, max_length=255)
    device_name: str = Field(default="Unknown device", min_length=1, max_length=255)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=255)
    user_id: int


class DeviceVerifyRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=255)
    user_id: int


class DeviceRevokeRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=255)
    user_id: int


class SessionStartRequest(BaseModel):
    ip: str = Field(..., min_length=2, max_length=45)
    port: int = Field(..., ge=1, le=65535)
    duration: int = Field(..., ge=1)

    class Config:
        extra = "ignore"


class SessionStopRequest(BaseModel):
    slot_id: int = Field(..., ge=1)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():
    logger.info("ZEUS API startup complete (database initialized by main process).")


# ============================================================
# HEALTH / VERSION
# ============================================================

@app.get("/health")
async def health():
    return success_response({
        "status": "ok",
        "service": "zeus-api",
        "version": APP_VERSION,
        "build": APP_BUILD,
        "timestamp": now_ts(),
    })


@app.get("/app/version")
async def app_version(
    current_build: Optional[int] = Query(None, ge=0),
    current_user: Optional[int] = Depends(get_optional_user),
):
    cfg = await get_app_version_config()
    latest_build = int(cfg.get("app_latest_build") or APP_BUILD)
    min_build = int(cfg.get("app_min_build") or APP_BUILD)
    return success_response({
        "version": str(APP_VERSION),
        "minimum_version": str(cfg.get("app_min_version") or APP_MIN_VERSION),
        "latest_version": str(cfg.get("app_latest_version") or APP_VERSION),
        "build": APP_BUILD,
        "latest_build": latest_build,
        "minimum_build": min_build,
        "update_required": bool(current_build is not None and current_build < min_build),
        "mandatory_update": bool(cfg.get("app_mandatory_update", False)),
        "download_url": cfg.get("app_download_url") or None,
        "sha256": cfg.get("app_sha256") or None,
        "release_notes": cfg.get("app_release_notes") or None,
    })


# ============================================================
# AUTH
# ============================================================

@app.post("/auth/login")
async def login(request: Request, req: LoginRequest):
    require_supported_app(request)
    _check_login_rate_limit(request)
    key = req.key.strip().upper()
    device_id = req.device_id.strip()
    device_name = req.device_name.strip() or "Unknown device"

    if not validate_key_format(key):
        http_error(422, "INVALID_KEY_FORMAT", "Invalid key format.")

    status = await refresh_key_status(key)
    if status == "NOT_FOUND":
        http_error(404, "KEY_NOT_FOUND", "Key not found.")
    if status == "REVOKED":
        http_error(403, "KEY_REVOKED", "Key has been revoked.")
    if status == "EXPIRED":
        http_error(403, "KEY_EXPIRED", "Key has expired.")

    key_info = await get_key_info(key)
    if not key_info:
        http_error(404, "KEY_NOT_FOUND", "Key not found.")
    plan = str(key_info.get("plan") or "").upper().strip()
    is_trial = int(key_info.get("is_trial", 0) or 0)
    if plan == "NORMAL":
        http_error(403, "BOT_KEY_REQUIRED", "This is a Bot key. Redeem it in the Telegram bot.")
    if plan not in {"", "PREMIUM"} or is_trial:
        http_error(403, "APK_KEY_REQUIRED", "Only APK keys or manual keys can be used in the APK.")

    binding = await get_binding_by_key(key)

    if binding:
        user_id = int(binding["user_id"])
        bound_device = binding["bound_device_id"]
        if bound_device != device_id:
            http_error(409, "DEVICE_MISMATCH", "This key is already bound to another device.")
    else:
        existing_owner = key_info.get("user_id")
        newly_redeemed = False

        if existing_owner is not None:
            user_id = int(existing_owner)
            if await is_user_banned(user_id):
                http_error(403, "USER_BANNED", "User is banned.")
            existing_device = await get_device_id_for_user(user_id)
            if existing_device and existing_device != device_id:
                http_error(
                    409,
                    "DEVICE_MISMATCH",
                    "This subscription is already bound to another device.",
                )
        else:
            user_id = await get_new_user_id()
            redeemed = await redeem_key_and_bind_device(
                key, user_id, device_id, device_name
            )
            if not redeemed.get("success"):
                await delete_user_if_empty(user_id)
                redeem_error = str(redeemed.get("error") or "Unable to redeem key.")
                redeem_code = "KEY_REDEEM_FAILED"
                redeem_status = 409
                redeem_message = redeem_error

                if redeem_error == "Key already used":
                    redeem_code = "KEY_ALREADY_USED"
                    redeem_status = 409
                    redeem_message = "This key has already been used."
                elif redeem_error == "Key not found or revoked":
                    redeem_code = "KEY_NOT_FOUND"
                    redeem_status = 404
                    redeem_message = "This key is invalid or no longer available."
                elif redeem_error == "Key expired":
                    redeem_code = "KEY_EXPIRED"
                    redeem_status = 403
                    redeem_message = "This key has expired."
                elif redeem_error == "Key is already bound to another user":
                    redeem_code = "DEVICE_MISMATCH"
                    redeem_status = 409
                    redeem_message = "This key is already bound to another device."
                elif redeem_error.startswith("Active subscription already exists"):
                    redeem_code = "ACTIVE_SUBSCRIPTION_EXISTS"
                    redeem_status = 409
                    redeem_message = "An active subscription already exists for this account."

                http_error(redeem_status, redeem_code, redeem_message)

            await save_user(user_id)
            newly_redeemed = True

        if not newly_redeemed:
            await bind_key_to_device(key, user_id, device_id, device_name)

        if newly_redeemed:
            await log_audit(
                user_id,
                "apk_login_key_redeemed",
                key,
                f"device_id={device_id[:64]} | device_name={device_name[:100]}",
            )
            await _notify_owner_of_apk_redemption(
                user_id=user_id,
                key=key,
                expires_at=int(redeemed["expires_at"]),
                plan=redeemed.get("plan"),
                device_name=device_name,
                device_id=device_id,
            )

    if await is_user_banned(user_id):
        http_error(403, "USER_BANNED", "User is banned.")

    sub = await require_subscription(user_id)  # noqa: F841 — side effect only

    auth_version = await get_user_auth_version(user_id)
    access_token = create_access_token({
        "sub": str(user_id),
        "ver": auth_version,
        "client": ANDROID_CLIENT_ID,
        "build": ANDROID_REQUIRED_BUILD,
    })
    refresh_token = create_refresh_token({
        "sub": str(user_id),
        "ver": auth_version,
        "client": ANDROID_CLIENT_ID,
        "build": ANDROID_REQUIRED_BUILD,
    })

    refresh_payload = decode_token(refresh_token)
    refresh_exp = int(refresh_payload["exp"])

    await save_refresh_token(user_id, refresh_token, refresh_exp)

    access_payload = decode_token(access_token)
    expires_in = int(access_payload["exp"]) - now_ts()

    return {
        "success": True,
        "code": "LOGIN_SUCCESS",
        "message": "Login successful",
        "user_id": user_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
    }


@app.post("/auth/refresh")
async def refresh(request: Request, req: RefreshRequest):
    require_supported_app(request)
    try:
        payload = decode_token(req.refresh_token)
    except HTTPException:
        http_error(401, "INVALID_REFRESH_TOKEN", "Invalid refresh token.")

    if payload.get("type") != "refresh":
        http_error(401, "INVALID_TOKEN_TYPE", "Refresh token required.")
    if payload.get("client") != ANDROID_CLIENT_ID:
        http_error(401, "INVALID_REFRESH_TOKEN", "Refresh token is revoked.")
    try:
        if int(payload.get("build", 0)) != ANDROID_REQUIRED_BUILD:
            http_error(401, "INVALID_REFRESH_TOKEN", "Refresh token is revoked.")
    except (TypeError, ValueError):
        http_error(401, "INVALID_REFRESH_TOKEN", "Refresh token is revoked.")

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        http_error(401, "INVALID_USER", "Invalid user in token.")

    stored = await get_refresh_token(req.refresh_token)
    if not stored:
        http_error(401, "REFRESH_TOKEN_REVOKED", "Refresh token is revoked.")

    if int(stored["expires_at"]) <= now_ts():
        await delete_refresh_token(req.refresh_token)
        http_error(401, "REFRESH_TOKEN_EXPIRED", "Refresh token has expired.")

    if await is_user_banned(user_id):
        http_error(403, "USER_BANNED", "User is banned.")

    current_version = await get_user_auth_version(user_id)
    if payload.get("ver") is not None:
        try:
            if int(payload.get("ver")) != current_version:
                http_error(401, "REFRESH_TOKEN_REVOKED", "Refresh token is revoked.")
        except (TypeError, ValueError):
            http_error(401, "INVALID_REFRESH_TOKEN", "Invalid refresh token.")
    elif current_version != 0:
        http_error(401, "REFRESH_TOKEN_REVOKED", "Refresh token is revoked.")

    await require_subscription(user_id)

    access_token = create_access_token({
        "sub": str(user_id),
        "ver": current_version,
        "client": ANDROID_CLIENT_ID,
        "build": ANDROID_REQUIRED_BUILD,
    })
    access_payload = decode_token(access_token)
    expires_in = int(access_payload["exp"]) - now_ts()

    return {
        "success": True,
        "message": "Token refreshed",
        "access_token": access_token,
        "expires_in": expires_in,
    }


# ============================================================
# SUBSCRIPTION
# ============================================================

@app.get("/subscription/me")
async def subscription_me(current_user: int = Depends(get_current_user)):
    sub = await require_subscription(current_user)
    return {
        "success": True,
        "message": "Subscription details fetched",
        "user_id": current_user,
        "active": sub.get("active", False),
        "time_left_seconds": sub.get("remaining_seconds", 0),
        "status": sub.get("status", "INACTIVE"),
        "expires_at": sub.get("expires_at"),
        "expires_at_iso": sub.get("expires_at_iso"),
        "plan": sub.get("plan"),
        "duration_minutes": sub.get("duration_minutes"),
        "duration_label": sub.get("duration_label"),
        "timer_seconds": sub.get("timer_seconds", 0),
        "cooldown_seconds": sub.get("cooldown_seconds", 0),
    }


# ============================================================
# DEVICE
# ============================================================

@app.post("/device/register")
async def device_register(
    req: DeviceRegisterRequest,
    current_user: int = Depends(get_current_user),
):
    await require_subscription(current_user)

    if req.user_id != current_user:
        http_error(403, "USER_MISMATCH", "user_id does not match token.")

    binding = await get_binding_by_user(current_user)
    if binding and binding.get("bound_device_id") != req.device_id:
        http_error(
            409,
            "DEVICE_MISMATCH",
            "Your subscription is already bound to another device.",
        )

    await register_device(current_user, req.device_id)

    return success_response({
        "user_id": current_user,
        "device_id": req.device_id,
        "registered": True,
        "verified": False,
    })


@app.post("/device/verify")
async def device_verify(
    req: DeviceVerifyRequest,
    current_user: int = Depends(get_current_user),
):
    await require_subscription(current_user)

    if req.user_id != current_user:
        http_error(403, "USER_MISMATCH", "user_id does not match token.")

    verified = await verify_device(current_user, req.device_id)
    return success_response({
        "user_id": current_user,
        "device_id": req.device_id,
        "verified": bool(verified),
    })


@app.get("/device/verify")
async def device_verify_get(
    device_id: str = Query(..., min_length=1),
    user_id: int = Query(...),
    current_user: int = Depends(get_current_user),
):
    await require_subscription(current_user)

    if user_id != current_user:
        http_error(403, "USER_MISMATCH", "user_id does not match token.")

    verified = await is_device_verified(current_user, device_id)
    return success_response({
        "user_id": current_user,
        "device_id": device_id,
        "verified": bool(verified),
    })


@app.post("/device/revoke")
async def device_revoke(
    req: DeviceRevokeRequest,
    current_user: int = Depends(get_current_user),
):
    if req.user_id != current_user:
        http_error(403, "USER_MISMATCH", "user_id does not match token.")

    success = await revoke_device(current_user, req.device_id)
    if not success:
        http_error(404, "DEVICE_NOT_FOUND", "Device is not registered.")

    await bump_user_auth_version(current_user)
    await log_audit(
        current_user,
        "device_revoked",
        req.device_id[:64],
        "access/refresh session version invalidated; refresh tokens and key binding removed",
    )
    return success_response(
        {"user_id": current_user, "device_id": req.device_id, "revoked": True},
        "Device revoked",
    )


# ============================================================
# SESSIONS
# ============================================================

@app.get("/sessions")
async def sessions(current_user: int = Depends(get_current_user)):
    await require_subscription(current_user)

    admin = await is_admin(current_user)
    sm = get_instance()

    if admin:
        data = await sm.get_all_active_sessions()
    else:
        data = await sm.get_active_sessions_for_user(current_user)

    normalized = []
    for item in data:
        if "slot_id" in item:
            item["slot_id"] = int(item["slot_id"]) + 1

        status = str(item.get("status", "unknown")).lower()
        normalized.append({
            **item,
            "status": status,
            "running": bool(item.get("running")),
            "status_text": status.upper(),
            "elapsed_seconds": item.get("elapsed_seconds", 0),
            "remaining_seconds": item.get("remaining_seconds", 0),
        })

    return success_response({
        "count": len(normalized),
        "sessions": normalized,
    })


@app.get("/session/status/{slot_id}")
async def session_status(
    slot_id: int,
    current_user: int = Depends(get_current_user),
):
    await require_subscription(current_user)

    max_slots = await get_max_concurrency()
    if slot_id < 1 or slot_id > max_slots:
        http_error(404, "SLOT_NOT_FOUND", "Session slot not found.")

    session = await get_session_by_slot(slot_id - 1)

    if not session:
        return success_response({
            "slot_id": slot_id,
            "session_id": None,
            "status": "available",
            "running": False,
            "status_text": "AVAILABLE",
            "user_id": None,
            "ip": None,
            "port": None,
            "duration": 0,
            "duration_seconds": 0,
            "elapsed": 0,
            "elapsed_seconds": 0,
            "remaining": 0,
            "remaining_seconds": 0,
            "started_at": None,
            "expires_at": None,
            "is_paid": False,
            "cooldown_sec": 0,
        })

    if session["user_id"] != current_user and not await is_admin(current_user):
        http_error(403, "SESSION_FORBIDDEN", "This session belongs to another user.")

    sm = get_instance()
    try:
        data = await sm.get_session_data(session["session_id"])
    except SessionNotFound:
        http_error(404, "SESSION_NOT_FOUND", "Session data unavailable.")

    status = str(data.get("status", "unknown")).lower()
    duration = int(data.get("duration", 0) or 0)
    elapsed = int(data.get("elapsed_seconds", 0) or 0)
    remaining = int(data.get("remaining_seconds", 0) or 0)
    running = bool(data.get("running"))

    response_data = {
        **data,
        "status": status,
        "running": running,
        "status_text": status.upper(),
        "duration": duration,
        "duration_seconds": duration,
        "elapsed": elapsed,
        "elapsed_seconds": elapsed,
        "remaining": remaining,
        "remaining_seconds": remaining,
        "slot_id": slot_id,
    }
    return success_response(response_data)


@app.post("/session/start")
async def session_start(
    req: SessionStartRequest,
    current_user: int = Depends(get_current_user),
):
    sub = await require_subscription(current_user)

    timer_sec = sub.get("timer_seconds", 0)
    if timer_sec > 0 and req.duration > timer_sec:
        http_error(
            403,
            "DURATION_EXCEEDS_LIMIT",
            f"Max duration is {timer_sec} seconds.",
        )

    if not validate_ip(req.ip):
        http_error(422, "INVALID_IP", "Invalid target IP address.")

    if not validate_port(str(req.port)):
        http_error(422, "INVALID_PORT", "Port must be between 1 and 65535.")

    sm = get_instance()
    try:
        session_data = await sm.start_session(
            user_id=current_user,
            ip=req.ip,
            port=req.port,
            duration=req.duration,
            is_paid=True,
            cooldown_sec=sub.get("cooldown_seconds", 0),
            group_id=None,
            update=None,
        )
    except SessionUserActive as e:
        http_error(409, "USER_ACTIVE", str(e))
    except SessionCapacityError as e:
        http_error(409, "SLOT_UNAVAILABLE", str(e))
    except SessionGroupLimit as e:
        http_error(409, "GROUP_LIMIT", str(e))
    except SessionTransportError as e:
        http_error(502, "UPSTREAM_REJECTED", str(e))
    except SessionInternalError as e:
        logger.exception("Session start internal error")
        http_error(500, "INTERNAL_ERROR", str(e))
    except ValueError as e:
        http_error(422, "VALIDATION_ERROR", str(e))
    except Exception:
        logger.exception("Unexpected session start error")
        http_error(500, "INTERNAL_ERROR", "Could not start session.")

    if "slot_id" in session_data:
        session_data["slot_id"] = int(session_data["slot_id"]) + 1

    status = str(session_data.get("status", "running")).lower()
    duration = int(session_data.get("duration", req.duration) or 0)
    elapsed = int(session_data.get("elapsed_seconds", 0) or 0)
    remaining = int(
        session_data.get("remaining_seconds", max(0, duration - elapsed)) or 0
    )

    session_data = {
        **session_data,
        "status": status,
        "running": bool(session_data.get("running")),
        "status_text": status.upper(),
        "duration": duration,
        "duration_seconds": duration,
        "elapsed": elapsed,
        "elapsed_seconds": elapsed,
        "remaining": remaining,
        "remaining_seconds": remaining,
    }

    await log_audit(
        current_user,
        "apk_session_started",
        str(session_data.get("session_id", "")),
        f"slot={session_data.get('slot_id')} ip={req.ip} port={req.port} duration={req.duration}",
    )

    return success_response(session_data, message="Session started")


@app.post("/session/stop")
async def session_stop(
    req: SessionStopRequest,
    current_user: int = Depends(get_current_user),
):
    session = await get_session_by_slot(req.slot_id - 1)
    if not session:
        http_error(404, "SESSION_NOT_FOUND", "No active session on this slot.")

    if session["user_id"] != current_user and not await is_admin(current_user):
        http_error(403, "SESSION_FORBIDDEN", "You are not the owner of this session.")

    sm = get_instance()
    try:
        data = await sm.stop_session(
            session_id=session["session_id"],
            user_id=current_user,
            is_admin=await is_admin(current_user),
        )
    except SessionNotFound as e:
        http_error(404, "SESSION_NOT_FOUND", str(e))
    except SessionPermissionError as e:
        http_error(403, "FORBIDDEN", str(e))
    except SessionAlreadyEnded as e:
        http_error(409, "STOP_FAILED", str(e))
    except SessionError as e:
        http_error(409, "STOP_FAILED", str(e))
    except Exception:
        logger.exception("Session stop error")
        http_error(500, "INTERNAL_ERROR", "Could not stop session.")

    if "slot_id" in data:
        data["slot_id"] = int(data["slot_id"]) + 1

    status = str(data.get("status", "stopped")).lower()
    duration = int(data.get("duration", 0) or 0)
    elapsed = int(data.get("elapsed_seconds", 0) or 0)
    remaining = int(data.get("remaining_seconds", max(0, duration - elapsed)) or 0)

    data = {
        **data,
        "status": status,
        "running": bool(data.get("running")),
        "status_text": status.upper(),
        "duration": duration,
        "duration_seconds": duration,
        "elapsed": elapsed,
        "elapsed_seconds": elapsed,
        "remaining": remaining,
        "remaining_seconds": remaining,
    }

    await log_audit(
        current_user,
        "apk_session_stopped",
        str(data.get("session_id", session.get("session_id", ""))),
        f"slot={data.get('slot_id', req.slot_id)}",
    )

    return success_response(data, message="Session stopped")


# ============================================================
# SLOTS STATUS
# ============================================================

@app.get("/slots/status")
async def slots_status(current_user: int = Depends(get_current_user)):
    await require_subscription(current_user)

    max_slots = await get_max_concurrency()
    rows = await get_all_non_terminal_sessions()
    busy = {int(r.get("slot_id") or 0) for r in rows}

    slots_free = [i not in busy for i in range(max_slots)]

    return success_response({
        "slots": slots_free,
        "active_slots": sum(1 for f in slots_free if not f),
        "available_slots": sum(1 for f in slots_free if f),
    })


# ============================================================
# SERVER START
# ============================================================

def start_flask():
    uvicorn.run(
        app,
        host=API_HOST,
        port=int(API_PORT),
        log_level="error",
        access_log=False,
    )


start_api = start_flask


def start_api_in_thread() -> threading.Thread:
    t = threading.Thread(target=start_flask, daemon=True, name="zeus-api")
    t.start()
    return t


if __name__ == "__main__":
    start_flask()