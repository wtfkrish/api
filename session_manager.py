"""APK session manager.

Bridges the APK's /session/* contract onto the NEW bot's submit pipeline.

The new bot has no Telethon slot pool. Its concurrency gate lives inside
`submit_commands._ACTIVE_ATTACKS` (shared with Telegram /attack). This
module reserves capacity there, performs the actual HTTPS submit via
`submit_commands._perform_submit_request`, and records a logical session
row in the existing `sessions` table so the APK sees the same shape it
always did.

Liveness invariant: a session is "running" only while
``started_at + duration > now``. The ``sessions.status`` column is a
best-effort mirror written by the lifecycle task; it MUST NOT be treated
as authoritative for API responses. All queries that decide whether a
session is active filter on ``expires_at`` as well.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import config

from database import (
    get_all_non_terminal_sessions_raw,
    get_active_sessions as db_get_active_sessions,
    get_session as db_get_session,
    get_max_concurrency,
    save_session,
    update_session_status,
)
from submit_commands import (
    _ACTIVE_ATTACKS,
    _reserve_attack_slot,
    _release_attack_slot,
    _perform_submit_request,
    _is_upstream_full,
)
from state import state

logger = logging.getLogger(__name__)

_TERMINAL = ("completed", "stopped", "failed")


class SessionError(Exception):
    """Base for all session-manager failures."""


class SessionUserActive(SessionError):
    """The caller already holds an active session."""


class SessionCapacityError(SessionError):
    """Global concurrency ceiling reached."""


class SessionGroupLimit(SessionError):
    """Group concurrency ceiling reached."""


class SessionInternalError(SessionError):
    """A booking/reconciliation invariant was violated."""


class SessionNotFound(SessionError):
    """No DB row for the requested session id."""


class SessionAlreadyEnded(SessionError):
    """The session is already terminal and cannot be stopped again."""


class SessionPermissionError(SessionError):
    """Caller does not own the session and is not an admin."""


class SessionTransportError(SessionError):
    """Upstream submit transport rejected the request."""


class _UserShim:
    __slots__ = ("id", "username", "first_name", "last_name", "full_name",
                 "is_bot", "language_code", "source")

    def __init__(self, user_id: int, username: str = "",
                 first_name: str = "", last_name: str = ""):
        self.id = int(user_id)
        self.username = username or ""
        self.first_name = first_name or ""
        self.last_name = last_name or ""
        self.full_name = " ".join(p for p in (self.first_name, self.last_name) if p)
        self.is_bot = False
        self.language_code = None
        self.source = "apk"


class _ChatShim:
    __slots__ = ("type", "id")

    def __init__(self):
        self.type = "private"
        self.id = 0


def _is_live(row: dict, now: int) -> bool:
    status = str(row.get("status") or "").lower()
    if status in _TERMINAL:
        return False
    expires_at = int(row.get("expires_at") or 0)
    if expires_at <= 0:
        return False
    return expires_at > now


class ApkSessionManager:
    """Logical session pool adapter for the APK API."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._completion_tasks: dict[str, asyncio.Task] = {}
        self._start_lock = asyncio.Lock()

    def initialize(self, slots, loop):
        self._loop = loop

    async def reconcile_startup(self) -> int:
        rows = await get_all_non_terminal_sessions_raw()
        now = int(time.time())
        count = 0
        for row in rows:
            sid = row.get("session_id")
            if not sid:
                continue
            if _is_live(row, now):
                duration = int(row.get("duration") or 0)
                started = int(row.get("started_at") or 0)
                remaining = max(1, (started + duration) - now) if started and duration else 1
                task = asyncio.create_task(
                    self._lifecycle(sid, remaining),
                    name=f"apk-session-resume-{sid[:8]}",
                )
                self._completion_tasks[sid] = task
                task.add_done_callback(
                    lambda _t, s=sid: self._completion_tasks.pop(s, None)
                )
                continue
            try:
                await update_session_status(sid, "failed")
                count += 1
            except Exception:
                logger.exception("Could not reconcile session %s", sid)
        return count

    async def _reconcile_pool(self) -> None:
        now = int(time.time())
        stale_ids: list[str] = []
        for pool_session_id, record in list(_ACTIVE_ATTACKS.items()):
            if str(record.get("source") or "") != "apk":
                continue
            row = await db_get_session(pool_session_id)
            if row is None:
                stale_ids.append(pool_session_id)
                continue
            if not _is_live(row, now):
                stale_ids.append(pool_session_id)

        for pool_session_id in stale_ids:
            record = _ACTIVE_ATTACKS.get(pool_session_id)
            if record and str(record.get("source") or "") == "apk":
                logger.info(
                    "Reconciling stale pool slot: session_id=%s user_id=%s",
                    pool_session_id,
                    record.get("user_id"),
                )
                await _release_attack_slot(pool_session_id)

    def _decorate(self, row: dict, now: Optional[int] = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        duration = int(row.get("duration") or 0)
        started = int(row.get("started_at") or 0)
        expires_at = int(row.get("expires_at") or 0)
        if not expires_at and started and duration:
            expires_at = started + duration

        live = _is_live(row, now)
        elapsed = max(0, now - started) if started else 0
        elapsed = min(duration, elapsed) if duration else elapsed
        remaining = max(0, expires_at - now) if expires_at else 0

        stored_status = str(row.get("status") or "running").lower()
        if live:
            effective_status = stored_status if stored_status in ("starting", "running", "stopping") else "running"
        else:
            effective_status = stored_status if stored_status in _TERMINAL else "completed"

        return {
            "slot_id": int(row.get("slot_id") or 0),
            "session_id": row.get("session_id"),
            "user_id": int(row.get("user_id") or 0),
            "ip": row.get("ip"),
            "port": int(row.get("port") or 0),
            "duration": duration,
            "duration_seconds": duration,
            "elapsed": elapsed,
            "elapsed_seconds": elapsed,
            "remaining": remaining,
            "remaining_seconds": remaining,
            "started_at": started or None,
            "expires_at": expires_at or None,
            "status": effective_status,
            "running": live,
            "status_text": effective_status.upper(),
            "is_paid": bool(row.get("is_paid", 1)),
            "cooldown_sec": int(row.get("cooldown_sec") or 0),
            "group_id": row.get("group_id"),
            "source": "apk",
        }

    async def get_all_active_sessions(self):
        rows = await get_all_non_terminal_sessions_raw()
        now = int(time.time())
        return [self._decorate(r, now) for r in rows if _is_live(r, now)]

    async def get_active_sessions_for_user(self, user_id: int):
        rows = await db_get_active_sessions(int(user_id))
        now = int(time.time())
        return [self._decorate(r, now) for r in rows if _is_live(r, now)]

    async def get_session_data(self, session_id: str) -> dict:
        row = await db_get_session(session_id)
        if not row:
            raise SessionNotFound("Session not found")
        return self._decorate(row)

    async def start_session(
        self,
        *,
        user_id: int,
        ip: str,
        port: int,
        duration: int,
        is_paid: bool = True,
        cooldown_sec: int = 0,
        group_id=None,
        update=None,
    ) -> dict:
        async with self._start_lock:
            await self._reconcile_pool()

            existing = await db_get_active_sessions(int(user_id))
            now = int(time.time())
            live_existing = [r for r in existing if _is_live(r, now)]
            if live_existing:
                raise SessionUserActive(
                    "An active session already exists for this account"
                )

            shim_user = _UserShim(user_id)
            shim_chat = _ChatShim()

            attempted_providers: set = set()
            pool_session_id = None
            pool_record = None
            provider = None
            result = None
            reservation_error = None

            while True:
                pool_session_id, provider_key, reservation_error = await _reserve_attack_slot(
                    shim_user, shim_chat, exclude=attempted_providers
                )

                if pool_session_id is None:
                    if reservation_error == "user_active":
                        raise SessionUserActive(
                            "An active session already exists for this account"
                        )
                    if reservation_error == "global_limit":
                        raise SessionCapacityError("All attack slots are currently busy")
                    if reservation_error == "group_limit":
                        raise SessionGroupLimit(
                            "This group has reached its concurrent slot limit"
                        )
                    logger.error(
                        "Unexpected reservation_error=%r for user_id=%s",
                        reservation_error, user_id,
                    )
                    raise SessionInternalError(
                        f"Attack slot reservation failed: "
                        f"{reservation_error or 'unknown reason'}"
                    )

                pool_record = _ACTIVE_ATTACKS.get(pool_session_id)
                if pool_record is None:
                    logger.error(
                        "Reservation returned session_id=%r but no pool record exists",
                        pool_session_id,
                    )
                    await _release_attack_slot(pool_session_id)
                    raise SessionInternalError(
                        "Internal slot reservation inconsistency"
                    )

                provider = config.get_provider(provider_key)
                if provider is None:
                    await _release_attack_slot(pool_session_id)
                    raise SessionInternalError(
                        "Reserved slot references an unknown provider"
                    )

                try:
                    result = await asyncio.to_thread(
                        _perform_submit_request, provider, ip, int(port), int(duration)
                    )
                except Exception as exc:
                    await _release_attack_slot(pool_session_id)
                    logger.exception("Submit API transport error for user=%s", user_id)
                    raise SessionTransportError(
                        f"Submit API transport error: {exc}"
                    ) from exc

                if result.get("ok"):
                    break

                upstream_message = result.get("message")
                await _release_attack_slot(pool_session_id)
                pool_session_id = None
                pool_record = None

                if _is_upstream_full(upstream_message):
                    attempted_providers.add(provider_key)
                    continue

                message = str(upstream_message or "Attack rejected by upstream API")
                normalized = " ".join(message.strip().lower().split())
                if "no concurrent slots" in normalized:
                    message = "Try again later."
                raise SessionTransportError(message)

            now = int(time.time())
            expires_at = now + int(duration)
            session_id = pool_session_id

            pool_slot_id = int(pool_record.get("slot_id") or 0)
            if pool_slot_id < 1:
                await _release_attack_slot(pool_session_id)
                raise SessionInternalError("Invalid shared attack slot reservation")
            slot_id = pool_slot_id - 1

            pool_record.update({
                "ip": ip,
                "port": int(port),
                "duration": int(duration),
                "started_at": now,
                "chat_id": 0,
                "bot": None,
                "message": None,
                "attack_message_id": None,
                "feedback_message": None,
                "source": "apk",
                "apk_user_id": int(user_id),
            })

            try:
                await save_session(
                    session_id=session_id,
                    slot_id=int(slot_id),
                    user_id=int(user_id),
                    ip=ip,
                    port=int(port),
                    duration=int(duration),
                    started_at=now,
                    expires_at=expires_at,
                    status="running",
                    is_paid=is_paid,
                    cooldown_sec=cooldown_sec,
                    group_id=group_id,
                )
            except Exception:
                await _release_attack_slot(pool_session_id)
                logger.exception(
                    "save_session failed for session=%s; pool slot released",
                    session_id,
                )
                raise

            task = asyncio.create_task(
                self._lifecycle(session_id, int(duration)),
                name=f"apk-session-{session_id[:8]}",
            )
            self._completion_tasks[session_id] = task
            task.add_done_callback(
                lambda _t, sid=session_id: self._completion_tasks.pop(sid, None)
            )

            return self._decorate({
                "slot_id": int(slot_id),
                "session_id": session_id,
                "user_id": int(user_id),
                "ip": ip,
                "port": int(port),
                "duration": int(duration),
                "started_at": now,
                "expires_at": expires_at,
                "status": "running",
                "is_paid": 1 if is_paid else 0,
                "cooldown_sec": cooldown_sec,
                "group_id": group_id,
            }, now)

    async def stop_session(
        self,
        *,
        session_id: str,
        user_id: int,
        is_admin: bool = False,
    ) -> dict:
        row = await db_get_session(session_id)
        if not row:
            raise SessionNotFound("Session not found")

        owner_id = int(row.get("user_id") or 0)
        if not is_admin and owner_id != int(user_id):
            raise SessionPermissionError("You do not own this session")

        if not _is_live(row, int(time.time())):
            await _release_attack_slot(session_id)
            return self._decorate(row)

        await update_session_status(session_id, "stopped")

        task = self._completion_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("lifecycle task raised during stop", exc_info=True)

        await _release_attack_slot(session_id)

        updated = await db_get_session(session_id) or row
        return self._decorate(updated)

    async def _lifecycle(self, session_id: str, duration: int):
        try:
            await asyncio.sleep(max(1, int(duration)))
        except asyncio.CancelledError:
            raise

        try:
            row = await db_get_session(session_id)
            if row and str(row.get("status") or "").lower() not in _TERMINAL:
                try:
                    await update_session_status(session_id, "completed")
                except Exception:
                    logger.exception(
                        "Failed to mark session %s completed in DB",
                        session_id,
                    )
        finally:
            await _release_attack_slot(session_id)
            self._completion_tasks.pop(session_id, None)


_instance: Optional[ApkSessionManager] = None


def initialize(slots, loop) -> ApkSessionManager:
    global _instance
    if _instance is None:
        _instance = ApkSessionManager()
    _instance.initialize(slots, loop)
    return _instance


def get_instance() -> ApkSessionManager:
    global _instance
    if _instance is None:
        _instance = ApkSessionManager()
    return _instance
