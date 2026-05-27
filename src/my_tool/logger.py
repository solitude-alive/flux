import atexit
import datetime
import gzip
import json
import logging
import logging.handlers
import os
import os.path as osp
import shutil
import sys
import threading
import time
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any

# ---------------------------------------------------------------------------
# Public level constants (kept as ints to mirror the original API)
# ---------------------------------------------------------------------------

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
DISABLED = logging.CRITICAL + 10

LEVEL_NAMES: dict[int, str] = {
    DEBUG: "DEBUG",
    INFO: "INFO",
    WARN: "WARN",
    ERROR: "ERROR",
}

_RESET = "\033[0m"
_COLORS: dict[str, str] = {
    "DEBUG": "\033[94m",
    "INFO": "\033[92m",
    "WARN": "\033[93m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
}

_LOGGER_NAME = "flux"
_KV_LOGGER_NAME = "flux.kv"


# ===========================================================================
# Zone 1: DailySizeRotatingFileHandler
# ===========================================================================


class DailySizeRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """Handler that rolls over on both day change and size threshold.

    The active file lives at::

        <base_dir>/<YYYY-MM-DD>/<basename>.log

    where the date is rendered in the configured timezone (UTC by default).
    When the file exceeds ``max_bytes`` we rotate it on the same day to
    ``<basename>.log.1`` (older ``.1`` becomes ``.2``, ..., up to
    ``backup_count``). Files beyond ``backup_count`` are deleted; rotated
    files are optionally gzipped in a daemon thread so emit() is not blocked.

    Cross-day rollover does not rename anything -- new days simply open a
    fresh file in a new dated directory. Old day directories can be reaped
    automatically by setting ``retention_days``.
    """

    def __init__(
        self,
        base_dir: str,
        basename: str,
        *,
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 20,
        tz: datetime.tzinfo = datetime.UTC,
        gzip_old: bool = True,
        retention_days: int | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.base_dir = base_dir
        self.basename = basename
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self.tz = tz
        self.gzip_old = gzip_old
        self.retention_days = retention_days

        self._current_date: str = self._today()
        self._next_midnight_ts: float = 0.0

        initial_path = self._build_path(self._current_date)
        os.makedirs(osp.dirname(initial_path), exist_ok=True)
        super().__init__(initial_path, mode="a", encoding=encoding, delay=False)
        self._next_midnight_ts = self._compute_next_midnight_ts()
        if self.retention_days is not None:
            self._sweep_old_days()

    # -- time / path helpers ------------------------------------------------

    def _today(self) -> str:
        return datetime.datetime.now(self.tz).strftime("%Y-%m-%d")

    def _build_path(self, date_str: str) -> str:
        return osp.join(self.base_dir, date_str, f"{self.basename}.log")

    def _compute_next_midnight_ts(self) -> float:
        now = datetime.datetime.now(self.tz)
        tomorrow = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow.timestamp()

    # -- rollover decision --------------------------------------------------

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802
        if time.time() >= self._next_midnight_ts:
            return True

        if self.max_bytes <= 0 or self.stream is None:
            return False
        try:
            msg = self.format(record) + self.terminator
            msg_bytes = len(msg.encode(self.encoding or "utf-8", errors="replace"))
            return self.stream.tell() + msg_bytes >= self.max_bytes
        except Exception:
            return False

    # -- rollover action ----------------------------------------------------

    def doRollover(self) -> None:  # noqa: N802
        if self.stream:
            try:
                self.stream.flush()
            finally:
                try:
                    self.stream.close()
                except Exception:
                    pass
                # Signal to FileHandler.emit() that the stream needs re-opening
                # (mirrors stdlib's RotatingFileHandler.doRollover lifecycle).
                # Routed through __setattr__ so static checkers don't flag it
                # against the stub typing of stream: TextIO (which omits the
                # transient None state the lifecycle actually requires).
                object.__setattr__(self, "stream", None)

        new_date = self._today()
        day_changed = new_date != self._current_date

        if day_changed:
            self._current_date = new_date
            self._next_midnight_ts = self._compute_next_midnight_ts()
            self.baseFilename = self._build_path(new_date)
            os.makedirs(osp.dirname(self.baseFilename), exist_ok=True)
            if self.retention_days is not None:
                self._sweep_old_days()
        else:
            self._rotate_size_suffix()

        if not self.delay:
            self.stream = self._open()

    def _rotate_size_suffix(self) -> None:
        base = self.baseFilename

        if self.backup_count <= 0:
            for ext in ("", ".gz"):
                victim = f"{base}{ext}" if ext else base
                if osp.exists(victim):
                    try:
                        os.remove(victim)
                    except OSError:
                        pass
            return

        for i in range(self.backup_count, self.backup_count + 200):
            for ext in ("", ".gz"):
                victim = f"{base}.{i}{ext}"
                if osp.exists(victim):
                    try:
                        os.remove(victim)
                    except OSError:
                        pass

        for i in range(self.backup_count - 1, 0, -1):
            for ext in ("", ".gz"):
                src = f"{base}.{i}{ext}"
                dst = f"{base}.{i + 1}{ext}"
                if osp.exists(src):
                    try:
                        os.replace(src, dst)
                    except OSError:
                        pass

        if osp.exists(base):
            try:
                os.replace(base, f"{base}.1")
            except OSError:
                return
            if self.gzip_old:
                self._gzip_async(f"{base}.1")

    @staticmethod
    def _gzip_async(path: str) -> None:
        def _run() -> None:
            try:
                with open(path, "rb") as fi, gzip.open(path + ".gz", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(path)
            except Exception:
                # Compression is best-effort; never crash the host service.
                pass

        threading.Thread(target=_run, daemon=True, name="flux-gzip").start()

    # -- retention ----------------------------------------------------------

    def _sweep_old_days(self) -> None:
        if self.retention_days is None or self.retention_days <= 0:
            return
        try:
            cutoff = datetime.datetime.now(self.tz) - datetime.timedelta(days=self.retention_days)
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            if not osp.isdir(self.base_dir):
                return
            for name in os.listdir(self.base_dir):
                day_path = osp.join(self.base_dir, name)
                if not osp.isdir(day_path):
                    continue
                try:
                    datetime.datetime.strptime(name, "%Y-%m-%d")
                except ValueError:
                    continue
                if name < cutoff_str:
                    shutil.rmtree(day_path, ignore_errors=True)
        except Exception:
            pass


# ===========================================================================
# Zone 2: Formatters
# ===========================================================================


def _truncate(s: str, maxlen: int = 30) -> str:
    return s[: maxlen - 3] + "..." if len(s) > maxlen else s


def _json_safe(val: Any) -> Any:
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


class _MillisecondTimeMixin:
    """Shared millisecond / tz-aware ``formatTime`` helper for formatters."""

    tz: datetime.tzinfo

    def formatTime(  # noqa: N802
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=self.tz)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{dt.microsecond // 1000:03d}"


class HumanFormatter(_MillisecondTimeMixin, logging.Formatter):
    """Human-readable formatter with millisecond timestamps + optional color.

    Output template (uncolored)::

        [LEVEL]-[YYYY-MM-DD HH:MM:SS.mmm] <message>

    When ``use_color=True`` the level tag is wrapped in ANSI sequences.
    Exception traceback is appended automatically when ``record.exc_info``
    is set (this is what makes :func:`exception` useful).
    """

    def __init__(
        self,
        *,
        tz: datetime.tzinfo = datetime.UTC,
        use_color: bool = False,
        include_location: bool = False,
    ) -> None:
        super().__init__()
        self.tz = tz
        self.use_color = use_color
        self.include_location = include_location

    def format(self, record: logging.LogRecord) -> str:
        level_name = LEVEL_NAMES.get(record.levelno, record.levelname)
        if self.use_color:
            color = _COLORS.get(level_name, "")
            level_tag = f"{color}[{level_name}]{_RESET}"
        else:
            level_tag = f"[{level_name}]"
        ts = self.formatTime(record)
        msg = record.getMessage()
        if self.include_location:
            head = f"{level_tag}-[{ts}]-({record.module}:{record.lineno})"
        else:
            head = f"{level_tag}-[{ts}]"
        out = f"{head} {msg}" if msg else head
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            out += "\n" + self.formatStack(record.stack_info)
        return out


class KVTableFormatter(_MillisecondTimeMixin, logging.Formatter):
    """Renders ``record.kv_table`` (a Mapping) as a bordered ASCII table."""

    def __init__(self, *, tz: datetime.tzinfo = datetime.UTC) -> None:
        super().__init__()
        self.tz = tz

    def format(self, record: logging.LogRecord) -> str:
        kv = getattr(record, "kv_table", None)
        ts = self.formatTime(record)
        if not isinstance(kv, Mapping) or not kv:
            return f"[KV]-[{ts}] {record.getMessage()}"
        key2str: dict[str, str] = {}
        for key, val in sorted(kv.items()):
            valstr = f"{val:-8.3g}" if hasattr(val, "__float__") else str(val)
            key2str[_truncate(str(key))] = _truncate(valstr)
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [f"[KV]-[{ts}]", dashes]
        for key, val in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                f"| {key}{' ' * (keywidth - len(key))} | {val}{' ' * (valwidth - len(val))} |"
            )
        lines.append(dashes)
        return "\n".join(lines)


class JsonFormatter(_MillisecondTimeMixin, logging.Formatter):
    """One JSON object per log line. Suitable for ELK / Loki ingest."""

    def __init__(self, *, tz: datetime.tzinfo = datetime.UTC) -> None:
        super().__init__()
        self.tz = tz

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.datetime.fromtimestamp(record.created, tz=self.tz)
        payload: dict[str, Any] = {
            "ts": dt.isoformat(timespec="milliseconds"),
            "level": LEVEL_NAMES.get(record.levelno, record.levelname),
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        kv = getattr(record, "kv_table", None)
        if isinstance(kv, Mapping):
            payload["kv"] = {str(k): _json_safe(v) for k, v in kv.items()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ===========================================================================
# Zone 3: Core (_KVAggregator + Logger)
# ===========================================================================


class _KVAggregator:
    """Thread-safe accumulator for periodic metric dumps.

    ``logkv(key, val)``       -- overwrite the current value for ``key``.
    ``logkv_mean(key, val)``  -- incrementally average the value (Welford).
    ``snapshot_and_clear()``  -- atomically return + reset all values.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.name2val: dict[str, float] = defaultdict(float)
        self.name2cnt: dict[str, int] = defaultdict(int)

    def logkv(self, key: str, val: float) -> None:
        with self._lock:
            self.name2val[key] = val

    def logkv_mean(self, key: str, val: float) -> None:
        with self._lock:
            cnt = self.name2cnt[key]
            old = self.name2val[key]
            # Welford-style stable mean: new = old + (val - old) / (cnt + 1)
            self.name2val[key] = old + (float(val) - old) / (cnt + 1)
            self.name2cnt[key] = cnt + 1

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.name2val)

    def snapshot_and_clear(self) -> dict[str, float]:
        with self._lock:
            snap = dict(self.name2val)
            self.name2val.clear()
            self.name2cnt.clear()
            return snap

    def counts_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.name2cnt)


class Logger:
    """Holds the active configuration. Thin shell over stdlib ``logging``.

    The on-the-wire delivery is done by ``logging.getLogger('flux')`` (for
    ``info`` / ``warn`` / ``error`` / ``debug`` / ``exception``) and
    ``logging.getLogger('flux.kv')`` (only for KV table dumps).
    """

    DEFAULT: "Logger | None" = None
    CURRENT: "Logger | None" = None

    def __init__(
        self,
        dir: str | None,
        main_handlers: list[logging.Handler],
        kv_handlers: list[logging.Handler],
        comm: Any = None,
        level: int = INFO,
    ) -> None:
        self.dir = dir
        self._main_handlers = list(main_handlers)
        self._kv_handlers = list(kv_handlers)
        self.output_formats: list[logging.Handler] = self._main_handlers + self._kv_handlers
        self.comm = comm
        self.level = level
        self._kv = _KVAggregator()
        self._logger = logging.getLogger(_LOGGER_NAME)
        self._kv_logger = logging.getLogger(_KV_LOGGER_NAME)
        self._logger.propagate = False
        self._kv_logger.propagate = False
        self._logger.setLevel(level)
        self._kv_logger.setLevel(INFO)

    # -- back-compat shims for code that pokes the legacy internal state ----

    @property
    def name2val(self) -> dict[str, float]:
        return self._kv.name2val

    @property
    def name2cnt(self) -> dict[str, int]:
        return self._kv.name2cnt

    # -- handler lifecycle --------------------------------------------------

    def _reattach_handlers(self) -> None:
        """Re-bind our stored handlers to the named stdlib loggers.

        Used when restoring a previously-active Logger after another one
        detached its handlers (``scoped_configure`` / ``reset``).
        """
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
        for h in list(self._kv_logger.handlers):
            self._kv_logger.removeHandler(h)
        for h in self._main_handlers:
            self._logger.addHandler(h)
        for h in self._kv_handlers:
            self._kv_logger.addHandler(h)
        self._logger.setLevel(self.level)

    def close(self) -> None:
        for h in list(self._logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            self._logger.removeHandler(h)
        for h in list(self._kv_logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
            self._kv_logger.removeHandler(h)

    # -- logging ------------------------------------------------------------

    def log(self, *args: Any, level: int = INFO) -> None:
        if not args:
            return
        if not self._logger.isEnabledFor(level):
            return
        msg = " ".join(str(a) for a in args)
        self._logger.log(level, msg, stacklevel=3)

    def exception(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args) if args else ""
        self._logger.exception(msg, stacklevel=3)

    # -- KV -----------------------------------------------------------------

    def logkv(self, key: str, val: float) -> None:
        self._kv.logkv(key, val)

    def logkv_mean(self, key: str, val: float) -> None:
        self._kv.logkv_mean(key, val)

    def dumpkvs(self) -> dict[str, float]:
        if self.comm is None:
            d = self._kv.snapshot_and_clear()
        else:
            local = {
                name: (val, self._kv.name2cnt.get(name, 1) or 1)
                for name, val in list(self._kv.name2val.items())
            }
            self._kv.snapshot_and_clear()
            d = mpi_weighted_mean(self.comm, local)
            if getattr(self.comm, "rank", 0) != 0:
                return {}
        if d:
            self._kv_logger.info("", extra={"kv_table": dict(d)})
        return d

    # -- config -------------------------------------------------------------

    def set_level(self, level: int) -> None:
        self.level = level
        self._logger.setLevel(level)

    def set_comm(self, comm: Any) -> None:
        self.comm = comm

    def get_dir(self) -> str | None:
        return self.dir


# ===========================================================================
# Zone 4: Public API
# ===========================================================================


def get_current() -> Logger:
    """Return the active Logger, configuring a default one if needed."""
    if Logger.CURRENT is None:
        _configure_default_logger()
    assert Logger.CURRENT is not None
    return Logger.CURRENT


def logkv(key: str, val: float) -> None:
    """Log the latest value of a diagnostic. Repeat calls overwrite."""
    get_current().logkv(key, val)


def logkv_mean(key: str, val: float) -> None:
    """Log a value that will be averaged across the current dump window."""
    get_current().logkv_mean(key, val)


def logkvs(d: Mapping[str, float]) -> None:
    """Log a dict of key/value pairs."""
    for k, v in d.items():
        logkv(k, v)


def dumpkvs() -> dict[str, float]:
    """Flush accumulated KV pairs to the kvs.log handler."""
    return get_current().dumpkvs()


def getkvs() -> dict[str, float]:
    """Return a live view of accumulated KV pairs (not a copy)."""
    return get_current().name2val


def log(*args: Any, level: int = INFO) -> None:
    """Write space-joined args at ``level`` to the configured outputs."""
    get_current().log(*args, level=level)


def debug(*args: Any) -> None:
    log(*args, level=DEBUG)


def info(*args: Any) -> None:
    log(*args, level=INFO)


def warn(*args: Any) -> None:
    log(*args, level=WARN)


def error(*args: Any) -> None:
    log(*args, level=ERROR)


def exception(*args: Any) -> None:
    """Log a message at ERROR level plus the current traceback."""
    get_current().exception(*args)


def set_level(level: int) -> None:
    """Set the active threshold of the global logger."""
    get_current().set_level(level)


def set_comm(comm: Any) -> None:
    """Attach an MPI communicator for cross-rank KV aggregation."""
    get_current().set_comm(comm)


def get_dir() -> str | None:
    """Return the root directory log files are being written to."""
    return get_current().get_dir()


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "disabled"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return str(n)


def describe_config() -> dict[str, Any]:
    """Return the active logger's configuration as a dict (for testing/programmatic use)."""
    cur = get_current()
    handlers_info: list[dict[str, Any]] = []
    for h in cur.output_formats:
        info_dict: dict[str, Any] = {
            "type": type(h).__name__,
            "level": logging.getLevelName(h.level),
            "formatter": type(h.formatter).__name__ if h.formatter else None,
        }
        if isinstance(h, DailySizeRotatingFileHandler):
            info_dict.update(
                {
                    "base_dir": h.base_dir,
                    "basename": h.basename,
                    "max_bytes": h.max_bytes,
                    "backup_count": h.backup_count,
                    "tz": str(h.tz),
                    "gzip_old": h.gzip_old,
                    "retention_days": h.retention_days,
                    "encoding": h.encoding,
                    "current_file": h.baseFilename,
                }
            )
        elif isinstance(h, logging.StreamHandler):
            stream = getattr(h, "stream", None)
            info_dict["stream"] = getattr(stream, "name", repr(stream))
        handlers_info.append(info_dict)

    return {
        "dir": cur.dir,
        "level": logging.getLevelName(cur.level),
        "comm": repr(cur.comm) if cur.comm is not None else None,
        "atexit_registered": _ATEXIT_REGISTERED,
        "raise_exceptions": logging.raiseExceptions,
        "num_handlers": len(cur.output_formats),
        "handlers": handlers_info,
    }


def print_config(file: Any = None) -> None:
    """Pretty-print the active logger's configuration.

    Useful right after :func:`configure` to verify what's actually in effect.

    Args:
        file: Optional stream to print to. Defaults to ``sys.stdout``.
    """
    out = file if file is not None else sys.stdout
    cfg = describe_config()

    print("=" * 70, file=out)
    print("flux.logger configuration", file=out)
    print("=" * 70, file=out)
    print(f"  log root         : {cfg['dir']}", file=out)
    print(f"  global level     : {cfg['level']}", file=out)
    print(f"  num handlers     : {cfg['num_handlers']}", file=out)
    print(f"  MPI comm         : {cfg['comm']}", file=out)
    print(f"  atexit hooked    : {cfg['atexit_registered']}", file=out)
    print(f"  raiseExceptions  : {cfg['raise_exceptions']}", file=out)
    print("-" * 70, file=out)
    for i, h in enumerate(cfg["handlers"]):
        tag = f"[{i}] {h['type']}"
        print(f"  {tag}", file=out)
        print(f"      level        : {h['level']}", file=out)
        print(f"      formatter    : {h['formatter']}", file=out)
        if "stream" in h:
            print(f"      stream       : {h['stream']}", file=out)
        if "current_file" in h:
            print(f"      current_file : {h['current_file']}", file=out)
            print(
                f"      max_bytes    : {h['max_bytes']} ({_fmt_bytes(h['max_bytes'])})",
                file=out,
            )
            print(f"      backup_count : {h['backup_count']}", file=out)
            retention_line = (
                f"      retention    : {h['retention_days']} days"
                if h["retention_days"]
                else "      retention    : disabled"
            )
            print(retention_line, file=out)
            print(f"      gzip_old     : {h['gzip_old']}", file=out)
            print(f"      tz           : {h['tz']}", file=out)
            print(f"      encoding     : {h['encoding']}", file=out)
    print("=" * 70, file=out)


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename: str):
    """Context manager that records elapsed time under ``wait_<scopename>``."""
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().logkv_mean(logkey, time.time() - tstart)


def profile(n: str):
    """Decorator equivalent of :func:`profile_kv`."""

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ===========================================================================
# MPI helpers (kept and bug-fixed)
# ===========================================================================


def get_rank_without_mpi_import() -> int:
    """Return the MPI rank read from env; never triggers ``MPI_Init()``."""
    for varname in ("PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        if varname in os.environ:
            try:
                return int(os.environ[varname])
            except ValueError:
                return 0
    return 0


def mpi_weighted_mean(
    comm: Any,
    local_name2valcount: Mapping[str, tuple[float, int]],
) -> dict[str, float]:
    """Weighted average over dicts that each live on a separate MPI rank.

    Args:
        comm: an ``mpi4py`` communicator (must support ``gather``).
        local_name2valcount: ``{name: (value, count)}`` on the local rank.

    Returns:
        On rank 0: ``{name: weighted_mean}``.
        On other ranks: ``{}``.
    """
    gathered = comm.gather(local_name2valcount)
    if getattr(comm, "rank", 0) != 0:
        return {}
    name2sum: dict[str, float] = defaultdict(float)
    name2count: dict[str, float] = defaultdict(float)
    for n2vc in gathered or []:
        for name, (val, count) in n2vc.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                warnings.warn(
                    f"mpi_weighted_mean: non-float {name}={val!r}, skipping",
                    stacklevel=2,
                )
                continue
            name2sum[name] += fval * count
            name2count[name] += count
    return {n: name2sum[n] / name2count[n] for n in name2sum if name2count[n]}


# ===========================================================================
# Zone 5: configure() + lifecycle
# ===========================================================================


_ATEXIT_REGISTERED = False


def _resolve_log_dir(dir_log: str | None, *, root_dir: bool = False) -> str:
    """Resolve the on-disk log root.

    Selection order for the raw path:
      1. ``dir_log`` argument (if provided).
      2. ``$LOG_DIR`` environment variable.
      3. Default ``"log"``.

    Anchoring (controlled by ``root_dir``):
      * ``root_dir=False`` (default): the path is always anchored under
        ``$STATIC_DIR`` (when set) else ``$CWD``. This means even an
        explicit ``dir_log`` is treated as a *sub-directory name* unless
        the caller opts out. (If the supplied path is itself absolute,
        ``os.path.join`` will naturally discard the anchor, so passing an
        absolute path also acts as an implicit opt-out.)
      * ``root_dir=True``: the supplied path is used verbatim (after
        ``expanduser`` + ``abspath``). Use this when you want to point at
        a fixed location like ``/var/log/myservice`` and you do NOT want
        ``$STATIC_DIR`` to apply.
    """
    if dir_log is None:
        dir_log = os.getenv("LOG_DIR")
    if dir_log is None:
        dir_log = "log"
    dir_log = osp.expanduser(dir_log)
    if not root_dir:
        static_dir = os.getenv("STATIC_DIR")
        anchor = static_dir if static_dir else os.getcwd()
        dir_log = osp.join(anchor, dir_log)
    return osp.abspath(dir_log)


_LEVEL_OF_FILE_SPEC: dict[str, int] = {
    "debug": DEBUG,
    "info": INFO,
    "warn": WARN,
    "error": ERROR,
}


def _build_handler(
    spec: str,
    *,
    dir_log: str,
    log_suffix: str,
    tz: datetime.tzinfo,
    max_bytes: int,
    backup_count: int,
    gzip_old: bool,
    retention_days: int | None,
    json_output: bool,
) -> logging.Handler:
    """Construct one handler from a string spec (e.g. ``"info"``, ``"stdout"``)."""
    if spec == "stdout":
        h: logging.Handler = logging.StreamHandler(sys.stdout)
        use_color = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        h.setFormatter(HumanFormatter(tz=tz, use_color=use_color))
        h.setLevel(DEBUG)
        return h

    if spec == "stderr":
        h = logging.StreamHandler(sys.stderr)
        use_color = sys.stderr.isatty() and not os.getenv("NO_COLOR")
        h.setFormatter(HumanFormatter(tz=tz, use_color=use_color))
        h.setLevel(WARN)
        return h

    if spec in _LEVEL_OF_FILE_SPEC:
        basename = f"{spec}{log_suffix}"
        fh = DailySizeRotatingFileHandler(
            base_dir=dir_log,
            basename=basename,
            max_bytes=max_bytes,
            backup_count=backup_count,
            tz=tz,
            gzip_old=gzip_old,
            retention_days=retention_days,
        )
        fh.setLevel(_LEVEL_OF_FILE_SPEC[spec])
        fh.setFormatter(JsonFormatter(tz=tz) if json_output else HumanFormatter(tz=tz))
        return fh

    if spec == "kv":
        basename = f"kvs{log_suffix}"
        fh = DailySizeRotatingFileHandler(
            base_dir=dir_log,
            basename=basename,
            max_bytes=max_bytes,
            backup_count=backup_count,
            tz=tz,
            gzip_old=gzip_old,
            retention_days=retention_days,
        )
        fh.setLevel(INFO)
        fh.setFormatter(JsonFormatter(tz=tz) if json_output else KVTableFormatter(tz=tz))
        return fh

    raise ValueError(f"Unknown format spec: {spec!r}")


def configure(
    dir_log: str | None = None,
    *,
    root_dir: bool = False,
    format_strs: Iterable[str] | str | None = None,
    level: int = INFO,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 20,
    tz: datetime.tzinfo = datetime.UTC,
    gzip_old: bool = True,
    retention_days: int | None = None,
    comm: Any = None,
    log_suffix: str = "",
    json_output: bool = False,
) -> None:
    """Set up (or re-set) the global logger.

    Files are written under::

        <log_root>/<YYYY-MM-DD>/<spec><log_suffix>.log

    where ``log_root`` is resolved per :func:`_resolve_log_dir`. Each file
    rolls over when the day changes (in ``tz``) or when its size exceeds
    ``max_bytes``. Up to ``backup_count`` size-rotated files are kept; older
    ones are removed. Set ``retention_days`` to also delete day-folders that
    are older than that.

    Args:
        dir_log: Explicit log path (highest precedence). By default this
            is treated as a *sub-directory name* under ``$STATIC_DIR`` (or
            ``$CWD`` when unset). Pass ``root_dir=True`` to use it as-is.
        root_dir: When True, ``dir_log`` is used verbatim and the
            ``$STATIC_DIR`` anchor is NOT applied. Use this for fixed
            paths like ``/var/log/myservice``.
        format_strs: Iterable of specs, e.g. ``["stdout", "info", "warn"]``.
            Comma-separated string also accepted. Defaults to
            ``$LOG_FORMAT`` or ``"stdout,info,warn,error"``.
        level: Global level threshold (``INFO`` by default).
        max_bytes: Per-file size threshold; ``0`` disables size rotation.
        backup_count: Maximum number of size-rotated files to keep.
        tz: Timezone used to compute day boundaries + timestamps.
        gzip_old: If True, gzip rotated ``.1`` files in a background thread.
        retention_days: If set, delete day-folders older than this on rollover.
        comm: Optional ``mpi4py`` communicator for cross-rank KV aggregation.
        log_suffix: Suffix appended to every file basename.
        json_output: Use :class:`JsonFormatter` instead of human-readable.
    """
    global _ATEXIT_REGISTERED

    resolved_dir = _resolve_log_dir(dir_log, root_dir=root_dir)
    os.makedirs(resolved_dir, exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = f"{log_suffix}-rank{rank:03d}"

    if format_strs is None:
        default = "stdout,info,warn,error" if rank == 0 else "info"
        env = os.getenv("LOG_FORMAT" if rank == 0 else "LOG_FORMAT_MPI", default)
        format_strs = env.split(",")
    elif isinstance(format_strs, str):
        format_strs = format_strs.split(",")
    fmt_list = [str(f).strip() for f in format_strs if f and str(f).strip()]

    if not any(f == "kv" for f in fmt_list):
        fmt_list.append("kv")

    handlers_for_main: list[logging.Handler] = []
    handlers_for_kv: list[logging.Handler] = []
    for spec in fmt_list:
        h = _build_handler(
            spec,
            dir_log=resolved_dir,
            log_suffix=log_suffix,
            tz=tz,
            max_bytes=max_bytes,
            backup_count=backup_count,
            gzip_old=gzip_old,
            retention_days=retention_days,
            json_output=json_output,
        )
        if spec == "kv":
            handlers_for_kv.append(h)
        else:
            handlers_for_main.append(h)

    if Logger.CURRENT is not None and Logger.CURRENT is not Logger.DEFAULT:
        try:
            Logger.CURRENT.close()
        except Exception:
            pass

    # Disk-full / perm-denied writes degrade to stderr instead of crashing.
    logging.raiseExceptions = False

    main_logger = logging.getLogger(_LOGGER_NAME)
    kv_logger = logging.getLogger(_KV_LOGGER_NAME)
    for h in list(main_logger.handlers):
        main_logger.removeHandler(h)
    for h in list(kv_logger.handlers):
        kv_logger.removeHandler(h)
    for h in handlers_for_main:
        main_logger.addHandler(h)
    for h in handlers_for_kv:
        kv_logger.addHandler(h)

    Logger.CURRENT = Logger(
        dir=resolved_dir,
        main_handlers=handlers_for_main,
        kv_handlers=handlers_for_kv,
        comm=comm,
        level=level,
    )

    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_cleanup)
        _ATEXIT_REGISTERED = True

    if handlers_for_main:
        log(f"Logging to {resolved_dir}")


def _atexit_cleanup() -> None:
    try:
        if Logger.CURRENT is not None:
            Logger.CURRENT.close()
    except Exception:
        pass


def _configure_default_logger() -> None:
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset() -> None:
    """Close the active logger and restore the default one."""
    if Logger.CURRENT is not None and Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        if Logger.CURRENT is not None:
            Logger.CURRENT._reattach_handlers()
        log("Reset logger")


@contextmanager
def scoped_configure(
    dir: str | None = None,
    format_strs: Iterable[str] | str | None = None,
    comm: Any = None,
):
    """Temporarily replace the active logger inside a ``with`` block."""
    prev = Logger.CURRENT
    configure(dir_log=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        if Logger.CURRENT is not None and Logger.CURRENT is not prev:
            Logger.CURRENT.close()
        Logger.CURRENT = prev
        if prev is not None:
            prev._reattach_handlers()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------


__all__ = [
    "DEBUG",
    "DISABLED",
    "ERROR",
    "INFO",
    "WARN",
    "DailySizeRotatingFileHandler",
    "HumanFormatter",
    "JsonFormatter",
    "KVTableFormatter",
    "Logger",
    "configure",
    "debug",
    "describe_config",
    "dump_tabular",
    "dumpkvs",
    "error",
    "exception",
    "get_current",
    "get_dir",
    "get_rank_without_mpi_import",
    "getkvs",
    "info",
    "log",
    "logkv",
    "logkv_mean",
    "logkvs",
    "mpi_weighted_mean",
    "print_config",
    "profile",
    "profile_kv",
    "record_tabular",
    "reset",
    "scoped_configure",
    "set_comm",
    "set_level",
    "warn",
]
