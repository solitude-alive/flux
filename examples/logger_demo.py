import os
import threading
import time
from pathlib import Path

from my_tool.logger import (
    DEBUG,
    profile_kv,
)
from my_tool.logger import (
    configure as log_configure,
)
from my_tool.logger import (
    debug as log_debug,
)
from my_tool.logger import (
    dumpkvs as log_dumpkvs,
)
from my_tool.logger import (
    error as log_error,
)
from my_tool.logger import (
    exception as log_exception,
)
from my_tool.logger import (
    info as log_info,
)
from my_tool.logger import (
    logkv as log_kv,
)
from my_tool.logger import (
    logkv_mean as log_kv_mean,
)
from my_tool.logger import (
    print_config as log_print_config,
)
from my_tool.logger import (
    warn as log_warn,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    log_dir = Path(__file__).resolve().parent.parent / "log"

    # ------------------------------------------------------------------
    # 1. Configure
    # ------------------------------------------------------------------
    # max_bytes is intentionally tiny (4 KiB) so the demo triggers a
    # size-rotation within a few writes; in production use the default
    # 100 MB or larger.
    log_configure(
        dir_log=str(log_dir),
        root_dir=True,
        format_strs=["stdout", "info", "warn", "error", "debug"],
        level=DEBUG,
        max_bytes=4 * 1024,
        backup_count=3,
        gzip_old=False,
        retention_days=7,
    )

    # Show what's actually in effect (handy right after configure()).
    log_print_config()

    # ------------------------------------------------------------------
    # 2. Levels + exception()
    # ------------------------------------------------------------------
    section("Levels")
    log_debug("a debug line (only visible because level=DEBUG)")
    log_info("service started", "pid=", os.getpid())
    log_warn("disk usage > 80%")
    log_error("upstream timed out, retrying")

    try:
        raise ValueError("simulated failure")
    except ValueError:
        log_exception("recoverable error while parsing payload")

    # ------------------------------------------------------------------
    # 3. KV metrics
    # ------------------------------------------------------------------
    section("KV metrics")
    log_kv("active_connections", 42)
    for rt in (12.0, 18.0, 14.0, 22.0):
        log_kv_mean("rt_ms", rt)

    with profile_kv("downstream_call"):
        time.sleep(0.05)
    with profile_kv("downstream_call"):
        time.sleep(0.03)

    snapshot = log_dumpkvs()
    print(f"KV snapshot just dumped to kvs.log: {snapshot}")

    # ------------------------------------------------------------------
    # 4. Concurrent logging
    # ------------------------------------------------------------------
    section("Concurrent logging (8 threads x 50 lines)")

    def worker(tid: int) -> None:
        for i in range(50):
            log_info(f"thread-{tid:02d}", f"event-{i:03d}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ------------------------------------------------------------------
    # 5. Trigger size rotation by emitting padded lines
    # ------------------------------------------------------------------
    section("Triggering size-based rotation")
    big = "X" * 300
    for i in range(20):
        log_info(f"big-line-{i:02d}", big)

    # ------------------------------------------------------------------
    # 6. Inspect what we wrote
    # ------------------------------------------------------------------
    section("File tree under log/")
    for day_dir in sorted(log_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        print(day_dir.name + "/")
        for f in sorted(day_dir.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name:24} {size:>8} bytes")


if __name__ == "__main__":
    """
    python examples/logger_demo.py
    """
    main()
