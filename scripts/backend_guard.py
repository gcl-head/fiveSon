"""Watchdog for FastAPI backend: health check + auto restart on pressure/hang."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml


class BackendGuard:
    def __init__(
        self,
        project_root: Path,
        health_url: str,
        check_interval_s: float,
        request_timeout_s: float,
        fail_threshold: int,
        restart_grace_s: float,
    ) -> None:
        self.project_root = project_root
        self.health_url = health_url
        self.check_interval_s = max(0.5, float(check_interval_s))
        self.request_timeout_s = max(0.2, float(request_timeout_s))
        self.fail_threshold = max(1, int(fail_threshold))
        self.restart_grace_s = max(0.5, float(restart_grace_s))

        self._consecutive_failures = 0
        self._stopped = False
        self._process: subprocess.Popen[str] | None = None

        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / "backend_guard.log"

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _start_backend(self) -> None:
        backend_log = self.project_root / "logs" / "backend.log"
        log_fp = backend_log.open("a", encoding="utf-8")
        cmd = [sys.executable, "scripts/run_backend.py"]

        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            self._process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=log_fp,
                stderr=log_fp,
                text=True,
                preexec_fn=os.setsid,
            )
        else:
            self._process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=log_fp,
                stderr=log_fp,
                text=True,
            )

        self.log(f"backend started pid={self._process.pid}")

    def _stop_backend(self) -> None:
        if self._process is None:
            return

        proc = self._process
        if proc.poll() is not None:
            self.log(f"backend already exited code={proc.returncode}")
            self._process = None
            return

        self.log(f"stopping backend pid={proc.pid}")
        try:
            if sys.platform == "darwin" or sys.platform.startswith("linux"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

        deadline = time.time() + self.restart_grace_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)

        if proc.poll() is None:
            self.log(f"backend pid={proc.pid} did not exit in time, force kill")
            try:
                if sys.platform == "darwin" or sys.platform.startswith("linux"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass

        self._process = None

    def _health_ok(self) -> bool:
        try:
            with urlopen(self.health_url, timeout=self.request_timeout_s) as resp:
                return 200 <= int(resp.status) < 300
        except (TimeoutError, URLError, OSError):
            return False

    def _ensure_process(self) -> None:
        if self._process is None:
            self._start_backend()
            return

        code = self._process.poll()
        if code is not None:
            self.log(f"backend exited unexpectedly code={code}, restarting")
            self._process = None
            self._start_backend()

    def loop(self) -> None:
        self._start_backend()

        while not self._stopped:
            self._ensure_process()
            if self._health_ok():
                if self._consecutive_failures > 0:
                    self.log("health check recovered")
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                self.log(f"health check failed ({self._consecutive_failures}/{self.fail_threshold})")
                if self._consecutive_failures >= self.fail_threshold:
                    self.log("health check threshold reached, restarting backend")
                    self._stop_backend()
                    self._start_backend()
                    self._consecutive_failures = 0

            time.sleep(self.check_interval_s)

        self._stop_backend()

    def stop(self) -> None:
        self._stopped = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backend watchdog with auto-restart")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--check-interval", type=float, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--fail-threshold", type=int, default=None)
    parser.add_argument("--restart-grace", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / str(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raw = {}

    guard_cfg = raw.get("backend_guard", {}) if isinstance(raw, dict) else {}
    app_cfg = raw.get("app", {}) if isinstance(raw, dict) else {}
    host = str(app_cfg.get("host", "127.0.0.1"))
    port = int(app_cfg.get("port", 8000))
    fallback_health_url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/api/health"

    guard = BackendGuard(
        project_root=project_root,
        health_url=str(args.health_url or guard_cfg.get("health_url", fallback_health_url)),
        check_interval_s=float(args.check_interval or guard_cfg.get("check_interval_seconds", 3.0)),
        request_timeout_s=float(args.request_timeout or guard_cfg.get("request_timeout_seconds", 1.5)),
        fail_threshold=int(args.fail_threshold or guard_cfg.get("fail_threshold", 3)),
        restart_grace_s=float(args.restart_grace or guard_cfg.get("restart_grace_seconds", 8.0)),
    )

    def _handle_stop(signum: int, _frame: object) -> None:
        guard.log(f"received signal={signum}, stopping guard")
        guard.stop()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    guard.log("backend guard started")
    guard.loop()


if __name__ == "__main__":
    main()
