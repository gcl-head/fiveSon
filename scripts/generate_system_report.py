"""Convenience wrapper for generating system_report.md."""

from __future__ import annotations

from tools.system_probe import main as probe_main

if __name__ == "__main__":
    # Delegates to argparse in system_probe; default output is system_report.md.
    probe_main()
