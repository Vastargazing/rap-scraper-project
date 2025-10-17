#!/usr/bin/env python3
"""
🔥 Modern Python Linting Tool
Cross-platform (Windows/Linux/Mac), no encoding issues, clean output.

Usage:
    python lint.py check           # Quick check (dev workflow)
    python lint.py fix             # Auto-fix issues
    python lint.py all             # Full pipeline (check + format + mypy)
    python lint.py all --log       # With file logging for CI/CD
    python lint.py watch           # Watch mode (auto-lint on changes)

Examples:
    python lint.py check           # Fast dev loop
    python lint.py all --log       # CI/CD with history
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class Colors:
    """ANSI colors for terminal output"""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class LintRunner:
    """Modern Python-based linting tool"""

    def __init__(self, log_enabled: bool = False):
        self.log_enabled = log_enabled
        self.log_dir = Path("logs/lint")
        self.log_file: Path | None = None
        self.log_latest: Path | None = None

        # Detect tool paths (venv or system)
        venv_path = Path(".venv/Scripts" if sys.platform == "win32" else ".venv/bin")
        self.ruff = str(venv_path / ("ruff.exe" if sys.platform == "win32" else "ruff"))
        self.mypy = str(venv_path / ("mypy.exe" if sys.platform == "win32" else "mypy"))

        # Fallback to system commands if venv not found
        if not Path(self.ruff).exists():
            self.ruff = "ruff"
        if not Path(self.mypy).exists():
            self.mypy = "mypy"

        if log_enabled:
            self._setup_logging()

    def _setup_logging(self):
        """Setup log directory and files"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.log_file = self.log_dir / f"lint_{timestamp}.log"
        self.log_latest = self.log_dir / "lint_latest.log"

        # Write header
        header = f"""
{"=" * 60}
🔥 RAP SCRAPER LINT RUN
{"=" * 60}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Python: {sys.version.split()[0]}
Platform: {sys.platform}
{"=" * 60}

"""
        self.log_file.write_text(header, encoding="utf-8")
        print(f"{Colors.CYAN}📝 Logging to: {self.log_file}{Colors.RESET}")

    def _log(self, message: str):
        """Write message to log file"""
        if self.log_enabled and self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def _run_command(self, name: str, args: list[str]) -> bool:
        """
        Run a command and return success status.

        Args:
            name: Display name of the command
            args: Command arguments

        Returns:
            True if command succeeded, False otherwise
        """
        print(f"{Colors.CYAN}🔍 Running: {name}...{Colors.RESET}")

        if self.log_enabled:
            self._log(f"[{name}] Running: {' '.join(args)}")

        try:
            # Run command and capture output
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Handle any encoding issues gracefully
            )

            # Print output
            output = result.stdout + result.stderr
            if output.strip():
                print(output)

            # Log output
            if self.log_enabled:
                self._log(output)

            # Check status
            if result.returncode != 0:
                msg = f"❌ FAILED: {name}"
                print(f"{Colors.RED}{msg}{Colors.RESET}")
                if self.log_enabled:
                    self._log(msg)
                return False

            msg = f"✅ PASSED: {name}"
            print(f"{Colors.GREEN}{msg}{Colors.RESET}")
            if self.log_enabled:
                self._log(msg)
            return True

        except FileNotFoundError:
            msg = f"❌ ERROR: Command not found: {args[0]}"
            print(f"{Colors.RED}{msg}{Colors.RESET}")
            if self.log_enabled:
                self._log(msg)
            return False
        except Exception as e:
            msg = f"❌ ERROR: {e}"
            print(f"{Colors.RED}{msg}{Colors.RESET}")
            if self.log_enabled:
                self._log(msg)
            return False

    def _finalize_log(self):
        """Copy log to latest.log"""
        if self.log_enabled and self.log_file and self.log_latest:
            import shutil

            shutil.copy(self.log_file, self.log_latest)
            print(f"{Colors.CYAN}📝 Log saved: {self.log_file}{Colors.RESET}")
            print(f"{Colors.CYAN}📌 Latest: {self.log_latest}{Colors.RESET}")

    def check(self):
        """Run Ruff check only"""
        print(f"{Colors.YELLOW}📋 Linting with Ruff...{Colors.RESET}")
        success = self._run_command("Ruff Check", [self.ruff, "check", "."])
        self._finalize_log()
        return success

    def fix(self):
        """Run Ruff with auto-fix and formatting"""
        print(f"{Colors.YELLOW}🔧 Fixing with Ruff...{Colors.RESET}")
        success = self._run_command("Ruff Fix", [self.ruff, "check", "--fix", "."])
        print()
        success = (
            self._run_command("Ruff Format", [self.ruff, "format", "."]) and success
        )
        self._finalize_log()
        return success

    def format_only(self):
        """Run Ruff format only"""
        print(f"{Colors.YELLOW}🎨 Formatting with Ruff...{Colors.RESET}")
        success = self._run_command("Ruff Format", [self.ruff, "format", "."])
        self._finalize_log()
        return success

    def mypy_check(self):
        """Run Mypy type checking"""
        print(f"{Colors.YELLOW}🔍 Type checking with Mypy...{Colors.RESET}")
        success = self._run_command("Mypy Check", [self.mypy, "."])
        self._finalize_log()
        return success

    def all(self):
        """Run full lint pipeline"""
        print(f"{Colors.MAGENTA}🚀 Running full lint pipeline...{Colors.RESET}")

        all_passed = True

        # Ruff check
        if not self._run_command("Ruff Check", [self.ruff, "check", "."]):
            all_passed = False
        print()

        # Ruff format
        if not self._run_command("Ruff Format", [self.ruff, "format", "."]):
            all_passed = False
        print()

        # Mypy
        if not self._run_command("Mypy Check", [self.mypy, "."]):
            all_passed = False
        print()

        # Final status
        if all_passed:
            print(f"{Colors.GREEN}🎉 All checks passed!{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️  Some checks found issues{Colors.RESET}")

        self._finalize_log()
        return all_passed

    def watch(self):
        """Watch mode - auto-lint on file changes"""
        print(
            f"{Colors.MAGENTA}👀 Watching for changes (Ctrl+C to stop)...{Colors.RESET}"
        )

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class PythonFileHandler(FileSystemEventHandler):
                def __init__(self, runner):
                    self.runner = runner
                    self.last_run = {}

                def on_modified(self, event):
                    if event.src_path.endswith(".py"):
                        # Debounce: only run if 1 second passed since last run
                        now = time.time()
                        if (
                            event.src_path not in self.last_run
                            or now - self.last_run[event.src_path] > 1
                        ):
                            print(
                                f"\n{Colors.YELLOW}📝 File changed: {event.src_path}{Colors.RESET}"
                            )
                            subprocess.run(
                                [self.ruff, "check", event.src_path], check=False
                            )
                            self.last_run[event.src_path] = now

            observer = Observer()
            observer.schedule(PythonFileHandler(self), ".", recursive=True)
            observer.start()

            print(f"{Colors.CYAN}⏳ Watching for changes...{Colors.RESET}")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        except ImportError:
            print(
                f"{Colors.RED}❌ watchdog not installed. Install with: pip install watchdog{Colors.RESET}"
            )
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🔥 Modern Python Linting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lint.py check           # Quick check (dev workflow)
  python lint.py fix             # Auto-fix issues
  python lint.py all             # Full pipeline
  python lint.py all --log       # With file logging (CI/CD)
  python lint.py watch           # Watch mode
        """,
    )

    parser.add_argument(
        "command",
        choices=["check", "fix", "format", "mypy", "all", "watch"],
        help="Linting command to run",
    )

    parser.add_argument(
        "--log", action="store_true", help="Enable file logging (useful for CI/CD)"
    )

    args = parser.parse_args()

    # Create runner
    runner = LintRunner(log_enabled=args.log)

    # Execute command
    command_map = {
        "check": runner.check,
        "fix": runner.fix,
        "format": runner.format_only,
        "mypy": runner.mypy_check,
        "all": runner.all,
        "watch": runner.watch,
    }

    success = command_map[args.command]()

    # Print tips
    print(f"\n{Colors.CYAN}📌 Tips:{Colors.RESET}")
    print(f"{Colors.GRAY}  • Fast dev: python lint.py check{Colors.RESET}")
    print(f"{Colors.GRAY}  • With logs: python lint.py all --log{Colors.RESET}")
    print(f"{Colors.GRAY}  • CI/CD: Always use --log flag for history{Colors.RESET}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
