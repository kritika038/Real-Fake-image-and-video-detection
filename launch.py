import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
