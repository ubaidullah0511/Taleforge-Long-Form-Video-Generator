"""System tray launcher for the Clip Generation server.

Compiled to ClipGeneration.exe via PyInstaller -- see requirements-launcher.txt
for build-time deps and installer.nsi for how the installer's shortcuts point
at the compiled exe. Replaces launcher.vbs's fully-invisible approach: this
keeps a tray icon alive for the life of the server so there's something the
user can see and click "Exit" on, which is also what guarantees the uvicorn
child process is terminated instead of left running in the background.

A tray icon was chosen over an always-visible window because this is a
background service (a local web server the user drives from a browser tab,
not from this process's own UI) -- a window would either have to stay open
uselessly on the taskbar or, if closed, ambiguously either quit or hide.
A tray icon has no such ambiguity: it's out of the way until wanted, and its
one menu action ("Exit") means exactly one thing.
"""

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser

import pystray
from PIL import Image

PORT = 8001
URL = f"http://localhost:{PORT}"
MB_ICONERROR = 0x10


def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _resource_path(name):
    # PyInstaller --onefile extracts bundled --add-data files (icon.ico) to
    # a temp dir at sys._MEIPASS. Installed app files (setup.bat, .venv)
    # live next to the exe instead -- that's BASE_DIR, not this.
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, name)
    return os.path.join(BASE_DIR, name)


BASE_DIR = _base_dir()
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
SETUP_BAT = os.path.join(BASE_DIR, "setup.bat")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "server.log")

server_process = None
server_lock = threading.Lock()


def show_error(title, message):
    ctypes.windll.user32.MessageBoxW(0, message, title, MB_ICONERROR)


def run_setup_if_needed():
    if os.path.isfile(VENV_PYTHON):
        return True
    # First run: no .venv yet. .bat files aren't directly executable via
    # CreateProcess -- they need cmd.exe as the interpreter -- so run it
    # through cmd /c. This process has no console of its own (built
    # --windowed), so that gets setup.bat a fresh visible console
    # automatically, same visible-progress behavior launcher.vbs had.
    subprocess.call(["cmd.exe", "/c", SETUP_BAT], cwd=BASE_DIR)
    if not os.path.isfile(VENV_PYTHON):
        show_error(
            "Clip Generation - Setup Failed",
            "Setup did not finish successfully -- the Python environment (.venv) "
            "was not created.\n\nCheck the setup window for error messages and try again.",
        )
        return False
    return True


def start_server():
    global server_process
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_handle = open(LOG_FILE, "a")
    log_handle.write(
        f"\n---------------------------------------------\n"
        f"[{time.ctime()}] Launching server (tray app)\n"
    )
    log_handle.flush()
    with server_lock:
        server_process = subprocess.Popen(
            [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(PORT)],
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return log_handle


def stop_server(timeout=10):
    global server_process
    with server_lock:
        proc = server_process
        server_process = None
    if proc is None or proc.poll() is not None:
        return
    # Try a graceful shutdown first (uvicorn installs a SIGBREAK handler on
    # Windows, which is what CTRL_BREAK_EVENT raises in the child), then
    # fall back to a hard kill so the process is never left orphaned.
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        proc.wait(timeout=timeout)
        return
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def on_open_browser(icon, item):
    webbrowser.open(URL)


def on_exit(icon, item):
    stop_server()
    icon.stop()


def wait_and_open_browser():
    time.sleep(3)
    webbrowser.open(URL)


def main():
    if not run_setup_if_needed():
        sys.exit(1)

    log_handle = start_server()
    threading.Thread(target=wait_and_open_browser, daemon=True).start()

    image = Image.open(_resource_path("icon.ico"))
    menu = pystray.Menu(
        pystray.MenuItem("Open in Browser", on_open_browser, default=True),
        pystray.MenuItem("Exit", on_exit),
    )
    icon = pystray.Icon("ClipGeneration", image, "Clip Generation Server", menu)

    try:
        icon.run()
    finally:
        # Belt-and-suspenders: guarantees the uvicorn child is gone even if
        # icon.run() ever returns through a path other than the Exit item.
        stop_server()
        try:
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
