; ============================================================================
; Clip Generation - Windows Installer (NSIS)
;
; Builds a Windows installer that:
;   - Copies the application's source files to Program Files (or a
;     user-chosen directory), WITHOUT bundling generated/large directories
;     (.venv, node_modules, projects/, downloaded clips, chroma DB, etc.)
;     -- those get created fresh on first run by setup.bat.
;   - Copies the pre-built ClipGeneration.exe (see BUILD STEPS below) as
;     the app's launcher, and creates a Desktop shortcut and a Start Menu
;     shortcut pointing at it directly.
;   - Registers a standard uninstaller in Add/Remove Programs.
;
; First-run behavior lives in ClipGeneration.exe (built from app_tray.py),
; not here: the very first launch detects that .venv doesn't exist yet,
; runs setup.bat (GPU-aware dependency install, shown with visible
; progress), and only then starts the server and shows a system tray icon.
; Every launch after that skips straight to starting the server, since
; setup is already done. Closing via the tray icon's "Exit" item cleanly
; terminates the server -- see app_tray.py for details.
;
; ----------------------------------------------------------------------
; HOW TO COMPILE (on a real Windows machine):
;   1. Build ClipGeneration.exe first (this script packages it, it does
;      not build it):
;        .venv\Scripts\pip install -r requirements-launcher.txt
;        .venv\Scripts\pyinstaller --onefile --windowed --icon=icon.ico ^
;            --add-data "icon.ico;." --name=ClipGeneration app_tray.py
;      This produces dist\ClipGeneration.exe.
;   2. Install NSIS (https://nsis.sourceforge.io/Download) -- this gives
;      you makensis.exe, typically at:
;        C:\Program Files (x86)\NSIS\makensis.exe
;   3. Place this file (installer.nsi) in the ROOT of the clip_generation
;      project folder (the same folder that contains app/, remotion/,
;      requirements.txt, setup.bat, dist\ClipGeneration.exe, etc.) --
;      this script packages files relative to its own location.
;   4. From that folder, run:
;        "C:\Program Files (x86)\NSIS\makensis.exe" installer.nsi
;      or, if makensis.exe is on PATH:
;        makensis installer.nsi
;   5. This produces ClipGeneration-Setup.exe in the same folder.
;
; This script has been compiled with makensis 3.x and smoke-tested end to
; end on this machine (install, launch shortcut, confirm server starts and
; stops cleanly, uninstall). See the accompanying report for what was
; actually verified.
; ============================================================================

!define APP_NAME        "Clip Generation"
!define APP_VERSION      "1.0.0"
!define APP_PUBLISHER   "Clip Generation"
!define APP_UNINST_KEY   "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClipGeneration"
!define APP_INSTALL_DIR  "$PROGRAMFILES64\ClipGeneration"

Name "${APP_NAME}"
OutFile "ClipGeneration-Setup.exe"
InstallDir "${APP_INSTALL_DIR}"
InstallDirRegKey HKLM "${APP_UNINST_KEY}" "InstallLocation"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

; ----------------------------------------------------------------------
; Modern UI
; ----------------------------------------------------------------------
!include "MUI2.nsh"

!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\ClipGeneration.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Run first-time setup and launch Clip Generation now"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ----------------------------------------------------------------------
; Install section
; ----------------------------------------------------------------------
Section "Application Files (required)" SEC_APP
    SectionIn RO
    SetOutPath "$INSTDIR"

    ; Recursively copy everything from the project root EXCEPT generated
    ; / large / machine-specific directories and files. Those are created
    ; fresh at first run (setup.bat builds .venv; the app creates its own
    ; data/projects/clips/uploads folders on demand) rather than shipped.
    ;
    ; NOTE: .env is deliberately excluded -- it holds this developer's own
    ; API keys and must never be bundled into a distributable installer.
    ; Only .env.example ships; each install needs its own .env.
    ;
    ; "dist" / "build" / "*.spec" are PyInstaller's output and intermediate
    ; directories for ClipGeneration.exe -- the exe itself is copied
    ; explicitly below instead of via this recursive glob, since dist\
    ; also accumulates stale intermediate files across rebuilds.
    File /r /x ".git" /x ".venv" /x "node_modules" /x ".remotion" /x "projects" /x "clips" /x "__pycache__" /x "*.pyc" /x ".pytest_cache" /x ".cache" /x ".claude" /x ".env" /x "uploads" /x "chroma" /x "*.log" /x "ClipGeneration-Setup.exe" /x "dist" /x "build" /x "*.spec" "*.*"

    ; The compiled tray launcher (see BUILD STEPS above) -- must exist at
    ; dist\ClipGeneration.exe before running makensis.
    File "dist\ClipGeneration.exe"

    ; Recreate the empty runtime directories the app expects to find
    ; (their contents are gitignored/excluded above, but the app assumes
    ; the folders themselves exist).
    CreateDirectory "$INSTDIR\clips\local"
    CreateDirectory "$INSTDIR\clips\downloaded"
    CreateDirectory "$INSTDIR\projects"
    CreateDirectory "$INSTDIR\uploads"
    CreateDirectory "$INSTDIR\data\chroma"
    CreateDirectory "$INSTDIR\logs"

    ; ------------------------------------------------------------------
    ; Shortcuts -- both target ClipGeneration.exe directly (it carries its
    ; own icon). First launch triggers setup.bat automatically (visible,
    ; so install progress is shown); every launch after that goes straight
    ; to starting the server, opening the browser, and showing a tray icon.
    ; ------------------------------------------------------------------
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\ClipGeneration.exe" "" "$INSTDIR\ClipGeneration.exe" 0 SW_SHOWNORMAL "" "Launch Clip Generation"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe"
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\ClipGeneration.exe" "" "$INSTDIR\ClipGeneration.exe" 0 SW_SHOWNORMAL "" "Launch Clip Generation"

    ; ------------------------------------------------------------------
    ; Uninstaller + Add/Remove Programs registration
    ; ------------------------------------------------------------------
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    WriteRegStr HKLM "${APP_UNINST_KEY}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "${APP_UNINST_KEY}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKLM "${APP_UNINST_KEY}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKLM "${APP_UNINST_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "${APP_UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegStr HKLM "${APP_UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
    WriteRegDWORD HKLM "${APP_UNINST_KEY}" "NoModify" 1
    WriteRegDWORD HKLM "${APP_UNINST_KEY}" "NoRepair" 1
SectionEnd

; ----------------------------------------------------------------------
; Uninstall section
; ----------------------------------------------------------------------
Section "Uninstall"
    ; Stop a running server first, if any, so files aren't locked. Uses a
    ; dedicated helper batch file (rather than an inline command) to avoid
    ; fragile quote/escape nesting inside the .nsi script itself.
    IfFileExists "$INSTDIR\_uninstall_stop_server.bat" 0 +2
        ExecWait '"$INSTDIR\_uninstall_stop_server.bat"'

    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    Delete "$DESKTOP\${APP_NAME}.lnk"

    ; Application source files (safe to remove unconditionally).
    RMDir /r "$INSTDIR\app"
    RMDir /r "$INSTDIR\remotion\src"
    Delete "$INSTDIR\remotion\package.json"
    Delete "$INSTDIR\remotion\package-lock.json"
    Delete "$INSTDIR\remotion\render.mjs"
    Delete "$INSTDIR\remotion\tsconfig.json"
    RMDir "$INSTDIR\remotion"
    RMDir /r "$INSTDIR\templates"
    RMDir /r "$INSTDIR\tests"
    RMDir /r "$INSTDIR\scripts"
    RMDir /r "$INSTDIR\docs"
    Delete "$INSTDIR\*.bat"
    Delete "$INSTDIR\*.vbs"
    Delete "$INSTDIR\ClipGeneration.exe"
    Delete "$INSTDIR\icon.ico"
    Delete "$INSTDIR\*.md"
    Delete "$INSTDIR\*.txt"
    Delete "$INSTDIR\.env.example"
    Delete "$INSTDIR\.gitignore"
    Delete "$INSTDIR\Uninstall.exe"

    ; Generated/user data (.venv, downloaded clips, rendered projects,
    ; the vector DB, uploads, logs) can be large and may contain output
    ; the user wants to keep -- ask before deleting rather than silently
    ; wiping it.
    MessageBox MB_YESNO|MB_ICONQUESTION \
        "Also remove generated data (Python environment, downloaded clips, rendered projects, logs)?$\r$\n$\r$\nChoose No to keep them in place at:$\r$\n$INSTDIR" \
        IDNO skip_data_removal
        RMDir /r "$INSTDIR\.venv"
        RMDir /r "$INSTDIR\clips"
        RMDir /r "$INSTDIR\projects"
        RMDir /r "$INSTDIR\uploads"
        RMDir /r "$INSTDIR\data"
        RMDir /r "$INSTDIR\logs"
    skip_data_removal:

    ; Only remove the install directory itself if it ended up empty
    ; (i.e. the user chose to remove generated data above, or never
    ; had any). If they kept their data, $INSTDIR still has content and
    ; RMDir without /r will simply no-op rather than fail loudly.
    RMDir "$INSTDIR"

    DeleteRegKey HKLM "${APP_UNINST_KEY}"
SectionEnd
