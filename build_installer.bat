@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  HROVER Windows Installer Builder
echo ============================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    goto :fail
)
echo [OK] Python found.

:: ── 2. Install / upgrade PyInstaller ─────────────────────────────────────────
echo.
echo [*] Installing PyInstaller...
pip install --quiet --upgrade pyinstaller
if errorlevel 1 (
    echo [ERROR] Failed to install PyInstaller.
    goto :fail
)
echo [OK] PyInstaller ready.

:: ── 3. Install app dependencies ──────────────────────────────────────────────
echo.
echo [*] Installing app dependencies...
pip install --quiet -e ".[gui]"
if errorlevel 1 (
    echo [ERROR] Failed to install app dependencies.
    goto :fail
)
echo [OK] Dependencies ready.

:: ── 4. Run PyInstaller ───────────────────────────────────────────────────────
echo.
echo [*] Bundling app with PyInstaller...
pyinstaller hrover.spec --noconfirm --clean
if errorlevel 1 (
    echo [ERROR] PyInstaller failed.
    goto :fail
)
echo [OK] App bundled to dist\HROVER\

:: ── 5. Run Inno Setup ────────────────────────────────────────────────────────
echo.
echo [*] Looking for Inno Setup...

set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC=C:\Program Files\Inno Setup 6\ISCC.exe
)

if "!ISCC!"=="" (
    echo.
    echo [WARN] Inno Setup 6 not found.
    echo        Download it from: https://jrsoftware.org/isinfo.php
    echo        Then run: "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
    echo.
    echo        The bundled app is already available at: dist\HROVER\HROVER.exe
    goto :done
)

echo [OK] Inno Setup found.
echo.
echo [*] Building installer...
"!ISCC!" installer.iss
if errorlevel 1 (
    echo [ERROR] Inno Setup failed.
    goto :fail
)

echo.
echo ============================================
echo  SUCCESS
echo  Installer: dist\HROVER_Setup_v0.1.0.exe
echo ============================================
goto :done

:fail
echo.
echo ============================================
echo  BUILD FAILED
echo ============================================
exit /b 1

:done
endlocal
