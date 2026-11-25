@echo off
title JournalistAI Studio - Server
echo.
echo  ██╗ ██████╗ ██╗   ██╗██████╗ ███╗   ██╗ █████╗ ██╗     ██╗███████╗████████╗ █████╗ ██╗
echo  ██║██╔═══██╗██║   ██║██╔══██╗████╗  ██║██╔══██╗██║     ██║██╔════╝╚══██╔══╝██╔══██╗██║
echo  ██║██║   ██║██║   ██║██████╔╝██╔██╗ ██║███████║██║     ██║███████╗   ██║   ███████║██║
echo  ██║██║   ██║██║   ██║██╔══██╗██║╚██╗██║██╔══██║██║     ██║╚════██║   ██║   ██╔══██║██║
echo  ██║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║██║  ██║███████╗██║███████║   ██║   ██║  ██║██║
echo  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝
echo.
echo  AI-Powered Interview Transcription + Fact-Checking for Investigative Journalism
echo  ─────────────────────────────────────────────────────────────────────────────────
echo.
echo  Starting server...
echo  Open your browser at: http://127.0.0.1:8000
echo  API Docs at:          http://127.0.0.1:8000/api/docs
echo.
echo  Press Ctrl+C to stop the server.
echo.

cd /d "%~dp0"
call .venv\Scripts\activate
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause

