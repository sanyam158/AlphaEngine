@echo off
REM Daily AlphaEngine end-of-day job (used by Windows Task Scheduler).
REM Runs from the repo folder so .env, alphaengine.db and var\data resolve correctly.
cd /d "%~dp0.."
call .venv\Scripts\alphaengine.exe scheduler run-eod >> var\eod.log 2>&1
