@echo off
cd /d "%~dp0"
uv run f5-audiobook
if errorlevel 1 pause
