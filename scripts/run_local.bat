@echo off
REM Double-click helper for Windows: creates venv (first time) and starts the app.
if not exist venv (
  python -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
if not exist .env copy .env.example .env
echo.
echo IMPORTANT: open the file .env in Notepad and set ADMIN_PASSWORD before logging in.
echo.
streamlit run app.py
