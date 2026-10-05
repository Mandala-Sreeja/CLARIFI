# ACTIVA AI — Turn Information Into Action

Upload a notice, bill, form or letter. ACTIVA AI classifies it, extracts key details and deadlines,
finds required documents, creates prioritized tasks, and answers questions with page citations (RAG).

## Run in VS Code
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```
1. Open `.env`, set `SECRET_KEY` and `ANTHROPIC_API_KEY`.
2. `python app.py` then open http://127.0.0.1:5000

Without an API key the app still runs using a built-in rule-based analyzer (PDF text only).
Image analysis, simplify/translate and rich Q&A need the key.

## Features
- Sign up / login / logout, hashed passwords, JWT in HttpOnly cookie, protected routes, forgot password, profile
- Dashboard: action counters, today's actions, deadline timeline, recent documents, data-driven AI insights
- Upload PDF/JPG/PNG -> classification, extraction, deadlines, requirements, missing-document detection, tasks
- Document page: key info, tasks, RAG Q&A with page citations, simplify, translate
- Reminder banner for tasks due within 3 days

## Structure
`app.py` (backend + templates) · `assets/style.css` · `uploads/` and `activa.db` are created at runtime.

## Notes
Forgot-password shows the reset link on screen (no email server configured). Add SMTP before production.
