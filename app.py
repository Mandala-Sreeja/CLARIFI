"""ACTIVA AI - Turn Information Into Action."""
import os, re, json, base64, sqlite3, datetime as dt
from functools import wraps
import jwt
from dotenv import load_dotenv
from flask import Flask, request, redirect, render_template, g, make_response, jsonify, flash, url_for
from jinja2 import DictLoader
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pypdf import PdfReader

load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
SECRET = os.getenv("SECRET_KEY", "dev-secret")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
UPLOADS = os.path.join(BASE, "uploads"); os.makedirs(UPLOADS, exist_ok=True)
DB = os.path.join(BASE, "activa.db")
DOC_TYPES = ["College Notice", "Internship Notice", "Scholarship Notice", "Job Offer", "Invoice", "Bill",
             "Warranty", "Rental Agreement", "School Circular", "Certificate", "Government Notice", "Other"]
app = Flask(__name__, static_folder="assets", static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
from google import genai
client=None
if os.getenv("GEMINI_API_KEY"):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db
@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()
def init_db():
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT);
    CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, user_id INT, title TEXT, doc_type TEXT, summary TEXT,
      info TEXT, pages TEXT, created TEXT, status TEXT, deadlines INT DEFAULT 0);
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, user_id INT, doc_id INT, title TEXT, deadline TEXT,
      status TEXT DEFAULT 'pending', effort TEXT, depends_on TEXT, kind TEXT DEFAULT 'action');""")
    c.commit(); c.close()

# ---------- auth (JWT in HttpOnly cookie) ----------
def make_token(uid, mins=60 * 24 * 7, purpose="auth"):
    return jwt.encode({"uid": uid, "p": purpose, "exp": dt.datetime.utcnow() + dt.timedelta(minutes=mins)}, SECRET, "HS256")
def read_token(t, purpose="auth"):
    try:
        d = jwt.decode(t, SECRET, ["HS256"]); return d["uid"] if d["p"] == purpose else None
    except Exception:
        return None
def login_required(f):
    @wraps(f)
    def w(*a, **k):
        uid = read_token(request.cookies.get("token", ""))
        g.user = db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone() if uid else None
        return f(*a, **k) if g.user else redirect(url_for("login"))
    return w

# ---------- AI layer ----------
from google.genai import types

def llm(prompt, image=None, system="You are ACTIVA AI, a precise document-action assistant."):

    if client is None:
        raise RuntimeError("Gemini API key is missing. Check your .env file.")

    contents = [prompt]

    if image:
        image_part = types.Part.from_bytes(
            data=image[1],
            mime_type=image[0]
        )
        contents.append(image_part)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=2000
        )
    )

    return response.text or ""

def parse_json(t):
    return json.loads(re.search(r"\{.*\}", t, re.S).group(0))

def rule_based(text):
    low = text.lower()
    kinds = {"Invoice": ["invoice", "gst"], "Bill": ["bill", "amount due"], "Scholarship Notice": ["scholarship"],
             "Internship Notice": ["internship"], "Job Offer": ["offer letter", "ctc"], "Warranty": ["warranty"],
             "Rental Agreement": ["rent", "tenant"], "College Notice": ["college", "exam", "circular"]}
    dtype = next((k for k, w in kinds.items() if any(x in low for x in w)), "Other")
    dates = []
    for m in re.finditer(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})|([A-Z][a-z]+) (\d{1,2}),? (\d{4})", text):
        try:
            d = dt.date(int(m[3]), int(m[2]), int(m[1])) if m[1] else dt.datetime.strptime(f"{m[4][:3]} {m[5]} {m[6]}", "%b %d %Y").date()
            dates.append(d.isoformat())
        except ValueError:
            pass
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\b(submit|pay|upload|bring|attach|register|obtain)\b", s, re.I)]
    tasks = [{"title": s[:110], "deadline": dates[i] if i < len(dates) else None, "effort": "", "depends_on": ""} for i, s in enumerate(sents[:6])]
    return {"doc_type": dtype, "summary": text[:300].replace("\n", " "), "key_info": {"Dates found": ", ".join(dates) or "None"},
            "required_documents": [], "tasks": tasks}

def analyze(text, image=None):
    if not client:
        return rule_based(text)
    prompt = f"""Analyze this document. Return ONLY JSON with keys:
doc_type (one of {DOC_TYPES}), summary (2 sentences), key_info (object of label->value),
required_documents (list of strings), tasks (list of objects: title, deadline as YYYY-MM-DD or null,
effort e.g. '30 min', depends_on short text or ''), transcript (full text if input is an image, else '').
Today is {dt.date.today()}.
Document:
{text[:12000]}"""
    return parse_json(llm(prompt, image))

def ask_doc(pages, q):
    words = set(re.findall(r"\w{3,}", q.lower()))
    scored = sorted(range(len(pages)), key=lambda i: -len(words & set(re.findall(r"\w{3,}", pages[i].lower()))))[:3]
    ctx = "\n\n".join(f"[Page {i+1}]\n{pages[i][:3000]}" for i in sorted(scored))
    if not client:
        return f"(Add an API key for full answers.) Most relevant text, page {scored[0]+1}:\n{pages[scored[0]][:500]}"
    return llm(f"Answer ONLY from the context. Cite pages like (Page 2). If absent, say so.\n\nContext:\n{ctx}\n\nQuestion: {q}")

def priority(deadline):
    if not deadline: return "low"
    days = (dt.date.fromisoformat(deadline) - dt.date.today()).days
    return "high" if days <= 2 else "medium" if days <= 7 else "low"
def due_label(deadline):
    if not deadline: return "No deadline"
    d = (dt.date.fromisoformat(deadline) - dt.date.today()).days
    return "Overdue" if d < 0 else "Due today" if d == 0 else "Due tomorrow" if d == 1 else f"Due in {d} days"
app.jinja_env.globals.update(priority=priority, due_label=due_label)

# ---------- routes: public + auth ----------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        f = request.form
        if len(f["password"]) < 8: flash("Use a password of at least 8 characters."); return render_template("signup.html")
        try:
            cur = db().execute("INSERT INTO users(name,email,pw) VALUES(?,?,?)", (f["name"], f["email"].lower(), generate_password_hash(f["password"])))
            db().commit()
        except sqlite3.IntegrityError:
            flash("That email already has an account. Log in instead."); return render_template("signup.html")
        r = make_response(redirect(url_for("dashboard"))); r.set_cookie("token", make_token(cur.lastrowid), httponly=True, samesite="Lax"); return r
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?", (request.form["email"].lower(),)).fetchone()
        if u and check_password_hash(u["pw"], request.form["password"]):
            r = make_response(redirect(url_for("dashboard"))); r.set_cookie("token", make_token(u["id"]), httponly=True, samesite="Lax"); return r
        flash("Email or password is incorrect.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    r = make_response(redirect(url_for("landing"))); r.delete_cookie("token"); return r

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        u = db().execute("SELECT id FROM users WHERE email=?", (request.form["email"].lower(),)).fetchone()
        if u:  # no email server: show link (dev). Replace with SMTP in production.
            flash("Reset link (valid 30 min): " + url_for("reset", token=make_token(u["id"], 30, "reset"), _external=True))
        else:
            flash("If that email exists, a reset link has been created.")
    return render_template("forgot.html")

@app.route("/reset/<token>", methods=["GET", "POST"])
def reset(token):
    uid = read_token(token, "reset")
    if not uid: flash("This reset link is invalid or expired."); return redirect(url_for("forgot"))
    if request.method == "POST" and len(request.form["password"]) >= 8:
        db().execute("UPDATE users SET pw=? WHERE id=?", (generate_password_hash(request.form["password"]), uid)); db().commit()
        flash("Password updated. Log in with your new password."); return redirect(url_for("login"))
    return render_template("reset.html")

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        db().execute("UPDATE users SET name=? WHERE id=?", (request.form["name"], g.user["id"])); db().commit()
        flash("Profile saved."); return redirect(url_for("profile"))
    return render_template("profile.html")

# ---------- routes: app ----------
@app.route("/dashboard")
@login_required
def dashboard():
    d, uid = db(), g.user["id"]
    tasks = d.execute("SELECT t.*, docs.title AS doc FROM tasks t JOIN docs ON docs.id=t.doc_id WHERE t.user_id=? ORDER BY deadline IS NULL, deadline", (uid,)).fetchall()
    pend = [t for t in tasks if t["status"] == "pending"]
    urgent = [t for t in pend if priority(t["deadline"]) == "high"]
    week = [t for t in pend if t["deadline"] and (dt.date.fromisoformat(t["deadline"]) - dt.date.today()).days <= 7]
    missing = [t for t in pend if t["kind"] == "document"]
    stats = dict(urgent=len(urgent), upcoming=len([t for t in pend if t["deadline"]]), pending_docs=len(missing),
                 done=len(tasks) - len(pend), docs=d.execute("SELECT COUNT(*) FROM docs WHERE user_id=?", (uid,)).fetchone()[0])
    insights = []
    if week: insights.append(f"You have {len(week)} deadline{'s' if len(week) > 1 else ''} in the next 7 days.")
    for t in missing[:3]: insights.append(f"You are missing a document: {t['title'].replace('Obtain ', '')} (needed for {t['doc']}).")
    for t in pend:
        if t["depends_on"]: insights.append(f"“{t['title']}” depends on: {t['depends_on']}."); break
    if tasks and not pend: insights.append("All tasks are complete. Nice work.")
    hour = dt.datetime.now().hour
    hello = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    reminders = [t for t in pend if t["deadline"] and 0 <= (dt.date.fromisoformat(t["deadline"]) - dt.date.today()).days <= 3]
    docs = d.execute("SELECT * FROM docs WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,)).fetchall()
    return render_template("dashboard.html", hello=hello, stats=stats, tasks=pend[:8] + [t for t in tasks if t["status"] == "done"][:3],
                           timeline=[t for t in pend if t["deadline"]][:8], docs=docs, insights=insights, reminders=reminders)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET": return render_template("upload.html")
    f = request.files.get("file")
    ext = (f.filename.rsplit(".", 1)[-1].lower() if f and "." in f.filename else "")
    if ext not in ("pdf", "jpg", "jpeg", "png"): return jsonify(error="Upload a PDF, JPG or PNG file."), 400
    name = secure_filename(f.filename); path = os.path.join(UPLOADS, f"{g.user['id']}_{int(dt.datetime.now().timestamp())}_{name}")
    f.save(path)
    try:
        image = None
        if ext == "pdf":
            pages = [(p.extract_text() or "") for p in PdfReader(path).pages]
            if not "".join(pages).strip(): return jsonify(error="No readable text in this PDF. Upload it as an image instead."), 400
            data = analyze("\n".join(f"[Page {i+1}] {p}" for i, p in enumerate(pages)))
        else:
            if not client: return jsonify(error="Image analysis needs ANTHROPIC_API_KEY in .env."), 400
            image = ("image/png" if ext == "png" else "image/jpeg", base64.b64encode(open(path, "rb").read()).decode())
            data = analyze("(see attached image)", image); pages = [data.get("transcript") or ""]
    except Exception as e:
        return jsonify(error=f"Could not analyze the document: {e}"), 500
    d, uid = db(), g.user["id"]
    tasks = data.get("tasks", [])
    cur = d.execute("INSERT INTO docs(user_id,title,doc_type,summary,info,pages,created,status,deadlines) VALUES(?,?,?,?,?,?,?,?,?)",
                    (uid, name, data.get("doc_type", "Other"), data.get("summary", ""), json.dumps(data.get("key_info", {})),
                     json.dumps(pages), dt.date.today().isoformat(), "Analyzed", sum(1 for t in tasks if t.get("deadline"))))
    did = cur.lastrowid
    for t in tasks:
        d.execute("INSERT INTO tasks(user_id,doc_id,title,deadline,effort,depends_on) VALUES(?,?,?,?,?,?)",
                  (uid, did, t.get("title", "Task"), t.get("deadline"), t.get("effort", ""), t.get("depends_on", "")))
    have = " ".join(r[0].lower() + " " + (r[1] or "").lower() for r in d.execute("SELECT title,doc_type FROM docs WHERE user_id=? AND id<>?", (uid, did)))
    for req in data.get("required_documents", []):  # compare against what the user already uploaded
        if not any(w in have for w in re.findall(r"\w{4,}", req.lower())):
            d.execute("INSERT INTO tasks(user_id,doc_id,title,kind) VALUES(?,?,?,'document')", (uid, did, f"Obtain {req}"))
    d.commit()
    return jsonify(redirect=url_for("document", doc_id=did))

@app.route("/documents/<int:doc_id>")
@login_required
def document(doc_id):
    doc = db().execute("SELECT * FROM docs WHERE id=? AND user_id=?", (doc_id, g.user["id"])).fetchone()
    if not doc: return redirect(url_for("dashboard"))
    tasks = db().execute("SELECT * FROM tasks WHERE doc_id=? ORDER BY deadline IS NULL, deadline", (doc_id,)).fetchall()
    return render_template("document.html", doc=doc, info=json.loads(doc["info"]), tasks=tasks)

@app.route("/documents/<int:doc_id>/ai", methods=["POST"])
@login_required
def doc_ai(doc_id):
    doc = db().execute("SELECT * FROM docs WHERE id=? AND user_id=?", (doc_id, g.user["id"])).fetchone()
    if not doc: return jsonify(error="Not found"), 404
    pages, mode = json.loads(doc["pages"]), request.json.get("mode")
    try:
        if mode == "ask": return jsonify(answer=ask_doc(pages, request.json["q"]))
        if not client: return jsonify(answer="Simplify and translate need ANTHROPIC_API_KEY in .env.")
        text = "\n".join(pages)[:12000]
        if mode == "simplify": return jsonify(answer=llm(f"Explain this document in simple, plain language for a non-expert:\n\n{text}"))
        return jsonify(answer=llm(f"Translate this document summary into {request.json.get('lang', 'Hindi')}:\n\n{doc['summary']}\n\n{text[:4000]}"))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/tasks/<int:tid>/toggle", methods=["POST"])
@login_required
def toggle(tid):
    db().execute("UPDATE tasks SET status=CASE status WHEN 'done' THEN 'pending' ELSE 'done' END WHERE id=? AND user_id=?", (tid, g.user["id"]))
    db().commit(); return redirect(request.referrer or url_for("dashboard"))

# ---------- templates (inline, so the project stays flat) ----------
NAV = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>CLARIFI</title><link rel=stylesheet href="/assets/style.css"></head><body>
<nav><b>CLARIFI</b>{% if g.user %}<a href="/dashboard">Dashboard</a><a href="/upload">Upload</a><a href="/profile">{{ g.user['name'] }}</a><a href="/logout">Log out</a>
{% else %}<a href="/#how">How it works</a><a href="/login">Log in</a><a class=btn href="/signup">Get Started</a>{% endif %}</nav>
<main>{% for m in get_flashed_messages() %}<div class=flash>{{ m }}</div>{% endfor %}"""
LAYOUT = NAV + "{% block c %}{% endblock %}</main></body></html>"
def page(body): return "{% extends 'layout.html' %}{% block c %}" + body + "{% endblock %}"
def form(title, fields, extra=""):
    inputs = "".join(f'<label>{l}<input name={n} type={t} required></label>' for n, l, t in fields)
    return page(f"<h1>{title}</h1><form class=stack method=post>{inputs}<button class=btn>{title}</button></form>{extra}")
FEATS = [("Understands any notice", "Bills, circulars, offer letters, agreements and scans are classified and summarized."),
         ("Finds every deadline", "Dates become a timeline ranked by urgency."),
         ("Spots missing documents", "Required papers are compared with what you have already uploaded."),
         ("Answers with sources", "Ask a question and get an answer with the page it came from."),
         ("Explains in plain language", "Simplify any document or translate it."),
         ("Reminds you in time", "Tasks due within 3 days appear as reminders on your dashboard.")]
T = {
 "layout.html": LAYOUT,
 "landing.html": page("""<section class=hero><h1>CLARIFI</h1><h2>Turn Information Into Action</h2>
<p>Upload a document, notice, bill, form, or important message. CLARIFI understands it, finds what matters, and turns it into clear tasks, deadlines, and next steps.</p>
<a class=btn href=/signup>Get Started</a> <a class="btn alt" href="#how">See How It Works</a></section>
<h2 id=how>How it works</h2><div class=grid>""" + "".join(f"<div class=card><h3>{a}</h3><p class=mute>{b}</p></div>" for a, b in FEATS) + """</div>
<h2>Built for</h2><p>Students with scholarship and exam notices. Parents with school circulars. Employees with offer letters. Households and small businesses with bills, invoices and agreements.</p>
<h2>Security and privacy</h2><p>Passwords are hashed, sessions use signed JWTs in HttpOnly cookies, and every document is visible only to its owner.</p>
<p><a class=btn href=/signup>Start with your first document</a></p>"""),
 "signup.html": form("Create account", [("name", "Name", "text"), ("email", "Email", "email"), ("password", "Password (8+ characters)", "password")], "<p><a href=/login>Already registered? Log in</a></p>"),
 "login.html": form("Log in", [("email", "Email", "email"), ("password", "Password", "password")], "<p><a href=/forgot>Forgot password?</a> · <a href=/signup>Create account</a></p>"),
 "forgot.html": form("Reset password", [("email", "Email", "email")]),
 "reset.html": form("Set new password", [("password", "New password (8+ characters)", "password")]),
 "profile.html": page("<h1>Profile</h1><form class=stack method=post><label>Name<input name=name value='{{ g.user['name'] }}'></label><p class=mute>{{ g.user['email'] }}</p><button class=btn>Save changes</button></form>"),
 "dashboard.html": page("""<h1>{{ hello }}, {{ g.user['name'] }} 👋</h1>
{% for t in reminders %}<div class=banner>Reminder: {{ t['title'] }} — {{ due_label(t['deadline']) }}</div>{% endfor %}
<div class=grid>{% for l,k in [('Urgent tasks','urgent'),('Upcoming deadlines','upcoming'),('Pending documents','pending_docs'),('Completed tasks','done'),('Total documents','docs')] %}
<div class="card stat"><b>{{ stats[k] }}</b>{{ l }}</div>{% endfor %}</div>
<h2>AI insights</h2><div class=card>{% for i in insights %}<div class=row>💡 {{ i }}</div>{% else %}<span class=mute>Upload a document to see insights.</span>{% endfor %}</div>
<h2>Today's actions</h2><div class=card>{% for t in tasks %}{% set p = priority(t['deadline']) %}
<div class=row><form method=post action="/tasks/{{ t['id'] }}/toggle"><input type=checkbox style=width:auto {{ 'checked' if t['status']=='done' }} onchange=this.form.submit()></form>
<div class=grow><span class="{{ 'done' if t['status']=='done' }}">{{ t['title'] }}</span><div class=mute>{{ t['doc'] }}{% if t['effort'] %} · {{ t['effort'] }}{% endif %}{% if t['depends_on'] %} · needs {{ t['depends_on'] }}{% endif %}</div></div>
<span class=mute>{{ due_label(t['deadline']) }}</span><span class="tag {{ p }}">{{ p }}</span></div>
{% else %}<span class=mute>No tasks yet. <a href=/upload>Upload your first document.</a></span>{% endfor %}</div>
<h2>Upcoming deadlines</h2><div class=card>{% for t in timeline %}<div class=row><b>{{ t['deadline'] }}</b><span class=grow>{{ t['title'] }}</span><span class="tag {{ priority(t['deadline']) }}">{{ due_label(t['deadline']) }}</span></div>
{% else %}<span class=mute>No deadlines detected.</span>{% endfor %}</div>
<h2>Recent documents</h2><div class=card>{% for d in docs %}<div class=row><a class=grow href="/documents/{{ d['id'] }}">{{ d['title'] }}</a><span class=mute>{{ d['doc_type'] }} · {{ d['created'] }} · {{ d['deadlines'] }} deadlines · {{ d['status'] }}</span></div>
{% else %}<span class=mute>Nothing uploaded yet.</span>{% endfor %}</div>"""),
 "upload.html": page("""<h1>Upload a document</h1><div class=drop id=drop><p>Drag a PDF, JPG or PNG here</p><input type=file id=file accept=".pdf,.jpg,.jpeg,.png" style=max-width:300px></div>
<div class=card id=prog hidden><h3>Analyzing your document...</h3><ol id=steps><li>Reading document</li><li>Identifying document type</li><li>Extracting important information</li><li>Detecting deadlines</li><li>Finding requirements</li><li>Creating action plan</li></ol><p id=err style=color:var(--red)></p></div>
<script>const $=s=>document.querySelector(s);
async function send(f){if(!f)return;$('#prog').hidden=false;$('#err').textContent='';const li=[...document.querySelectorAll('#steps li')];let i=0;li[0].className='on';
const t=setInterval(()=>{if(i<li.length-1){li[i].className='';li[++i].className='on'}},1800);
const fd=new FormData();fd.append('file',f);const r=await fetch('/upload',{method:'POST',body:fd});const j=await r.json();clearInterval(t);
if(j.redirect)location=j.redirect;else $('#err').textContent=j.error||'Upload failed. Try again.'}
$('#file').onchange=e=>send(e.target.files[0]);const d=$('#drop');d.ondragover=e=>{e.preventDefault();d.classList.add('over')};
d.ondragleave=()=>d.classList.remove('over');d.ondrop=e=>{e.preventDefault();send(e.dataTransfer.files[0])}</script>"""),
 "document.html": page("""<h1>{{ doc['title'] }}</h1><p><span class="tag low" style=background:var(--teal)>{{ doc['doc_type'] }}</span> <span class=mute>{{ doc['created'] }}</span></p>
<div class=card><h3>Summary</h3><p>{{ doc['summary'] }}</p>{% for k,v in info.items() %}<div class=row><b>{{ k }}</b><span class=grow>{{ v }}</span></div>{% endfor %}</div>
<h2>Action plan</h2><div class=card>{% for t in tasks %}<div class=row><form method=post action="/tasks/{{ t['id'] }}/toggle"><input type=checkbox style=width:auto {{ 'checked' if t['status']=='done' }} onchange=this.form.submit()></form>
<span class="grow {{ 'done' if t['status']=='done' }}">{{ t['title'] }}</span><span class=mute>{{ due_label(t['deadline']) }}</span><span class="tag {{ priority(t['deadline']) }}">{{ priority(t['deadline']) }}</span></div>{% else %}<span class=mute>No actions found.</span>{% endfor %}</div>
<h2>Ask this document</h2><div class=card><input id=q placeholder="e.g. What is the last date to apply?"><p><button class=btn onclick="ai('ask')">Ask</button>
<button class="btn alt" onclick="ai('simplify')">Simplify</button> <select id=lang style=width:auto><option>Hindi</option><option>Telugu</option><option>Tamil</option><option>Spanish</option><option>French</option></select>
<button class="btn alt" onclick="ai('translate')">Translate</button></p><pre id=out class=mute></pre></div>
<script>async function ai(mode){const o=document.getElementById('out');o.textContent='Thinking...';
const r=await fetch(location.pathname+'/ai',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode,q:document.getElementById('q').value,lang:document.getElementById('lang').value})});
const j=await r.json();o.textContent=j.answer||j.error}</script>"""),
}
app.jinja_loader = DictLoader(T)

if __name__ == "__main__":
    
    app.run(debug=False,use_reloader=False)
