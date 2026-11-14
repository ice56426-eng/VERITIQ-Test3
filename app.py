"""VERITIQ demo app (Gradio). Run: python app.py
Standard use stores nothing. The Setup panel turns on opt-in Pro features (see pro.py)."""
import base64, io, os, tempfile, gradio as gr, pandas as pd
import pro
from veritiq import make_data, train_final_model, flag_transactions, cards_html

MODEL = train_final_model()
DISCLAIMER = "Prototype, not financial advice. Model trained on synthetic data; flags are leads to review, not verdicts."
COLS = ["id", "tx_date", "vendor", "amount", "type", "est_impact", "status", "actual_savings", "reason"]
SHOW = ["date", "vendor", "amount", "score", "type", "est_impact", "reason"]

# ---------- logo (found in assets/ or the repo root, embedded so the layout needs no file paths) ----------
def _logo_src():
    for p in (os.path.join("assets", "logo.png"), "logo.png"):
        if os.path.exists(p):
            try:
                from PIL import Image
                im = Image.open(p).convert("RGBA"); im.thumbnail((360, 360))
                buf = io.BytesIO(); im.save(buf, "PNG"); data = buf.getvalue()
            except Exception:
                data = open(p, "rb").read()
            return "data:image/png;base64," + base64.b64encode(data).decode()
    return ""
LOGO_SRC = _logo_src()

# ---------- logic (same as before, plus the chat box) ----------
def _path(f):
    if f is None: return None
    if isinstance(f, dict): return f.get("path") or f.get("name")
    return getattr(f, "name", f)

def run(path, use_demo, sens):
    try:
        raw = make_data()[["date", "vendor", "amount"]] if (use_demo or path is None) else pd.read_csv(path)
        flags = flag_transactions(raw, MODEL, threshold=float(sens))
    except Exception as e:   # readable message instead of a crash
        return pd.DataFrame(), "", f"Could not check this file: {e}"
    if pro.get("llm_enabled") == "1":   # optional: reword top 5 only; sends vendor, amount, reason to the provider you chose
        for i in flags.index[:5]:
            flags.loc[i, "reason"] = pro.llm_reword(flags.loc[i, "vendor"], flags.loc[i, "amount"], flags.loc[i, "reason"])
    note = ""
    if pro.is_enabled(): pro.save_run(flags, len(raw)); note = " Saved to your Pro workspace."
    warn = " ".join(flags.attrs.get("warnings", []))
    out = flags[SHOW].copy(); out["score"] = out.score.round(2); out["date"] = out.date.astype(str).str[:10]
    return out, cards_html(flags), f"**{len(raw)} charges checked, {len(flags)} flagged.**{note} {warn}\n\n_{DISCLAIMER}_"

def _flag_lines(df, limit=15):
    return "\n".join(f"- {r.date} {r.vendor} ${r.amount:,.2f} ({r.type}, score {r.score}): {r.reason}" for r in df.head(limit).itertuples())

def chat(msg, sens, table):
    msg = msg or {}
    text = (msg.get("text") or "").strip()
    files = [p for p in (_path(f) for f in (msg.get("files") or [])) if p]
    csvs = [p for p in files if p.lower().endswith(".csv")]
    others = [p for p in files if not p.lower().endswith(".csv")]
    keep = gr.update()
    if csvs:
        out, cards, status = run(csvs[0], False, sens)
        if others: status = f"Read the CSV and ignored {len(others)} other file(s): VERITIQ only reads CSV statements.\n\n" + status
        return out, cards, status
    if others:
        return keep, keep, "I can only read CSV statements with the columns `date, vendor, amount`. Images and PDFs aren't supported yet, so export your statement as a CSV."
    if text.lower() in ("demo", "demo data", "try demo"):
        return run(None, True, sens)
    if text:
        if table is None or "vendor" not in getattr(table, "columns", []) or len(table) == 0:
            return keep, keep, "Check a statement first: attach a CSV or type **demo**. Then you can ask questions about the flags."
        if pro.get("llm_enabled") != "1":
            return keep, keep, "AI answers are off. Open **Setup**, pick a provider, paste your API key and turn AI on."
        try: ans = pro.llm_ask(text, _flag_lines(table))
        except Exception as e: return keep, keep, f"The AI could not answer: {e}"
        return keep, keep, f"{ans}\n\n_AI-written from the flagged list only. Check it against the table._"
    return keep, keep, "Attach a CSV with `date, vendor, amount`, type **demo**, or ask a question about your flags."

def tracker(status):
    try:
        recs = pro.list_recs(status); s = pro.impact_summary()
    except pro.ProDisabled as e:
        return pd.DataFrame(columns=COLS), str(e)
    df = pd.DataFrame(recs, columns=COLS) if recs else pd.DataFrame(columns=COLS)
    md = (f"**Identified (estimated):** ${s['identified_total']:,.2f} across {s['n_recs']} flags  \n"
          f"**Accepted/planned/in progress/implemented (estimated, NOT realized):** ${s['accepted_est']:,.2f}  \n"
          f"**Implemented:** {s['n_implemented']}  ·  **Pending estimate:** ${s['pending_est']:,.2f}  \n"
          f"**Projected vs reported actual** (for {s['n_with_outcome']} items with an outcome): "
          f"${s['projected_for_reported']:,.2f} vs ${s['reported_actual']:,.2f} (difference ${s['difference']:,.2f}).  \n"
          "_Estimates are not guaranteed. 'Reported actual' is what you typed in; VERITIQ does not verify it._")
    return df, md

def update(rec_id, status, actual, notes):
    try:
        pro.update_rec(int(rec_id), status, None if actual in (None, "") else float(actual), notes or "")
        return "Updated."
    except Exception as e:
        return f"Could not update: {e}"

def history():
    try: return pd.DataFrame(pro.history())
    except pro.ProDisabled as e: return pd.DataFrame({"message": [str(e)]})

PROV0 = pro.get("llm_provider") if pro.get("llm_provider") in pro.PROVIDERS else "gemini"

def prov_note(pid):
    p = pro.PROVIDERS[pid]
    link = f' <a href="{p["key"]}" target="_blank" rel="noopener">Get an API key</a>' if p["key"] else ""
    return f'<p class="tiny">{p["note"]}{link}</p>'

def prov_change(pid):
    m = pro.PROVIDERS[pid]["model"]
    return prov_note(pid), gr.update(choices=[m], value=m)

def save_ai(on, pid, endpoint, model, key):
    pro.set_("llm_enabled", "1" if on else "0"); pro.set_("llm_provider", pid)
    pro.set_("llm_endpoint", endpoint or ""); pro.set_("llm_model", model or ""); pro.set_api_key(key, pid)
    n = pro.PROVIDERS[pid]["n"]
    if on and pid != "custom" and pid not in pro._KEYS: return f"Saved, but {n} needs an API key before AI will work."
    return f"Saved. AI is {'ON' if on else 'OFF'} ({n}). Press Test to check the connection."

def list_ui(pid, endpoint, key):
    try: ids = pro.list_models(pid, endpoint, key)
    except Exception as e: return gr.update(), f"Could not list models: {e}"
    return gr.update(choices=ids), f"{len(ids)} models found. Pick one in the Model box."

def save_pro(enabled, consent):
    pro.set_("pro_enabled", "1" if enabled else "0"); pro.set_("consent", "1" if consent else "0")
    if enabled and not consent: return "Saved, but persistent storage stays OFF until you tick the consent box."
    return "Saved. Pro workspace is " + ("ON." if pro.is_enabled() else "OFF (nothing is stored).")

def export():
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); f.write(pro.export_csv()); f.close(); return f.name
    except pro.ProDisabled: return None

def clear(confirm):
    if not confirm: return "Tick the confirmation box first."
    try: pro.clear_workspace(); return "Workspace cleared (settings kept)."
    except pro.ProDisabled as e: return str(e)

# ---------- look ----------
STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;1,400&family=Inter:wght@400;500;600&display=swap');
.gradio-container{--body-background-fill:#0a0714;--body-text-color:#f3f0ff;--body-text-color-subdued:#a39bc9;
--background-fill-primary:#120f22;--background-fill-secondary:#171329;--block-background-fill:#120f22;
--panel-background-fill:#120f22;--block-border-color:#2b2547;--border-color-primary:#2b2547;--border-color-accent:#8b5cf6;
--input-background-fill:#171329;--input-border-color:#2b2547;--block-label-background-fill:#171329;--block-label-text-color:#a39bc9;
--block-title-text-color:#d8d0ff;--table-even-background-fill:#120f22;--table-odd-background-fill:#171329;--table-row-focus:#241d44;
--button-secondary-background-fill:#171329;--button-secondary-background-fill-hover:#241d44;--button-secondary-text-color:#f3f0ff;
--button-secondary-border-color:#2b2547;--button-primary-background-fill:#7c3aed;--button-primary-background-fill-hover:#8b5cf6;
--button-primary-text-color:#fff;--checkbox-label-background-fill:#171329;--checkbox-background-color:#171329;
background:#0a0714!important;font-family:'Inter',system-ui,sans-serif}
#shell{gap:0!important;align-items:stretch}
#side{background:#0d0a1b;border-right:1px solid #2b2547;padding:22px 14px!important;min-height:94vh;gap:4px!important}
.brand img{width:148px;display:block}
.slogan{color:#a39bc9;font-size:.82rem;line-height:1.45;margin:12px 0 22px}
button.nav{justify-content:flex-start;text-align:left;border:none!important;background:transparent!important;color:#cfc7f5!important;box-shadow:none!important;border-radius:10px!important;font-weight:500}
button.nav:hover{background:rgba(139,92,246,.12)!important;color:#fff!important}
button.nav.primary{background:rgba(139,92,246,.20)!important;color:#fff!important;box-shadow:inset 3px 0 0 #a78bfa!important}
#spacer{flex:1 1 auto}
#setup-btn{margin-top:auto;border-top:1px solid #2b2547!important;border-radius:0!important;padding-top:14px}
#main{padding:30px 5vw!important}
.hero{max-width:780px;margin:5vh auto 22px;padding:34px 30px;border-radius:22px;border:1px solid #2b2547;
background:radial-gradient(70% 140% at 20% 0%,rgba(124,58,237,.30),rgba(124,58,237,0) 70%),#0f0c1e}
.hero h1{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:clamp(2.2rem,5vw,3.4rem);line-height:1.06;margin:0 0 14px;color:#fff}
.hero h1 em{color:#c4b5fd}
.hero p{color:#bdb5e0;margin:0;max-width:52ch;line-height:1.55}
#chat,#sens,.bubble,.cards,.chips{max-width:780px;margin-left:auto!important;margin-right:auto!important}
#chat{border:1px solid #3b2f6b;border-radius:18px;background:#120f22;box-shadow:0 14px 44px rgba(124,58,237,.14)}
#chat:focus-within{border-color:#8b5cf6;box-shadow:0 0 0 3px rgba(139,92,246,.22),0 14px 44px rgba(124,58,237,.18)}
#chat textarea{background:transparent!important;color:#fff!important}
.chips{gap:8px!important;flex-wrap:wrap}
button.chip{flex:0 0 auto!important;width:auto!important;min-width:0!important;border-radius:999px!important;background:#171329!important;border:1px solid #2b2547!important;color:#d8d0ff!important;font-size:.84rem!important;padding:6px 14px!important}
button.chip:hover{border-color:#8b5cf6!important;color:#fff!important}
.bubble{border-left:3px solid #8b5cf6;padding:4px 0 4px 14px;color:#f3f0ff}
.cards .vq{background:#120f22;border-color:#3b2f6b}
.card{background:#120f22;border:1px solid #2b2547!important;border-radius:16px!important;padding:18px 20px!important;margin-bottom:14px}
.card h3{margin-top:0;color:#fff}
.pills .wrap{gap:8px!important}
.pills label{border-radius:999px!important;background:#171329!important;border:1px solid #2b2547!important;padding:7px 16px!important;color:#d8d0ff!important}
.pills label.selected{background:#7c3aed!important;border-color:#a78bfa!important;color:#fff!important}
.tiny{color:#a39bc9;font-size:.85rem;margin:2px 0 8px}.tiny a{color:#c4b5fd}
.page-title{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:2rem;margin:0 0 12px;color:#fff}
.fine{color:#8f87b5;font-size:.8rem;text-align:center;margin-top:28px}
</style>"""

HERO = ('<div class="hero"><h1>Your statement.<br><em>Every charge checked.</em></h1>'
        '<p>Drop a CSV with date, vendor and amount, or type <b>demo</b>. In standard mode nothing is saved.</p></div>')
BRAND = (f'<div class="brand"><img src="{LOGO_SRC}" alt="VERITIQ"><p class="slogan">Analyze. Optimize. Evolve. Today, with VERITIQ.</p></div>')
VIEWS = ["check", "tracker", "history", "setup"]

def go(name):
    return ([gr.update(visible=(v == name)) for v in VIEWS] +
            [gr.update(variant="primary" if v == name else "secondary") for v in VIEWS])

def _nav_fn(name):
    def f(): return go(name)
    return f

with gr.Blocks(title="VERITIQ", theme=gr.themes.Base(primary_hue="purple")) as demo:
    gr.HTML(STYLE)
    with gr.Row(equal_height=False, elem_id="shell"):
        # ----- sidebar -----
        with gr.Column(scale=0, min_width=250, elem_id="side"):
            gr.HTML(BRAND)
            n_check = gr.Button("Check charges", variant="primary", elem_classes="nav")
            n_track = gr.Button("Tracker (Pro)", variant="secondary", elem_classes="nav")
            n_hist = gr.Button("History (Pro)", variant="secondary", elem_classes="nav")
            gr.HTML("", elem_id="spacer")
            n_setup = gr.Button("Setup", variant="secondary", elem_classes="nav", elem_id="setup-btn")
        # ----- main area -----
        with gr.Column(scale=1, elem_id="main"):
            with gr.Column(visible=True) as v_check:
                gr.HTML(HERO)
                mm = gr.MultimodalTextbox(placeholder="Attach a statement CSV, type demo, or ask about your flags", show_label=False,
                                          file_count="multiple", elem_id="chat")
                with gr.Row(elem_classes="chips"):
                    demo_btn = gr.Button("Try the demo data", size="sm", elem_classes="chip")
                    q1 = gr.Button("Which flag matters most?", size="sm", elem_classes="chip")
                    q2 = gr.Button("What should I do next?", size="sm", elem_classes="chip")
                with gr.Accordion("Sensitivity (lower catches more, with more false alarms)", open=False, elem_id="sens"):
                    sens = gr.Slider(0.1, 0.8, value=0.2, step=0.05, label="Threshold (0.2 matches our test)")
                status = gr.Markdown(elem_classes="bubble")
                cards = gr.HTML(elem_classes="cards")
                with gr.Accordion("All flagged charges (table)", open=False):
                    table = gr.Dataframe()
                outs = [table, cards, status]
                mm.submit(chat, [mm, sens, table], outs)
                demo_btn.click(lambda s: run(None, True, s), sens, outs)
                q1.click(lambda s, t: chat({"text": "Which flagged charge should I look at first, and why?"}, s, t), [sens, table], outs)
                q2.click(lambda s, t: chat({"text": "What should I do next about these flags?"}, s, t), [sens, table], outs)
            with gr.Column(visible=False) as v_tracker:
                gr.HTML('<h2 class="page-title">Tracker</h2>')
                flt = gr.Dropdown(["All"] + pro.STATUSES, value="All", label="Filter by status"); rb = gr.Button("Refresh")
                tt = gr.Dataframe(label="Recommendations"); sm = gr.Markdown(); rb.click(tracker, flt, [tt, sm])
                gr.Markdown("### Update a recommendation")
                rid = gr.Number(label="ID", precision=0); st = gr.Dropdown(pro.STATUSES, value="Accepted", label="Status")
                act = gr.Number(label="Actual savings you measured (optional, set after implementing)"); nt = gr.Textbox(label="Notes")
                ub = gr.Button("Update"); um = gr.Markdown(); ub.click(update, [rid, st, act, nt], um)
            with gr.Column(visible=False) as v_history:
                gr.HTML('<h2 class="page-title">History</h2>')
                hb = gr.Button("Load past runs"); ht = gr.Dataframe(); hb.click(history, None, ht)
            with gr.Column(visible=False) as v_setup:
                gr.HTML('<h2 class="page-title">Setup</h2>')
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### AI engine\nOptional. Pick a provider, paste your own API key, and VERITIQ can reword explanations and answer questions about your flags. "
                                "The flagging itself never uses AI. Free: Gemini, Groq, OpenRouter (:free models). ChatGPT and Claude APIs are paid.")
                    prov = gr.Radio([(v["n"], k) for k, v in pro.PROVIDERS.items()], value=PROV0, label="Provider preset", elem_classes="pills")
                    pnote = gr.HTML(prov_note(PROV0))
                    ky = gr.Textbox(label="API key", type="password", placeholder="Paste key. Kept in memory only; clears when the app stops.")
                    m0 = pro.get("llm_model") or pro.PROVIDERS[PROV0]["model"]
                    mdl = gr.Dropdown(choices=[m0], value=m0, allow_custom_value=True, label="Model")
                    ep = gr.Textbox(label="Endpoint (only used for Custom / local)", value=pro.get("llm_endpoint"))
                    lo = gr.Checkbox(label="Turn AI on (sends vendor, amount and reason of flagged charges to the provider you chose)", value=pro.get("llm_enabled") == "1")
                    with gr.Row():
                        sv = gr.Button("Save", variant="primary"); tc = gr.Button("Test"); lm = gr.Button("List models")
                    aim = gr.Markdown()
                    prov.change(prov_change, prov, [pnote, mdl])
                    sv.click(save_ai, [lo, prov, ep, mdl, ky], aim); tc.click(lambda: pro.llm_test(), None, aim)
                    lm.click(list_ui, [prov, ep, ky], [mdl, aim])
                    gr.HTML('<p class="tiny">Anyone who opens a shared link to this running app can use the key you enter, so only add a key on a session you control.</p>')
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Pro workspace\nOff by default. Turning it on stores your results in a local file (`veritiq_pro.db`) on this computer.")
                    e = gr.Checkbox(label="Enable Pro features (persistent memory, history, tracker)", value=pro.get("pro_enabled") == "1")
                    c = gr.Checkbox(label="I consent to saving flagged charges and my decisions locally", value=pro.get("consent") == "1")
                    sv2 = gr.Button("Save Pro settings", variant="primary"); sm2 = gr.Markdown(); sv2.click(save_pro, [e, c], sm2)
                    eb = gr.Button("Export history (CSV)"); ef = gr.File(); eb.click(export, None, ef)
                    cf = gr.Checkbox(label="Yes, permanently delete all saved runs and recommendations"); cb = gr.Button("Clear workspace"); cm = gr.Markdown(); cb.click(clear, cf, cm)
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Not implemented\nGoogle Drive / Sheets / Gmail, scheduled reports and alerts. These are roadmap items, not working features.")
            gr.HTML('<p class="fine">This project is a prototype and is not financial advice.</p>')
    views, btns = [v_check, v_tracker, v_history, v_setup], [n_check, n_track, n_hist, n_setup]
    for b, name in zip(btns, VIEWS):
        b.click(_nav_fn(name), None, views + btns)

if __name__ == "__main__":
    demo.launch(share=True)
