# app.py
import streamlit as st
import hashlib, time, json, tempfile, os, re
from io import BytesIO
from collections import defaultdict

# ---------------- Text parsing ----------------
import nltk

# ---------------- Fix NLTK punkt path issues ----------------
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path and download punkt
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    from nltk.tokenize import sent_tokenize

    # Test if punkt works
    sent_tokenize("This is a test sentence.")
except (LookupError, OSError):
    print("Downloading punkt tokenizer...")
    nltk.download('punkt', download_dir=nltk_data_path, quiet=False)
    from nltk.tokenize import sent_tokenize

import pdfplumber
import docx

# ---------------- Embeddings ----------------
from sentence_transformers import SentenceTransformer, util


# ---------- Helpers: extract text ----------
def extract_text_from_pdf_bytes(b):
    text = []
    with pdfplumber.open(BytesIO(b)) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def extract_text_from_docx_bytes(b):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(b)
        tmp.flush()
        tmp_name = tmp.name
    doc = docx.Document(tmp_name)
    paras = [p.text for p in doc.paragraphs]
    try:
        os.unlink(tmp_name)
    except:
        pass
    return "\n".join(paras)


def extract_text_from_bytes(fname, b):
    fname = fname.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if fname.endswith(".docx"):
        return extract_text_from_docx_bytes(b)
    # txt / md fallback
    try:
        return b.decode('utf-8', errors='ignore')
    except:
        return str(b)


def split_to_sentences(text):
    try:
        sents = []
        for p in text.splitlines():
            p = p.strip()
            if not p:
                continue
            sents.extend([s.strip() for s in sent_tokenize(p) if s.strip()])
        return sents
    except Exception as e:
        st.error(f"Error in sentence tokenization: {e}")
        # Fallback to simple splitting
        sentences = []
        for p in text.splitlines():
            p = p.strip()
            if not p:
                continue
            # Simple sentence splitting on periods
            simple_sents = [s.strip() + '.' for s in p.split('.') if s.strip()]
            sentences.extend(simple_sents)
        return sentences


# ---------- Small heuristics ----------
NUM_RE = re.compile(r'[-+]?\d[\d,\,\.]*')


def extract_numbers(s):
    found = NUM_RE.findall(s)
    cleaned = []
    for n in found:
        n2 = n.replace(',', '')
        try:
            cleaned.append(float(n2))
        except:
            pass
    return cleaned


MONTHS_RE = re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                       re.I)


def has_month_word(s):
    return bool(MONTHS_RE.search(s))


NEG_WORDS = {"not", "never", "no", "without", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "mustn't",
             "unless"}


def has_negation(s):
    sl = s.lower()
    return any(w in sl for w in NEG_WORDS)


# ---------- Models (cached) ----------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def load_nli_model_safe(model_name="roberta-large-mnli"):
    """
    Loads NLI model. Returns (tokenizer, model, device, id2label) or None on failure.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        st.warning("transformers/torch not available — NLI disabled. Install optional requirements to enable.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        raw_map = dict(model.config.id2label)
        id2label = {int(k): v.lower() for k, v in raw_map.items()}
        return (tokenizer, model, device, id2label)
    except Exception as e:
        st.error(f"Failed to load NLI model: {e}")
        return None


def run_nli(tokenizer, model, device, id2label, premise, hypothesis):
    import torch
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label_scores = {}
    for i, p in enumerate(probs):
        lab = id2label.get(i, f"label_{i}")
        label_scores[lab] = float(p)
    top = max(label_scores.items(), key=lambda x: x[1])[0]
    return top, label_scores


# ---------- Analysis ----------
def analyze_documents(docs, emb_model, use_nli=False, nli_objects=None,
                      sim_threshold=0.72, contradiction_prob_threshold=0.40):
    flagged = []
    all_sents = []
    meta = []
    for d in docs:
        for idx, s in enumerate(d['sentences']):
            all_sents.append(s)
            meta.append({'doc_id': d['id'], 'doc_name': d['name'], 'sent_idx': idx})
    if len(all_sents) < 2:
        return flagged

    embeddings = emb_model.encode(all_sents, convert_to_tensor=True, show_progress_bar=False)

    import itertools
    for i, j in itertools.combinations(range(len(all_sents)), 2):
        if meta[i]['doc_id'] == meta[j]['doc_id']:
            continue
        sa, sb = all_sents[i], all_sents[j]
        sim = float(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
        if sim < sim_threshold:
            continue

        nums_a, nums_b = extract_numbers(sa), extract_numbers(sb)
        num_conflict = False
        if nums_a and nums_b:
            if abs(nums_a[0] - nums_b[0]) > max(1.0, 0.05 * max(abs(nums_a[0]), abs(nums_b[0]))):
                num_conflict = True

        date_conflict = has_month_word(sa) or has_month_word(sb)
        neg_conflict = has_negation(sa) != has_negation(sb)

        reasons = []
        if num_conflict: reasons.append("number_mismatch")
        if date_conflict: reasons.append("date_mismatch")
        if neg_conflict: reasons.append("negation_mismatch")
        if not reasons: reasons.append("semantic_similarity")

        nli_result = None
        estimated_prob = 0.0
        if use_nli and nli_objects:
            tok, model, device, id2label = nli_objects
            lab_ab, scores_ab = run_nli(tok, model, device, id2label, sa, sb)
            lab_ba, scores_ba = run_nli(tok, model, device, id2label, sb, sa)
            contradiction_prob = max(scores_ab.get('contradiction', 0.0), scores_ba.get('contradiction', 0.0))
            nli_result = {"a_to_b_label": lab_ab, "a_to_b_scores": scores_ab,
                          "b_to_a_label": lab_ba, "b_to_a_scores": scores_ba,
                          "contradiction_prob": float(contradiction_prob)}
            estimated_prob = float(contradiction_prob)
            is_flag = (contradiction_prob >= contradiction_prob_threshold)
        else:
            heuristic_prob = 0.65 if (num_conflict or date_conflict or neg_conflict) else 0.0
            nli_result = {"heuristic_contradiction_prob": heuristic_prob}
            estimated_prob = heuristic_prob
            is_flag = (num_conflict or date_conflict or neg_conflict)

        if is_flag:
            flagged.append({
                "docA": meta[i]['doc_name'],
                "sentenceA": sa,
                "docB": meta[j]['doc_name'],
                "sentenceB": sb,
                "similarity": sim,
                "reasons": reasons,
                "nli": nli_result,
                "estimated_contradiction_prob": estimated_prob
            })

    flagged.sort(key=lambda x: (-x.get("estimated_contradiction_prob", 0.0), -x.get("similarity", 0.0)))
    return flagged


# ---------- Generate readable report ----------
def generate_readable_report(report, flags):
    from datetime import datetime

    report_text = []
    report_text.append("=" * 60)
    report_text.append("SMART DOC CHECKER - CONTRADICTION ANALYSIS REPORT")
    report_text.append("=" * 60)
    report_text.append("")

    # Report metadata
    report_text.append(f"Generated: {datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
    report_text.append(f"Total contradictions found: {len(flags)}")
    report_text.append("")

    if not flags:
        report_text.append("No contradictions detected between uploaded documents.")
        report_text.append("")
        report_text.append("This suggests that your documents are consistent with each other.")
    else:
        report_text.append("DETAILED FINDINGS:")
        report_text.append("-" * 40)
        report_text.append("")

        for i, flag in enumerate(flags, 1):
            report_text.append(f"{i}. CONTRADICTION BETWEEN: {flag['docA']} ↔ {flag['docB']}")
            report_text.append("")

            # Conflicting sentences
            report_text.append("   CONFLICTING STATEMENTS:")
            report_text.append(f"   Document A: \"{flag['sentenceA']}\"")
            report_text.append(f"   Document B: \"{flag['sentenceB']}\"")
            report_text.append("")

            # Analysis details
            report_text.append("   ANALYSIS:")
            report_text.append(
                f"   • Similarity Score: {flag['similarity']:.2f} (0.0 = completely different, 1.0 = identical)")
            report_text.append(f"   • Detected Issues: {', '.join(flag['reasons'])}")

            if 'nli' in flag and isinstance(flag['nli'], dict):
                if 'contradiction_prob' in flag['nli']:
                    report_text.append(f"   • AI Contradiction Confidence: {flag['nli']['contradiction_prob']:.2f}")
                elif 'heuristic_contradiction_prob' in flag['nli']:
                    report_text.append(
                        f"   • Estimated Contradiction Probability: {flag['nli']['heuristic_contradiction_prob']:.2f}")
            report_text.append("")

            # Suggestions
            report_text.append("   RECOMMENDED ACTIONS:")
            suggestions = suggest_fix(flag)
            for j, suggestion in enumerate(suggestions, 1):
                report_text.append(f"   {j}. {suggestion}")

            report_text.append("")
            report_text.append("-" * 60)
            report_text.append("")

    # Footer
    report_text.append("")
    report_text.append("ABOUT THIS REPORT:")
    report_text.append("This analysis uses AI-powered natural language processing to detect")
    report_text.append("potential contradictions between documents. Review all flagged items")
    report_text.append("and use human judgment to determine which issues require resolution.")
    report_text.append("")
    report_text.append("Generated by Smart Doc Checker - Agent (Prototype)")

    return "\n".join(report_text)


# ---------- Suggest fixes ----------
def suggest_fix(flag):
    sa, sb = flag['sentenceA'], flag['sentenceB']
    reasons = flag.get('reasons', [])
    suggestions = []
    if "number_mismatch" in reasons:
        nums_a, nums_b = extract_numbers(sa), extract_numbers(sb)
        if nums_a and nums_b:
            suggestions.append(
                f"Decide authoritative value; update both docs to one value (e.g., {nums_a[0]} or {nums_b[0]}), or add a clear range.")
        else:
            suggestions.append("Clarify numeric values and units (e.g., '10 PM' vs '22:00').")
    if "date_mismatch" in reasons:
        suggestions.append("Pick a single canonical deadline/time and reference it; mention timezone if relevant.")
    if "negation_mismatch" in reasons:
        suggestions.append("Make sentences explicit: avoid implicit negatives; use 'must'/'must not' phrasing.")
    if "semantic_similarity" in reasons:
        suggestions.append("If statements have different scopes, add clarifying condition sentences (who/when/where).")
    suggestions.append("Add a canonical 'policy source' line and 'last updated' date for authoritative reference.")
    return suggestions


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Smart Doc Checker", layout="wide")
st.title("Smart Doc Checker — Agent (Prototype)")

# session state
if 'docs' not in st.session_state: st.session_state['docs'] = {}
if 'flexprice' not in st.session_state: st.session_state['flexprice'] = {"docs_analyzed": 0, "reports_generated": 0}
if 'last_report' not in st.session_state: st.session_state['last_report'] = None

col1, col2 = st.columns([1, 2])
with col1:
    st.header("1) Upload documents (2–3)")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT (multiple)", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            raw = f.read()
            fid = hashlib.blake2b(raw, digest_size=8).hexdigest()
            if fid in st.session_state['docs']: continue
            text = extract_text_from_bytes(f.name, raw)
            sents = split_to_sentences(text)
            st.session_state['docs'][fid] = {"id": fid, "name": f.name, "text": text, "sentences": sents,
                                             "uploaded_at": time.time()}
        st.success("Uploaded and parsed.")

    st.markdown("*Uploaded docs:*")
    if st.session_state['docs']:
        for d in st.session_state['docs'].values():
            st.write(f"- {d['name']} (sentences: {len(d['sentences'])})")
    else:
        st.write("No docs yet.")

    st.markdown("---")
    st.header("Pathway (mock external update)")
    st.write("Paste or upload a policy text to simulate Pathway delivering an external update.")
    pathway_text = st.text_area("Paste external doc text (optional)", height=120)
    pathway_file = st.file_uploader("Or upload single external file (Pathway)", key="pathway")
    if st.button("Ingest external update (Pathway)"):
        if pathway_file:
            raw = pathway_file.read()
            name = "PATHWAY:" + pathway_file.name
            fid = hashlib.blake2b(raw + b'pathway', digest_size=8).hexdigest()
            text = extract_text_from_bytes(pathway_file.name, raw)
        elif pathway_text.strip():
            text = pathway_text.strip()
            name = "PATHWAY:manual_paste"
            fid = hashlib.blake2b(text.encode('utf-8') + b'pathway', digest_size=8).hexdigest()
        else:
            st.warning("Please paste text or upload a file to ingest.")
            st.stop()
        sents = split_to_sentences(text)
        st.session_state['docs'][fid] = {"id": fid, "name": name, "text": text, "sentences": sents,
                                         "uploaded_at": time.time()}
        st.success(f"Ingested {name}. Run analysis to detect new conflicts.")

    st.markdown("---")
    st.header("Flexprice billing")
    doc_cost = st.number_input("Cost per document analyzed (credits)", min_value=0.0, value=1.0, step=0.5)
    report_cost = st.number_input("Cost per report generated (credits)", min_value=0.0, value=2.0, step=0.5)
    st.write("Usage so far:")
    st.metric("Docs analyzed", st.session_state['flexprice']['docs_analyzed'])
    st.metric("Reports generated", st.session_state['flexprice']['reports_generated'])
    total_credits = st.session_state['flexprice']['docs_analyzed'] * doc_cost + st.session_state['flexprice'][
        'reports_generated'] * report_cost
    st.write(f"Total (simulated) credits billed: *{total_credits:.2f}*")

with col2:
    st.header("2) Analyzer & Report")
    use_nli = st.checkbox("Enable NLI-based contradiction check (optional, slower)", value=False)
    sim_threshold = st.slider("Embedding similarity threshold", min_value=0.5, max_value=0.95, value=0.72, step=0.01)
    contr_prob_threshold = st.slider("NLI contradiction probability threshold", min_value=0.1, max_value=0.9,
                                     value=0.40, step=0.05)
    st.info("For a fast demo keep NLI off. NLI requires large downloads and more CPU/RAM.")

    emb_model = load_embedding_model()
    nli_objects = None
    if use_nli:
        with st.spinner("Loading NLI model (may take time)..."):
            nli_objects = load_nli_model_safe()
        if nli_objects is None:
            st.warning("NLI unavailable. Continuing without NLI.")
            use_nli = False
        else:
            st.success("NLI model ready.")

    if st.button("Analyze documents for contradictions"):
        docs_list = list(st.session_state['docs'].values())
        if len(docs_list) < 2:
            st.warning("Upload at least two documents to analyze.")
        else:
            with st.spinner("Running analysis..."):
                flagged = analyze_documents(docs_list, emb_model, use_nli=use_nli, nli_objects=nli_objects,
                                            sim_threshold=sim_threshold,
                                            contradiction_prob_threshold=contr_prob_threshold)
            st.session_state['last_report'] = {"generated_at": time.time(), "flags": flagged}
            st.session_state['flexprice']['docs_analyzed'] += len(docs_list)
            st.session_state['flexprice']['reports_generated'] += 1
            st.success(f"Analysis complete — flagged {len(flagged)} potential contradictions.")

    st.markdown("---")
    if st.session_state['last_report'] is None:
        st.write("No analysis run yet.")
    else:
        report = st.session_state['last_report']
        flags = report.get("flags", [])
        st.header("Auto-generated report")
        st.write(f"Generated at: {time.ctime(report['generated_at'])}")
        st.write(f"Total flagged contradictions: *{len(flags)}*")
        for i, f in enumerate(flags[:40]):
            st.subheader(f"{i + 1}. {f['docA']}  ↔  {f['docB']}")
            st.markdown(f"*A:* {f['sentenceA']}")
            st.markdown(f"*B:* {f['sentenceB']}")
            st.write(f"Similarity: {f['similarity']:.2f} | Reasons: {', '.join(f['reasons'])}")
            if 'nli' in f and isinstance(f['nli'], dict):
                if 'contradiction_prob' in f['nli']:
                    st.write(f"NLI contradiction probability: {f['nli']['contradiction_prob']:.2f}")
                elif 'heuristic_contradiction_prob' in f['nli']:
                    st.write(
                        f"Estimated (heuristic) contradiction probability: {f['nli']['heuristic_contradiction_prob']:.2f}")
            st.markdown("*Suggested clarifications:*")
            for s in suggest_fix(f):
                st.write(f"- {s}")
            st.markdown("---")

        # Download options
        col_download1, col_download2 = st.columns(2)

        with col_download1:
            st.download_button("Download full report (JSON)", json.dumps(report, indent=2, ensure_ascii=False),
                               file_name="smart_doc_checker_report.json", mime="application/json")

        with col_download2:
            # Generate human-readable report
            readable_report = generate_readable_report(report, flags)
            st.download_button("Download readable report (TXT)", readable_report,
                               file_name="smart_doc_checker_report.txt", mime="text/plain")

st.caption("Prototype — data stored in session memory (no DB). For hackathon/demo use.")