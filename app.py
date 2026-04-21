import streamlit as st
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Temple Lens – Glyph Analysis",
    layout="wide",
)

# --------------------------------------------------
# STYLE INDIANA JONES
# --------------------------------------------------
st.markdown("""
<style>

/* GLOBAL */
body {
    background: radial-gradient(circle at top, #3b2a1a, #0e0b07);
    color: #f3e6c8;
    font-family: 'Georgia', serif;
}

/* MAIN CONTAINER */
.block-container {
    padding: 2rem 3rem;
}

/* HEADERS */
h1, h2, h3 {
    color: #f5c76a;
    text-shadow: 0 0 6px rgba(245,199,106,0.4);
}

/* IMAGE FRAME */
.image-frame {
    border: 4px solid #6b4a2d;
    background: linear-gradient(145deg, #2a1c12, #1a120b);
    padding: 1rem;
    box-shadow: inset 0 0 30px rgba(0,0,0,0.8),
                0 0 25px rgba(245,199,106,0.2);
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(180deg, #7b4f2b, #3b2312);
    color: #f5deb3;
    border: 2px solid #d6a756;
    border-radius: 6px;
    padding: 0.6rem;
    font-weight: bold;
    box-shadow: 0 4px 0 #2a170a;
}

.stButton > button:hover {
    background: linear-gradient(180deg, #a06a3a, #4a2a15);
    color: #fff2cc;
}

/* ANALYSIS PANEL */
.analysis-box {
    border: 2px solid #d6a756;
    background: linear-gradient(145deg, #1c130b, #0f0a05);
    padding: 1rem;
    min-height: 220px;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.9);
    font-family: 'Courier New', monospace;
}

/* NAV BUTTONS */
.nav-btn button {
    font-size: 28px;
    padding: 0.3rem 1rem;
}

/* UPLOADER */
section[data-testid="stFileUploader"] {
    background-color: #1b120b;
    border: 2px dashed #d6a756;
    padding: 1rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "pages" not in st.session_state:
    st.session_state.pages = [{
        "image": None,
        "results": "📜 Aucune inscription analysée."
    }]

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

pages = st.session_state.pages
idx = st.session_state.current_page
page = pages[idx]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧭 Temple Lens")
st.caption("Analyse d’artefacts, glyphes et reliques anciennes")

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
nav_l, nav_c, nav_r = st.columns([1, 6, 1])

with nav_l:
    if st.button("⬅", key="prev"):
        st.session_state.current_page = (idx - 1) % len(pages)
        st.rerun()

with nav_r:
    if st.button("➡", key="next"):
        if idx == len(pages) - 1:
            pages.append({
                "image": None,
                "results": "📜 Nouvelle analyse prête."
            })
        st.session_state.current_page += 1
        st.rerun()

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([1.2, 1])

# LEFT – IMAGE
with left:
    st.subheader("🗿 Artefact visuel")

    uploaded = st.file_uploader(
        "Importer une relique",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        page["image"] = Image.open(uploaded)

    if page["image"]:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(page["image"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Aucun artefact chargé.")

# RIGHT – ANALYSIS
with right:
    st.subheader("⚙️ Panneau d’analyse")

    if st.button("🔍 Lancer l’analyse"):
        page["results"] = (
            "🔎 Analyse déclenchée...\n"
            "Objets détectés : fragments, symboles, gravures.\n"
        )

    if st.button("📚 Recherche glyphes"):
        page["results"] += "\n📖 Glyphes comparés aux archives anciennes."

    st.subheader("📜 Informations déchiffrées")
    st.markdown(
        f"<div class='analysis-box'>{page['results'].replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(f"Analyse {idx + 1} / {len(pages)} — Temple Lens")
