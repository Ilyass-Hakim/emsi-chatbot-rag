"""Developer dashboard: add documents, rebuild the vector index and test the search."""

import streamlit as st
from qdrant_client import QdrantClient

import config
import ingest
from retriever import search

st.set_page_config(page_title="EMSI ChatBot - Ingestion", layout="wide")
st.title("EMSI ChatBot - Ingestion")
client = QdrantClient(url=config.QDRANT_URL)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Documents ------------------------------------------------------------------
st.header("1. Documents")
uploaded = st.file_uploader("Add PDF documents", type="pdf", accept_multiple_files=True)
for file in uploaded or []:
    (config.DATA_DIR / file.name).write_bytes(file.getvalue())
if uploaded:
    st.success(f"{len(uploaded)} document(s) added. Rebuild the index to include them.")

pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
if pdfs:
    st.table([{"Document": pdf.name, "Size (MB)": round(pdf.stat().st_size / 1e6, 1)}
              for pdf in pdfs])
else:
    st.info("No document yet.")

# 2. Vector index ---------------------------------------------------------------
st.header("2. Vector index")
if st.button("Rebuild the index", type="primary", disabled=not pdfs):
    with st.status("Rebuilding the index...", expanded=True) as status:
        total = ingest.run(log=st.write)
        status.update(label=f"Index rebuilt: {total} points", state="complete", expanded=False)

if client.collection_exists(config.COLLECTION):
    st.metric("Points in Qdrant", client.count(config.COLLECTION).count)
else:
    st.info("The index has not been built yet.")

# 3. Search test ----------------------------------------------------------------
st.header("3. Search test")
question = st.text_input("Question", placeholder="Quelles filières propose l'EMSI ?")
if question:
    for hit in search(question, k=config.TOP_K):
        with st.container(border=True):
            st.caption(f"Page {hit['page']} - {hit['type']} - score {hit['score']:.3f}")
            if hit["type"] == "image":
                st.image(str(config.IMAGES_DIR / hit["image"]), width=240)
            st.write(hit["content"])
