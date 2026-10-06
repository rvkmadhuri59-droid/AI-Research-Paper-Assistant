import hashlib

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# VISUAL DESIGN
# =========================================================

st.markdown(
    """
    <style>

    /* Hide only unnecessary Streamlit items.
       Do NOT hide the toolbar because it contains
       the sidebar reopening control. */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* Main page */

    .block-container {
        max-width: 1250px;
        padding-top: 3.2rem;
        padding-bottom: 3rem;
    }


    /* Sidebar */

    [data-testid="stSidebar"] {
        background: #f8f9fb;
        border-right: 1px solid #e7e9ee;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }


    /* Keep sidebar controls visible */

    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
    }


    /* Sidebar navigation */

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.55rem 0.65rem;
        border-radius: 9px;
        margin-bottom: 0.15rem;
        transition: 0.15s;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #eef1f6;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #e8edf8;
        font-weight: 600;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        border: 1px solid #e1e4e9;
        font-weight: 500;
    }

    .stButton > button:hover {
        border-color: #9ca3af;
    }


    /* Inputs */

    .stTextInput input {
        border-radius: 10px;
    }

    textarea {
        border-radius: 12px !important;
        font-size: 1rem !important;
    }


    /* File uploader */

    [data-testid="stFileUploader"] {
        margin-top: 0.2rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
        border: 1px dashed #c8cdd5;
        background: #fafbfc;
        padding-top: 0.65rem;
        padding-bottom: 0.65rem;
    }


    /* Metrics */

    [data-testid="stMetric"] {
        border: 1px solid #e6e8ed;
        border-radius: 11px;
        padding: 12px 14px;
        background: #fafbfc;
    }


    /* Home branding */

    .app-brand {
        font-size: 1.15rem;
        font-weight: 650;
        color: #555c68;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        padding-top: 0.6rem;
    }

    .hero-title {
        text-align: center;
        font-size: 2.35rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.65rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #777d87;
        font-size: 1.02rem;
        margin-bottom: 2rem;
    }


    /* Uploaded paper card */

    .paper-chip {
        padding: 10px 14px;
        border: 1px solid #e2e5ea;
        border-radius: 10px;
        background: #fafbfc;
        margin-top: 10px;
        margin-bottom: 12px;
        font-size: 0.94rem;
    }

    .paper-chip-small {
        color: #7c818a;
        font-size: 0.84rem;
        margin-top: 3px;
    }


    /* Labels */

    .section-caption {
        text-align: center;
        color: #858a93;
        font-size: 0.88rem;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    .sidebar-label {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #858a93;
        font-weight: 650;
        margin-top: 0.3rem;
        margin-bottom: 0.4rem;
    }

    .result-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "current_view": "Home",
    "paper_processed": False,
    "paper_name": "",
    "paper_hash": "",
    "page_count": 0,
    "chunk_count": 0,
    "all_chunks": [],
    "last_answer": "",
    "last_documents": [],
    "last_metadata": [],
    "last_result_title": "Research Assistant",
    "chat_history": [],
    "upload_version": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =========================================================
# CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# RESET RESEARCH
# =========================================================

def reset_research():

    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    st.session_state.paper_processed = False
    st.session_state.paper_name = ""
    st.session_state.paper_hash = ""
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.all_chunks = []

    st.session_state.last_answer = ""
    st.session_state.last_documents = []
    st.session_state.last_metadata = []
    st.session_state.last_result_title = "Research Assistant"

    st.session_state.chat_history = []

    st.session_state.current_view = "Home"

    st.session_state.upload_version += 1

    if "nav_radio" in st.session_state:
        del st.session_state["nav_radio"]


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(
        uploaded_file
    )

    all_chunks = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        chunks = create_chunks(
            page_text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk
                }
            )

    if not all_chunks:
        return False

    texts = [
        item["text"]
        for item in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    ids = []
    documents = []
    metadatas = []
    embedding_list = []

    for index, item in enumerate(
        all_chunks
    ):

        ids.append(
            f"chunk_{index}"
        )

        documents.append(
            item["text"]
        )

        metadatas.append(
            {
                "page": item["page"],
                "chunk": item["chunk"],
                "filename": uploaded_file.name
            }
        )

        embedding_list.append(
            embeddings[index].tolist()
        )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding_list
    )

    st.session_state.paper_processed = True
    st.session_state.paper_name = uploaded_file.name
    st.session_state.page_count = len(reader.pages)
    st.session_state.chunk_count = len(all_chunks)
    st.session_state.all_chunks = all_chunks

    st.session_state.last_answer = ""
    st.session_state.last_documents = []
    st.session_state.last_metadata = []
    st.session_state.chat_history = []

    return True


# =========================================================
# RETRIEVE CONTEXT
# =========================================================

def retrieve_context(question, n_results=4):

    total_chunks = collection.count()

    if total_chunks == 0:
        return [], [], ""

    n_results = min(
        n_results,
        total_chunks
    )

    question_embedding = embedding_model.encode(
        [question],
        show_progress_bar=False
    )[0]

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    context_parts = []

    for i, document in enumerate(
        documents
    ):

        page = metadata[i]["page"]
        chunk = metadata[i]["chunk"]

        context_parts.append(
            f"""
SOURCE {i + 1}

Page: {page}

Chunk: {chunk}

{document}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    return documents, metadata, context


# =========================================================
# OLLAMA
# =========================================================

def ask_ollama(question, context):

    prompt = f"""
You are a professional AI Research Paper Assistant.

Use ONLY the research-paper context supplied below.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the requested information is unavailable, say:

"The information is not available in the uploaded research paper."

4. Keep answers clear and research-oriented.
5. Use headings or bullet points where useful.

RESEARCH PAPER CONTEXT:

{context}

USER REQUEST:

{question}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 450
        }
    )

    return response["message"]["content"]


# =========================================================
# RUN ANALYSIS
# =========================================================

def run_analysis(question):

    documents, metadata, context = retrieve_context(
        question,
        n_results=4
    )

    if not context:

        return (
            "No indexed research-paper content is available.",
            [],
            []
        )

    answer = ask_ollama(
        question,
        context
    )

    return answer, documents, metadata


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 📚 Research Assistant"
    )

    st.caption(
        "Local research workspace"
    )

    st.write("")

    if st.button(
        "＋ New Research",
        use_container_width=True
    ):

        reset_research()
        st.rerun()

    st.markdown(
        "<div class='sidebar-label'>Workspace</div>",
        unsafe_allow_html=True
    )

    navigation_items = {
        "🏠  Home": "Home",
        "📄  Paper Overview": "Overview",
        "💬  Ask AI": "Ask AI",
        "🧠  Deep Analysis": "Deep Analysis",
        "📚  Sources": "Sources",
        "🔄  Compare Papers": "Compare"
    }

    labels = list(
        navigation_items.keys()
    )

    current_label = next(
        (
            label
            for label, value
            in navigation_items.items()
            if value == st.session_state.current_view
        ),
        "🏠  Home"
    )

    selected_label = st.radio(
        "Navigation",
        labels,
        index=labels.index(current_label),
        label_visibility="collapsed",
        key="nav_radio"
    )

    selected_view = navigation_items[
        selected_label
    ]

    if selected_view != st.session_state.current_view:

        st.session_state.current_view = selected_view
        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # CURRENT PAPER
    # -----------------------------------------------------

    if st.session_state.paper_processed:

        st.markdown(
            "<div class='sidebar-label'>Current Paper</div>",
            unsafe_allow_html=True
        )

        st.write(
            f"📄 **{st.session_state.paper_name}**"
        )

        st.caption(
            f"{st.session_state.page_count} pages · "
            f"{st.session_state.chunk_count} chunks"
        )

        st.success(
            "● Ready"
        )

    else:

        st.caption(
            "No paper currently loaded."
        )

    st.divider()

    # -----------------------------------------------------
    # LOCAL AI INFORMATION
    # -----------------------------------------------------

    with st.expander(
        "⚙ Local AI System"
    ):

        st.write(
            "**Language Model**"
        )

        st.caption(
            "Llama 3.2 : 3B"
        )

        st.write(
            "**Embeddings**"
        )

        st.caption(
            "all-MiniLM-L6-v2"
        )

        st.write(
            "**Vector Database**"
        )

        st.caption(
            "ChromaDB"
        )

        st.write(
            "**Processing**"
        )

        st.caption(
            "100% Local"
        )


# =========================================================
# HOME VIEW
# =========================================================

if st.session_state.current_view == "Home":

    st.markdown(
        '<div class="app-brand">📚 AI Research Paper Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero-title">
            How can I help with your research?
        </div>

        <div class="hero-subtitle">
            Upload a paper, ask a question, or choose a research tool.
        </div>
        """,
        unsafe_allow_html=True
    )

    side1, centre, side2 = st.columns(
        [1.15, 3.6, 1.15]
    )

    with centre:

        with st.container(
            border=True
        ):

            question = st.text_area(
                "Research request",
                placeholder="Give me any task to work on...",
                height=105,
                label_visibility="collapsed",
                key="home_question"
            )

            uploaded_file = st.file_uploader(
                "Add a research paper",
                type=["pdf"],
                key=(
                    f"paper_upload_"
                    f"{st.session_state.upload_version}"
                )
            )

            # ------------------------------------------------
            # PROCESS UPLOADED PAPER
            # ------------------------------------------------

            if uploaded_file is not None:

                file_bytes = uploaded_file.getvalue()

                new_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                needs_processing = (
                    not st.session_state.paper_processed
                    or new_hash != st.session_state.paper_hash
                )

                if needs_processing:

                    with st.spinner(
                        "Reading and indexing the research paper..."
                    ):

                        success = process_pdf(
                            uploaded_file
                        )

                    if success:

                        st.session_state.paper_hash = new_hash

                        st.success(
                            "Paper indexed successfully."
                        )

                    else:

                        st.error(
                            "No readable text was found in this PDF."
                        )

            # ------------------------------------------------
            # PAPER STATUS - FIXED VERSION
            # ------------------------------------------------

            if st.session_state.paper_processed:

                paper_html = (
                    f'<div class="paper-chip">'
                    f'<b>📄 {st.session_state.paper_name}</b>'
                    f'<div class="paper-chip-small">'
                    f'{st.session_state.page_count} pages'
                    f' &nbsp;•&nbsp; '
                    f'{st.session_state.chunk_count} indexed chunks'
                    f' &nbsp;•&nbsp; '
                    f'Ready'
                    f'</div>'
                    f'</div>'
                )

                st.markdown(
                    paper_html,
                    unsafe_allow_html=True
                )

            ask_button = st.button(
                "Ask Research Assistant  ➜",
                type="primary",
                use_container_width=True
            )

    # =====================================================
    # QUICK RESEARCH TOOLS
    # =====================================================

    st.markdown(
        "<div class='section-caption'>QUICK RESEARCH TOOLS</div>",
        unsafe_allow_html=True
    )

    tool_space1, tool_area, tool_space2 = st.columns(
        [0.7, 4.6, 0.7]
    )

    with tool_area:

        t1, t2, t3 = st.columns(3)

        with t1:

            summary_button = st.button(
                "📝 Summary",
                use_container_width=True
            )

        with t2:

            methodology_button = st.button(
                "🔬 Methodology",
                use_container_width=True
            )

        with t3:

            results_button = st.button(
                "📈 Key Results",
                use_container_width=True
            )

        t4, t5, t6 = st.columns(3)

        with t4:

            keywords_button = st.button(
                "🔑 Keywords",
                use_container_width=True
            )

        with t5:

            gaps_button = st.button(
                "🔎 Research Gaps",
                use_container_width=True
            )

        with t6:

            literature_button = st.button(
                "📚 Literature Review",
                use_container_width=True
            )

    # =====================================================
    # HANDLE HOME ACTION
    # =====================================================

    requested_prompt = None
    result_title = ""

    if ask_button:

        if question.strip():

            requested_prompt = question
            result_title = "Research Assistant"

        else:

            st.warning(
                "Enter a question or research task."
            )

    elif summary_button:

        requested_prompt = """
Provide a structured summary of this research paper.

Include:
- Research problem
- Objective
- Methodology
- Main findings
- Conclusion
"""

        result_title = "Paper Summary"

    elif methodology_button:

        requested_prompt = """
Explain the methodology used in this research paper.

Include:
- Research workflow
- Algorithms
- Models
- Datasets
- Processing steps
- Experimental procedure

Include only information available in the paper.
"""

        result_title = "Methodology"

    elif results_button:

        requested_prompt = """
Identify and explain the main results and findings
reported in this research paper.
"""

        result_title = "Key Results"

    elif keywords_button:

        requested_prompt = """
Extract the important technical keywords,
algorithms, models, datasets and research concepts
contained in this research paper.
"""

        result_title = "Keywords"

    elif gaps_button:

        requested_prompt = """
Identify potential research gaps based only on:

- Limitations
- Unresolved problems
- Discussion
- Results
- Future work

Describe these as potential gaps derived from this
paper and not definitive gaps in the whole literature.
"""

        result_title = "Potential Research Gaps"

    elif literature_button:

        requested_prompt = """
Create a concise literature-review-style overview
based only on the content contained in this
uploaded research paper.
"""

        result_title = "Literature Review"

    # =====================================================
    # GENERATE RESULT
    # =====================================================

    if requested_prompt:

        if not st.session_state.paper_processed:

            st.warning(
                "Upload a research paper first."
            )

        else:

            with st.spinner(
                "Analyzing the research paper..."
            ):

                answer, docs, metadata = run_analysis(
                    requested_prompt
                )

            st.session_state.last_answer = answer
            st.session_state.last_documents = docs
            st.session_state.last_metadata = metadata
            st.session_state.last_result_title = result_title

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    if st.session_state.last_answer:

        st.divider()

        r1, result_area, r2 = st.columns(
            [0.55, 4.9, 0.55]
        )

        with result_area:

            st.markdown(
                f'<div class="result-heading">'
                f'🤖 {st.session_state.last_result_title}'
                f'</div>',
                unsafe_allow_html=True
            )

            with st.container(
                border=True
            ):

                st.write(
                    st.session_state.last_answer
                )

            with st.expander(
                "📄 Supporting evidence"
            ):

                for i, document in enumerate(
                    st.session_state.last_documents
                ):

                    page = st.session_state.last_metadata[i]["page"]

                    chunk = st.session_state.last_metadata[i]["chunk"]

                    st.markdown(
                        f"**Source {i + 1} — "
                        f"Page {page}, Chunk {chunk}**"
                    )

                    st.write(
                        document
                    )

                    if i < len(
                        st.session_state.last_documents
                    ) - 1:

                        st.divider()


# =========================================================
# PAPER OVERVIEW VIEW
# =========================================================

elif st.session_state.current_view == "Overview":

    st.title(
        "📄 Paper Overview"
    )

    if not st.session_state.paper_processed:

        st.info(
            "Upload a research paper from Home first."
        )

    else:

        st.caption(
            st.session_state.paper_name
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Pages",
            st.session_state.page_count
        )

        m2.metric(
            "Chunks",
            st.session_state.chunk_count
        )

        m3.metric(
            "LLM",
            "3B"
        )

        m4.metric(
            "Status",
            "Ready"
        )

        st.write("")

        o1, o2, o3 = st.columns(3)

        with o1:

            overview_summary = st.button(
                "📝 Summary",
                use_container_width=True
            )

        with o2:

            overview_objective = st.button(
                "🎯 Research Objective",
                use_container_width=True
            )

        with o3:

            overview_method = st.button(
                "🔬 Methodology",
                use_container_width=True
            )

        overview_prompt = None
        overview_title = ""

        if overview_summary:

            overview_prompt = """
Provide a structured summary of this research paper.

Include:
- Research problem
- Objective
- Methodology
- Main results
- Conclusion
"""

            overview_title = "Paper Summary"

        elif overview_objective:

            overview_prompt = """
Identify:
- Main research problem
- Research objectives
- Key contributions

Use only information contained in the paper.
"""

            overview_title = "Research Objective"

        elif overview_method:

            overview_prompt = """
Explain the research methodology used in this paper
in a clear step-by-step manner.
"""

            overview_title = "Methodology"

        if overview_prompt:

            with st.spinner(
                "Analyzing the paper..."
            ):

                answer, docs, metadata = run_analysis(
                    overview_prompt
                )

            st.divider()

            st.subheader(
                f"🤖 {overview_title}"
            )

            with st.container(
                border=True
            ):

                st.write(
                    answer
                )

            with st.expander(
                "Supporting evidence"
            ):

                for i, doc in enumerate(
                    docs
                ):

                    st.markdown(
                        f"**Page {metadata[i]['page']} — "
                        f"Chunk {metadata[i]['chunk']}**"
                    )

                    st.write(
                        doc
                    )


# =========================================================
# ASK AI VIEW
# =========================================================

elif st.session_state.current_view == "Ask AI":

    st.title(
        "💬 Ask AI"
    )

    if not st.session_state.paper_processed:

        st.info(
            "Upload a research paper from Home first."
        )

    else:

        st.caption(
            f"Discussing · {st.session_state.paper_name}"
        )

        st.divider()

        if not st.session_state.chat_history:

            st.info(
                "Ask your first question about the research paper."
            )

        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):

                st.write(
                    message["content"]
                )

        user_question = st.chat_input(
            "Ask anything about this paper..."
        )

        if user_question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_question
                }
            )

            with st.chat_message(
                "user"
            ):

                st.write(
                    user_question
                )

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching the paper..."
                ):

                    answer, docs, metadata = run_analysis(
                        user_question
                    )

                st.write(
                    answer
                )

                with st.expander(
                    "📄 Sources"
                ):

                    for i, doc in enumerate(
                        docs
                    ):

                        st.markdown(
                            f"**Page {metadata[i]['page']} — "
                            f"Chunk {metadata[i]['chunk']}**"
                        )

                        st.write(
                            doc
                        )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


# =========================================================
# DEEP ANALYSIS VIEW
# =========================================================

elif st.session_state.current_view == "Deep Analysis":

    st.title(
        "🧠 Deep Analysis"
    )

    st.caption(
        "Detailed examination of data, results, "
        "limitations and research opportunities."
    )

    if not st.session_state.paper_processed:

        st.info(
            "Upload a research paper from Home first."
        )

    else:

        d1, d2, d3 = st.columns(3)

        with d1:

            dataset_button = st.button(
                "📊 Dataset Analysis",
                use_container_width=True
            )

        with d2:

            result_button = st.button(
                "📈 Results Analysis",
                use_container_width=True
            )

        with d3:

            limitations_button = st.button(
                "⚠ Limitations",
                use_container_width=True
            )

        d4, d5, d6 = st.columns(3)

        with d4:

            future_button = st.button(
                "🔮 Future Work",
                use_container_width=True
            )

        with d5:

            strengths_button = st.button(
                "⚖ Strengths & Weaknesses",
                use_container_width=True
            )

        with d6:

            gap_button = st.button(
                "🔎 Research Gaps",
                use_container_width=True
            )

        deep_prompt = None
        deep_title = ""

        if dataset_button:

            deep_prompt = """
Analyze the datasets used in this paper.

Where available include:
- Dataset name
- Source
- Size
- Purpose
- Characteristics
- Training/testing configuration
"""

            deep_title = "Dataset Analysis"

        elif result_button:

            deep_prompt = """
Analyze the major quantitative and qualitative
results reported in this research paper.
"""

            deep_title = "Results Analysis"

        elif limitations_button:

            deep_prompt = """
Identify and explain the limitations described
or clearly supported by this research paper.
"""

            deep_title = "Research Limitations"

        elif future_button:

            deep_prompt = """
Identify future-work directions explicitly
mentioned or strongly supported by this paper.
"""

            deep_title = "Future Work"

        elif strengths_button:

            deep_prompt = """
Critically analyze the strengths and weaknesses
of this research using only evidence from the paper.
"""

            deep_title = "Strengths & Weaknesses"

        elif gap_button:

            deep_prompt = """
Identify potential research gaps supported by:
- Limitations
- Results
- Discussion
- Unresolved problems
- Future work

Do not claim these are definitive literature-wide gaps.
"""

            deep_title = "Potential Research Gaps"

        if deep_prompt:

            with st.spinner(
                "Performing deep analysis..."
            ):

                answer, docs, metadata = run_analysis(
                    deep_prompt
                )

            st.divider()

            st.subheader(
                f"🤖 {deep_title}"
            )

            with st.container(
                border=True
            ):

                st.write(
                    answer
                )

            with st.expander(
                "📄 Evidence used"
            ):

                for i, doc in enumerate(
                    docs
                ):

                    st.markdown(
                        f"**Source {i + 1} — "
                        f"Page {metadata[i]['page']}, "
                        f"Chunk {metadata[i]['chunk']}**"
                    )

                    st.write(
                        doc
                    )


# =========================================================
# SOURCES VIEW
# =========================================================

elif st.session_state.current_view == "Sources":

    st.title(
        "📚 Sources"
    )

    if not st.session_state.paper_processed:

        st.info(
            "Upload a research paper from Home first."
        )

    else:

        st.caption(
            st.session_state.paper_name
        )

        s1, s2 = st.columns(2)

        s1.metric(
            "Pages",
            st.session_state.page_count
        )

        s2.metric(
            "Indexed Chunks",
            st.session_state.chunk_count
        )

        page_filter = st.selectbox(
            "Filter by page",
            ["All Pages"]
            + list(
                range(
                    1,
                    st.session_state.page_count + 1
                )
            )
        )

        displayed = 0

        for item in st.session_state.all_chunks:

            if (
                page_filter != "All Pages"
                and item["page"] != page_filter
            ):
                continue

            displayed += 1

            with st.expander(
                f"Page {item['page']} · "
                f"Chunk {item['chunk']}"
            ):

                st.write(
                    item["text"]
                )

        st.caption(
            f"{displayed} chunks displayed"
        )


# =========================================================
# COMPARE PAPERS VIEW
# =========================================================

elif st.session_state.current_view == "Compare":

    st.title(
        "🔄 Compare Papers"
    )

    st.caption(
        "Multi-paper comparison workspace"
    )

    st.info(
        "The full comparison workflow will be "
        "developed after completing the "
        "single-paper assistant."
    )

    comparison_files = st.file_uploader(
        "Select research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if comparison_files:

        st.success(
            f"{len(comparison_files)} papers selected."
        )

        for file in comparison_files:

            st.write(
                f"📄 {file.name}"
            )
