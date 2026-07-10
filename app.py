import os
import re
import uuid
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangGraph and Langchain imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

# --- Streamlit UI Setup ---
# NOTE: set_page_config() must be the very first Streamlit command that runs
# in the script, before any other st.* call (including ones inside functions
# that might get triggered by an exception, like st.error()/st.stop() below).
st.set_page_config(page_title="Academic QA", page_icon="🎓", layout="wide")

# --- Professional theme polish (colors also set in .streamlit/config.toml) ---
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 900px;}
[data-testid="stSidebar"] {background-color: #F4F6FA;}
[data-testid="stSidebar"] button, div.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(49,51,63,0.15) !important;
}
[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid rgba(49,51,63,0.08);
}
.app-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 4px;
}
.app-header .logo-circle {
    width: 42px; height: 42px; border-radius: 50%;
    background: #E6F1FB; display: flex; align-items: center;
    justify-content: center; font-size: 20px; flex-shrink: 0;
}
.app-header .title {
    font-weight: 700; font-size: 20px; line-height: 1.2; margin: 0;
}
.app-header .subtitle {
    font-size: 12.5px; color: #6b7280; margin: 0;
}
.stamp-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; border-radius: 50%;
    background: #FBEAEA; border: 2px solid #A32D2D;
    transform: rotate(-8deg); margin-bottom: 8px;
}
.stamp-badge .num {font-size: 15px; font-weight: 700; color: #791F1F; line-height: 1;}
.stamp-badge .lbl {font-size: 7px; color: #791F1F; letter-spacing: 0.3px;}
.lang-pill {
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 14px; margin-bottom: 8px;
    background: #FDF3E3; color: #7A4A0A;
}
.source-footer {
    margin-top: 10px; font-size: 11.5px; color: #6b7280;
    border-top: 1px dashed rgba(49,51,63,0.15); padding-top: 6px;
}
</style>
""", unsafe_allow_html=True)

load_dotenv()

# --- Configuration and Initialization ---

# Set Groq API Key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Initialize embeddings model
def get_embeddings_model():
    """Caches the HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


embeddings = get_embeddings_model()


# Initialize Chroma vector store
def get_vector_store(embed_func):
    """Caches the Chroma vector store."""
    try:
        return Chroma(persist_directory=config.CHROMA_PERSIST_DIRECTORY, embedding_function=embed_func)
    except Exception as e:
        st.error(f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop()  # Stop the app if DB cannot be loaded


vectordb = get_vector_store(embeddings)


# Initialize the ChatGroq model
def get_chat_model():
    """Caches the ChatGroq model."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,  # Lowering temperature for more consistent answers
        max_tokens=2048  # Increased so 16/20 mark detailed answers aren't cut off
    )


model = get_chat_model()

# --- Mark-based Answer Length Configuration ---
# Maps common exam mark values to how long/detailed the answer should be.
# This lets the same chatbot give a 2-line answer for a 2 mark question and
# a full detailed, structured answer for a 16 or 20 mark question.
MARK_LENGTH_GUIDE = {
    1:  "Answer in just 1-2 lines. Give only the direct, precise answer, no extra explanation.",
    2:  "Answer in 2-3 lines. Be short and precise, only the key point(s), no lengthy explanation.",
    3:  "Answer in 3-4 lines. Give a short, clear explanation covering the main idea, with one example if needed.",
    5:  "Answer in about 6-8 lines. Include a short definition, 2-3 key points, and a brief example.",
    8:  "Answer in about 10-15 lines. Include definition, detailed explanation broken into points/sub-points, and an example.",
    10: "Answer in about 12-18 lines. Include definition, detailed explanation with points, and an example or diagram description.",
    12: "Answer in about 15-20 lines. Include an introduction, detailed explanation with clear points/sub-points, and examples.",
    15: "Answer in detail, about 20-25 lines, with introduction, well-structured points/sub-points, examples, and a short conclusion.",
    16: "Answer in full detail, about 25-35 lines. Include an introduction, detailed explanation with headings and sub-points, "
        "examples, diagram description (in words) if relevant, and a conclusion.",
    20: "Answer in complete, essay-type detail, about 35-50 lines. Include an introduction, multiple headings with sub-points, "
        "detailed explanation, examples, diagram description (in words) if relevant, advantages/disadvantages if applicable, "
        "and a proper conclusion.",
}


# Any mark value not listed above (e.g. 4, 6, 7, 9, 14...) falls back to the
# nearest lower mark's guideline, so every mark value from 1-20+ is covered.
def get_length_instruction(prompt: str) -> str:
    """
    Looks for a mark value (e.g. '5 marks', '16marks', '8 marks kku', '20m')
    in the student's question and returns the matching answer-length instruction.
    If no mark value is found, a moderate default length is used.
    """
    match = re.search(r'(\d{1,3})\s*(?:-|\s)?\s*(?:marks?|m\b)', prompt, flags=re.IGNORECASE)
    if not match:
        return (
            "No specific mark value was mentioned, so answer in a moderate, student-friendly length "
            "(about 4-6 lines) covering the key point(s) clearly."
        )

    marks = int(match.group(1))
    if marks in MARK_LENGTH_GUIDE:
        return MARK_LENGTH_GUIDE[marks]

    # Fallback: use the closest defined mark value that is <= the requested marks,
    # or the smallest defined value if the requested marks is below all keys.
    defined_marks = sorted(MARK_LENGTH_GUIDE.keys())
    closest = defined_marks[0]
    for m in defined_marks:
        if m <= marks:
            closest = m
    return MARK_LENGTH_GUIDE[closest]


def get_marks_value(prompt: str) -> int:
    """Extracts the raw mark number mentioned in the question, or 0 if none found."""
    match = re.search(r'(\d{1,3})\s*(?:-|\s)?\s*(?:marks?|m\b)', prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def get_retrieval_k(marks: int) -> int:
    """
    Higher mark questions need more supporting context chunks to write a
    longer, well-rounded answer.
    """
    if marks >= 16:
        return 6
    elif marks >= 8:
        return 5
    elif marks >= 5:
        return 4
    return 3


# --- Multi-language detection (English / Tamil / Tanglish) ---

# Tamil unicode block: U+0B80 to U+0BFF
_TAMIL_SCRIPT_RE = re.compile(r'[\u0B80-\u0BFF]')

# Common Tanglish (Tamil written using English letters) marker words.
# These are everyday spoken-Tamil words that students commonly type in
# English script, so if we see several of these we treat the question
# as Tanglish rather than plain English.
_TANGLISH_MARKERS = {
    "enna", "eppadi", "epdi", "irukku", "iruku", "venum", "vendum", "panna",
    "pannunga", "pannuga", "panren", "pannu", "illa", "yaaru", "yenga",
    "enga", "epo", "eppo", "nalla", "seri", "sari", "vandhu", "vanthu",
    "poidum", "romba", "solunga", "sollunga", "kekkalam", "kettal",
    "kettalum", "vendam", "irundha", "irunthu", "irukkum", "solla",
    "paathu", "pathu", "edhuku", "yedhuku", "yenna", "semma", "unga",
    "ungaluku", "namma", "enakku", "ethu", "athu", "ithu", "kku", "la",
    "ku", "nu", "aa", "explain pannu", "sollu", "theriyum", "theriyala",
    "korachu", "konjam", "vera", "aprm", "apo", "mattum",
}


def detect_language(text: str) -> str:
    """
    Detects whether the student's question is in Tamil script, Tanglish
    (Tamil words typed using English letters), or English, so the answer
    can be generated back in the same language/style.
    """
    if _TAMIL_SCRIPT_RE.search(text):
        return "Tamil"

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if words:
        tanglish_hits = sum(1 for w in words if w in _TANGLISH_MARKERS)
        # If a decent share of the words look like Tanglish markers,
        # treat the whole question as Tanglish.
        if tanglish_hits >= 1 and (tanglish_hits / len(words)) >= 0.12:
            return "Tanglish"

    return "English"


def get_language_instruction(language: str) -> str:
    """Builds the instruction telling the model which language/style to answer in."""
    if language == "Tamil":
        return (
            "The student asked the question in Tamil (Tamil script). "
            "Write the ENTIRE answer in Tamil script (\u0b95\u0bc1 \u0b95\u0bcd\u0b95\u0bb3\u0bcd), "
            "including headings, points and the conclusion. Do not switch to English."
        )
    if language == "Tanglish":
        return (
            "The student asked the question in Tanglish (spoken Tamil written using English letters). "
            "Write the ENTIRE answer in the same Tanglish style - Tamil words spelled out using English letters, "
            "in a natural, student-friendly spoken-Tamil tone. Do not switch to pure English or to Tamil script."
        )
    return (
        "The student asked the question in English. Write the ENTIRE answer in clear, simple English."
    )


# Define the LangGraph node function
def call_model(state: MessagesState):
    """
    This function defines the 'model' node in the LangGraph workflow.
    It takes the current state (conversation messages) and invokes the LLM.
    """
    system_prompt = (
        "You are a student-friendly academic assistant for question-answering tasks. "
        "Use the retrieved context provided to answer the student's question accurately, in your own words. "
        "Each question will come with a length/detail instruction based on how many marks it is worth "
        "(for example: 2, 3, 5, 8, 16, 20 marks) - follow that instruction exactly. "
        "Short mark questions (1-3 marks) need only a few precise lines. "
        "Medium mark questions (5-10 marks) need a short structured explanation with points and an example. "
        "High mark questions (12-20 marks) need a detailed, well-structured, essay-style answer with headings, "
        "sub-points, examples, and a conclusion. "
        "Each question will also come with a LANGUAGE instruction telling you which language/style "
        "(English, Tamil, or Tanglish) to answer in - follow that instruction exactly, no matter which "
        "language the retrieved context itself is written in. "
        "If you don't know the answer from the context, just say that you don't know honestly, "
        "in the same language/style as the question."
    )
    # Prepend the system message to the current conversation history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Build and compile the LangGraph workflow
@st.cache_resource
def get_langgraph_app():
    """Caches and compiles the LangGraph workflow."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add simple in-memory checkpointer for conversation history
    memory = MemorySaver()
    langgraph_app = workflow.compile(checkpointer=memory)
    return langgraph_app


app = get_langgraph_app()


def format_source_info(docs):
    """
    Builds the 'Source Document' and 'Reference Page Numbers' strings.

    Fixes the old bug where these always showed 'N/A': it now reads the
    metadata safely (falls back to 'N/A' only when a value is genuinely
    missing), shows just the file name instead of the full path, and
    converts 0-indexed PDF page numbers into human-friendly 1-indexed pages,
    sorted numerically and de-duplicated.
    """
    if not docs:
        return "N/A", "N/A"

    # Source document: take it from the top (most relevant) match.
    raw_source = docs[0][0].metadata.get("source")
    source_document = os.path.basename(raw_source) if raw_source else "N/A"

    # Page numbers: collect from every retrieved chunk, not just the first.
    pages = []
    for doc, _score in docs:
        page = doc.metadata.get("page")
        if page is not None:
            try:
                pages.append(int(page) + 1)  # PyPDFLoader pages are 0-indexed
            except (TypeError, ValueError):
                continue

    unique_pages = sorted(set(pages))[:3]
    page_numbers_str = ", ".join(str(p) for p in unique_pages) if unique_pages else "N/A"

    return source_document, page_numbers_str


# --- Chat history state (multiple chats, pin/unpin, revisit) ---

def new_chat(title: str = "New chat"):
    """Creates a new chat session and makes it the active one."""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": title,
        "messages": [],
        "pinned": False,
        "thread_id": chat_id,
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    new_chat()

current_chat = st.session_state.chats[st.session_state.current_chat_id]

# --- Sidebar: chat history with pin/unpin ---

with st.sidebar:
    st.title("\U0001F4AC Chats")

    if st.button("\u2795 New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    # Pinned chats first, then the rest, most recently created first.
    chat_items = list(st.session_state.chats.items())
    pinned_items = [c for c in chat_items if c[1]["pinned"]]
    other_items = [c for c in chat_items if not c[1]["pinned"]]

    def render_chat_row(chat_id: str, chat: dict):
        is_active = chat_id == st.session_state.current_chat_id
        cols = st.columns([5, 1, 1])
        label = ("\u25CF " if is_active else "") + chat["title"]
        if cols[0].button(label, key=f"open_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.rerun()
        pin_icon = "\U0001F4CC" if chat["pinned"] else "\U0001F4CD"
        if cols[1].button(pin_icon, key=f"pin_{chat_id}", help="Unpin" if chat["pinned"] else "Pin"):
            chat["pinned"] = not chat["pinned"]
            st.rerun()
        if cols[2].button("\U0001F5D1", key=f"del_{chat_id}", help="Delete chat"):
            del st.session_state.chats[chat_id]
            if st.session_state.current_chat_id == chat_id:
                if st.session_state.chats:
                    st.session_state.current_chat_id = next(iter(st.session_state.chats))
                else:
                    new_chat()
            st.rerun()

    if pinned_items:
        st.caption("Pinned")
        for chat_id, chat in pinned_items:
            render_chat_row(chat_id, chat)
        st.divider()

    st.caption("History")
    if not other_items:
        st.caption("No other chats yet.")
    for chat_id, chat in reversed(other_items):
        render_chat_row(chat_id, chat)

st.markdown("""
<div class="app-header">
  <div class="logo-circle">🎓</div>
  <div>
    <p class="title">Academic QA</p>
    <p class="subtitle">Ask in English, Tamil or Tanglish — the answer comes back in the same language.</p>
  </div>
</div>
""", unsafe_allow_html=True)


def render_assistant_message(msg: dict):
    """Renders an assistant answer with a marks 'stamp' badge, a detected-language
    pill, and a clean source/page footer — instead of a flat markdown blob."""
    marks = msg.get("marks", 0)
    language = msg.get("language", "English")
    answer = msg.get("answer", msg.get("content", ""))
    source = msg.get("source", "N/A")
    pages = msg.get("pages", "N/A")

    top_cols = st.columns([1, 5])
    with top_cols[0]:
        if marks:
            st.markdown(
                f'<div class="stamp-badge"><div style="text-align:center;">'
                f'<div class="num">{marks}</div><div class="lbl">MARKS</div></div></div>',
                unsafe_allow_html=True,
            )
    with top_cols[1]:
        st.markdown(f'<span class="lang-pill">{language}</span>', unsafe_allow_html=True)

    st.markdown(answer)
    st.markdown(
        f'<div class="source-footer">📄 <b>Source Document:</b> {source} '
        f'&nbsp;·&nbsp; 📑 <b>Reference Page Numbers:</b> {pages}</div>',
        unsafe_allow_html=True,
    )


# Display previous messages of the active chat
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])


# --- Chat Input and Logic ---

if prompt := st.chat_input("Ask any question..."):
    # Auto-title the chat from the first question asked in it
    if not current_chat["messages"]:
        current_chat["title"] = (prompt[:40] + "...") if len(prompt) > 40 else prompt

    # Add user message to chat history and display
    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Detect the question's language so the answer matches it
                detected_language = detect_language(prompt)
                language_instruction = get_language_instruction(detected_language)

                # 2. Retrieve context for the current user question
                # Retrieval depth scales with the mark value (16/20 mark answers get more context)
                marks_value = get_marks_value(prompt)
                retrieval_k = get_retrieval_k(marks_value)
                docs = vectordb.similarity_search_with_score(prompt, k=retrieval_k)
                current_context = "\n\n".join(doc[0].page_content for doc in docs) if docs else ""

                # 3. Build the answer-length instruction matching the detected mark value
                # (2, 3, 5, 8, 16, 20, etc.)
                length_instruction = get_length_instruction(prompt)

                # 4. Construct the HumanMessage for the current turn, including context,
                # the length instruction, and the language instruction.
                current_turn_message = HumanMessage(
                    content=(
                        f"Context: {current_context if current_context else 'No relevant context was found.'}\n\n"
                        f"Answer length instruction: {length_instruction}\n\n"
                        f"Language instruction: {language_instruction}\n\n"
                        f"Question: {prompt}"
                    )
                )

                # 5. Invoke the LangGraph app with the new message and this chat's own thread_id
                # LangGraph's checkpointer handles loading previous state and appending this message.
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": current_chat["thread_id"]}},
                )

                # Get the AI's response (the last AIMessage in the result)
                ai_response = result['messages'][-1].content

                # Extract source document and page numbers (properly falls back to
                # N/A only when nothing relevant was actually retrieved)
                source_document, page_numbers_str = format_source_info(docs)

                assistant_msg = {
                    "role": "assistant",
                    "answer": ai_response,
                    "marks": marks_value,
                    "language": detected_language,
                    "source": source_document,
                    "pages": page_numbers_str,
                }

                render_assistant_message(assistant_msg)

                # Add AI response to this chat's history
                current_chat["messages"].append(assistant_msg)

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                current_chat["messages"].append({"role": "assistant", "answer": "I encountered an error. Please try again.", "marks": 0, "language": "English", "source": "N/A", "pages": "N/A"})

# To run this Streamlit app:
# 1. Save the code above as a Python file (e.g., `app.py`).
# 2. Make sure you have your 'docs/chroma/' directory correctly set up with your vector store.
# 3. Run from your terminal: `streamlit run app.py`
