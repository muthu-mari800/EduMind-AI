import os

from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from config import config

def load_youtube_content(url: str, save_dir: str):
    """
    Loads audio from a YouTube URL, transcribes it using FasterWhisperParser,
    and returns the loaded documents.

    Args:
        url (str): The URL of the YouTube video.
        save_dir (str): Directory to save the audio files temporarily.

    Returns:
        list: A list of documents loaded from the YouTube content.
    """
    print(f"Starting YouTube content loading from: {url}")
    try:
        loader = GenericLoader(
            YoutubeAudioLoader([url], save_dir),
            FasterWhisperParser()
        )
        youtube_docs = loader.load()
        print(f"Successfully loaded {len(youtube_docs)} documents from YouTube.")
        return youtube_docs
    except Exception as e:
        print(f"Error loading YouTube content from {url}: {e}")
        return []

def load_pdf_content(pdf_directory: str):
    """
    Loads PDF documents from a specified directory and returns them as a list of pages.

    Args:
        pdf_directory (str): The directory containing PDF files to ingest.

    Returns:
        list: A list of loaded document pages from the PDF files.
    """
    print(f"Starting PDF document loading from '{pdf_directory}'...")

    # Ensure the PDF directory exists
    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        print("Please create this directory and place your PDF files inside.")
        return []

    all_pdf_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"Loading PDF document: {filepath}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                all_pdf_docs.extend(pages)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    if not all_pdf_docs:
        print("No PDF documents found or loaded in the specified directory.")
    else:
        print(f"Loaded {len(all_pdf_docs)} pages from PDF documents.")
    return all_pdf_docs

def ingest_all_documents(
    youtube_url: str,
    youtube_save_dir: str,
    pdf_directory: str = "data",
    persist_directory: str = "docs/chroma"
):
    """
    Orchestrates the loading of YouTube and PDF documents, combines them,
    splits them into chunks, generates embeddings, and creates/persists a
    Chroma vector database.

    Args:
        youtube_url (str): The URL of the YouTube video to ingest.
        youtube_save_dir (str): Directory to save YouTube audio files temporarily.
        pdf_directory (str): The directory containing PDF files to ingest.
                             Defaults to "data".
        persist_directory (str): The directory where the Chroma vector database
                                 will be persisted. Defaults to "docs/chroma".
    """
    print("\n--- Starting overall document ingestion process ---")

    # 1. Load YouTube content
    youtube_docs = load_youtube_content(youtube_url, youtube_save_dir)

    # 2. Load PDF content
    pdf_docs = load_pdf_content(pdf_directory)

    # Combine all loaded documents
    combined_docs = youtube_docs + pdf_docs
    if not combined_docs:
        print("No documents (PDF or YouTube) were loaded. Exiting ingestion.")
        return

    print(f"\nTotal combined documents loaded: {len(combined_docs)}")

    # 3. Document Splitters
    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP
    
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = r_splitter.split_documents(combined_docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")

    # 4. Embeddings
    model_name = config.EMBEDDING_MODEL_NAME

    print(f"Initializing embeddings with model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 5. Create and Persist Vector DB
    print(f"Creating and persisting Chroma DB to '{persist_directory}'...")
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist() # Explicitly persist the database
    print(f"Successfully processed {len(chunked_docs)} document chunks and persisted Chroma DB.")
    print(f"You can now run your application using the data in '{persist_directory}'.")
    print("--- Document ingestion process complete ---")

if __name__ == "__main__":
    
    ingest_all_documents(
        youtube_url=config.YOUTUBE_VIDEO_URL,
        youtube_save_dir=config.YOUTUBE_AUDIO_SAVE_DIRECTORY,
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )