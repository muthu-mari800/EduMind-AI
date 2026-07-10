<<<<<<< HEAD
# RAG - QA

This repository contains a Retrieval Augmented Generation (RAG) application. This application allows you to ingest data and then interact with it through a user-friendly interface.

## Prerequisites

In this project I have used `uv` to install, please feel free to use any package manager to install.

### Install uv (Python package manager):

`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Verify the installation:

`uv --version`

## Clone the repository or Download as zip file

`git clone https://github.com/akash-balakrishnan-22/RAG.git`

## Environment Setup

### Create a Virual Environment

`uv venv`

#### Activate the environment

`source .venv/bin/activate`

#### Install Dependencies

`uv sync`

## Running the Application

### 1. Data Ingestion

Run the data ingestion script to prepare your data:
`uv run ingest.py`

### 2. Start the Application

Add your Groq API key in the .env file
Launch the Streamlit application:
`uv run streamlit run app.py`

## Note:

- Make sure you have the required data files in place before running the ingestion script and change the `config.py` accordingly.
- The application will be available in your browser once Streamlit starts.
- To deactivate the virtual environment simply run `deactivate`.
=======
# AI-ACADEMIC-SYSTEM-USING-RAG
>>>>>>> 152d1d7e2eda4db4c7ace849f2ebc9ca76283f9e
