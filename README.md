# RAG Research Assistant

A simple Retrieval-Augmented Generation (RAG) application built using Flask and Anthropic Claude API.  
Users can upload documents and ask questions based on their content.

## Features

- Upload PDF, TXT, and DOCX files
- Document-based question answering
- Claude API integration
- Metadata extraction
- Simple and responsive UI

## Tech Stack

- Python
- Flask
- Anthropic Claude API
- PyPDF2
- python-docx

## Installation

### Clone Repository

```bash
git clone https://github.com/ann29-tdk/rag_app.git
cd rag_app
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variable

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

## Run Application

```bash
python app_flask.py
```

Open in browser:

```text
http://localhost:5000
```

## Supported File Formats

- PDF
- TXT
- DOCX

## Future Improvements

- Vector database support
- Semantic search
- Docker deployment
- Conversation memory

## License

For educational and research purposes only.
