import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
import os
import requests


load_dotenv()


groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


JINA_API_KEY = os.environ.get("JINA_API_KEY")


chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="pdf_collection"
)


def get_embeddings(texts):

    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "jina-embeddings-v5-text-small",
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return [
        item["embedding"]
        for item in data["data"]
    ]


def get_query_embedding(query):

    embeddings = get_embeddings([query])

    return embeddings[0]


def read_pdf(file):

    reader = PdfReader(file.name)

    texts = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts)


def process_pdf(file):

    if file is None:
        return

    try:

        full_text = read_pdf(file)

        if not full_text.strip():
            return

        existing = collection.get()

        if existing["ids"]:

            collection.delete(
                ids=existing["ids"]
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(
            full_text
        )

        if not chunks:
            return

        vectors = []

        batch_size = 32

        for i in range(
            0,
            len(chunks),
            batch_size
        ):

            batch = chunks[
                i:i + batch_size
            ]

            batch_vectors = get_embeddings(
                batch
            )

            vectors.extend(
                batch_vectors
            )

        ids = [
            str(i)
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors
        )

    except Exception:
        return


def too_many_questions(query):

    question_words = [
        "what",
        "which",
        "how",
        "who",
        "why",
        "when",
        "where",
        "mention",
        "list"
    ]

    words = query.lower().split()

    count = sum(
        1
        for word in words
        if word.strip(
            ".,?!:;()[]{}"
        ) in question_words
    )

    return count > 3


def query_file(query):

    if not query.strip():

        return "Please enter a question."

    if too_many_questions(query):

        return (
            "Please ask a maximum of 3 questions at a time."
        )

    existing = collection.get()

    if not existing["ids"]:

        return "Please upload a PDF first."

    try:

        query_vector = get_query_embedding(
            query
        )

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5
        )

        semantic_documents = results[
            "documents"
        ][0]

        all_documents = existing[
            "documents"
        ]

        words = [
            word.lower().strip(
                ".,?!:;()[]{}"
            )
            for word in query.split()
            if len(
                word.strip(
                    ".,?!:;()[]{}"
                )
            ) > 3
        ]

        keyword_documents = []

        for document in all_documents:

            document_lower = document.lower()

            matches = sum(
                1
                for word in words
                if word in document_lower
            )

            if matches > 0:

                keyword_documents.append(
                    (matches, document)
                )

        keyword_documents.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        keyword_documents = [
            document
            for _, document
            in keyword_documents[:3]
        ]

        documents = []

        for document in semantic_documents:

            if document not in documents:

                documents.append(document)

        for document in keyword_documents:

            if document not in documents:

                documents.append(document)

        context = "\n\n".join(
            documents
        )

        prompt = f"""
Answer the user's question using ONLY the information in the CONTEXT.

Rules:

1. Answer every part of the question.
2. If the question contains multiple parts, answer each part.
3. Do not stop after answering only the first part.
4. Keep the answer concise but complete.
5. Use only information present in the CONTEXT.
6. Do not use your own knowledge.
7. Do not guess or invent information.
8. If the answer is not present in the CONTEXT, say:
The answer is not available in the provided document.
9. Do not repeat the question.
10. Do not give unnecessary explanations.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        response = groq_client.chat.completions.create(

            model="qwen/qwen3.8-27b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=300,

            reasoning_effort="none"
        )

        return response.choices[
            0
        ].message.content.strip()

    except Exception as e:

        return f"Error generating answer: {str(e)}"


def chat_with_pdf(
    query,
    history
):

    answer = query_file(
        query
    )

    history = history or []

    history.append(
        {
            "role": "user",
            "content": query
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return history, ""


def get_pages(file):

    if file is None:

        return ""

    try:

        reader = PdfReader(
            file.name
        )

        return len(
            reader.pages
        )

    except:

        return ""


def get_upload_time(file):

    if file is None:

        return ""

    return datetime.now().strftime(
        "%I:%M %p"
    )


with gr.Blocks() as app:

    gr.Markdown(
        "# RAG Chatbot"
    )

    with gr.Row():

        with gr.Column(scale=1):

            file = gr.File(
                label="Upload your PDF",
                file_types=[".pdf"]
            )

            page_textbox = gr.Textbox(
                label="Number of pages",
                interactive=False
            )

            time_textbox = gr.Textbox(
                label="Time of upload",
                interactive=False
            )

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                height=500
            )

            query_textbox = gr.Textbox(
                label="Ask any question from the file",
                lines=3
            )

            query_button = gr.Button(
                "Search"
            )

    file.change(
        get_pages,
        inputs=file,
        outputs=page_textbox
    )

    file.change(
        get_upload_time,
        inputs=file,
        outputs=time_textbox
    )

    file.change(
        process_pdf,
        inputs=file
    )

    query_button.click(
        chat_with_pdf,
        inputs=[
            query_textbox,
            chatbot
        ],
        outputs=[
            chatbot,
            query_textbox
        ]
    )


app.launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get(
            "PORT",
            10000
        )
    )
)