import ollama
import chromadb
import pdfplumber

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL_4B = "gemma3:4b"
CHAT_MODEL_8B = "llama3.1:latest"

#  Load and extract text from PDF
#  simple text extraction via pdfplumber for now,

def load_pdf_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    return pages

#   CHUNK text for embedding and indexing
def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# Embed and store text chunks in ChromaDB
client = chromadb.PersistentClient(path="rag_db")
collection = client.get_or_create_collection("docs")

def index_pdf(path):
    ids, docs , metas = [],[],[]
    for page_num, text in load_pdf_text(path):
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path}-p{page_num}-c{j}")
            docs.append(chunk)
            metas.append({"source": path, "page": page_num})
    embeddings = ollama.embed(model=EMBED_MODEL, input=docs)["embeddings"]
    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} chunks from {path}")


# Retrieve and generate
# Values to be adjusted/ config file?
def ask(question, k=4):
    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    results = collection.query(query_embeddings=q_emb, n_results=k)

    context = "\n\n--\n\n".join(
        f"[page {m['page']}]\n{d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    )

    # Prompt for Eval, keeping it simple for now

    prompt = (
        "Answer the question using only the context below, "
        " If the answer is not contained within the context, say 'I don't know'."
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = ollama.chat(model=CHAT_MODEL_4B,messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


if __name__ == "__main__":
    pdf_path = r"C:\Users\Albert Kaczan\PycharmProjects\PDF_to_RAG\data\quarterly_report.pdf"

    if collection.count() == 0:
        index_pdf(pdf_path)

    print("Ask questions about the PDF. Type 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        print("Model:", ask(question), "\n")