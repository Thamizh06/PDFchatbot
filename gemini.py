import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Load and split PDF
def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    return chunks


# Create FAISS vector store
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    return vectorstore


# Create QA chain using Gemini
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        transport="rest" 
    )


    prompt_template = """
    You are a document-based question answering assistant.

    RULES:
    - Answer directly and concisely.
    - Do NOT add introductions, explanations, or commentary.
    - Do NOT say phrases like:
      "Based on the document",
      "Given this information",
      "The document appears to".
    - If the answer is not present in the context, say exactly:
      "Answer not found in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer (direct, concise):
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# Ask the chatbot
def ask_bot(qa_chain, query: str):
    result = qa_chain.invoke({"query": query})

    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }
