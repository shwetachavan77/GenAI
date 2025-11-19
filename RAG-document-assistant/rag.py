# imports
import logging
import ollama
import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader  #Langchain PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import MultiQueryRetriever


logging.basicConfig(level=logging.INFO)

doc_path = "./data/1706.03762v7.pdf"
model_name = "llama3.2"
embedding_model = "nomic-embed-text"
vector_store_name = "simple-rag"
persist_directory = "./chroma_db"

def ingest_pdf(doc_path):
    '''Ingesting PDF document'''

    if os.path.exists(doc_path):
        loader = PyMuPDFLoader(doc_path)
        data = loader.load()
        logging.info("PDF loaded")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None
    
def split_document_to_chunks(document):
    '''Chunking'''

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(document)
    logging.info("Documents split into chunks")
    return chunks
    
@st.cache_resource
def create_vdb():
    '''Create/Load Vector DB with embeddings'''
    
    ollama.pull(embedding_model)
    embedding = OllamaEmbeddings(model=embedding_model)
    rebuild_db = False
    if os.path.exists(persist_directory):
        try:
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=vector_store_name,
                persist_directory=persist_directory,
            )
            logging.info("Loaded existing vector database")
        except Exception as e:
            logging.warning(f"Failed to load existing DB: {e}")
            rebuild_db = True
    else:
        rebuild_db = True

    if rebuild_db:
        data = ingest_pdf(doc_path)
        if data is None:
            return None

        chunks = split_document_to_chunks(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=vector_store_name,
            persist_directory=persist_directory,
        )
        vector_db.persist()
        logging.info("Vector Database created and persisted")

    return vector_db


def create_retriever(vdb,llm):
    '''Creating a Multi Query Retriever'''

    query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. 
    Your task is to generate five different versions of the given 
    user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, 
    your goal is to help the user to help the user overcome some of the 
    limitations of the distance-based similarity search. 
    Provide these alternative questions seperated by newlines. 
    Original questions: {question}
    """
    )   
    retriever = MultiQueryRetriever.from_llm(vdb.as_retriever(), llm, prompt=query_prompt)
    logging.info("Retriever Created")
    return retriever

def create_chain(retriever, llm):
    '''Creating Chain'''

    template = """Answer the question based ONLY on the following {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm   
        | StrOutputParser()
    )
    logging.info("Chain created")
    return chain

# '''Inline version'''
# def main():
#     data = ingest_pdf(doc_path)
#     if data is None:
#         return
#     chunks = split_document_to_chunks(data)
#     vector_db = create_vdb(chunks)
#     llm = ChatOllama(model=model_name)
#     retriever = create_retriever(vector_db, llm)
#     chain = create_chain(retriever, llm)

#     question = "Give me abrief summary of whay is htis document about?"

#     res = chain.invoke(input=question)
#     print("Response:", res)

# Streamlit version
def main():
    '''Streamlit Version'''

    st.title("Document RAG Assistant")
    user_input = st.text_input("Enter your question: ")

    if user_input:
        with st.spinner("Generating response"):
            try:
                llm = ChatOllama(model=model_name)
                vector_db = create_vdb()
                if vector_db is None:
                    st.error("Failed to load or create vector db")
                    return 
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)

                res = chain.invoke(input=user_input)
                st.markdown("**Assistant**")
                st.write(res)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
                st.info("Please enter a question to get started")


if __name__ == "__main__":
    main()
