import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Ensure it is set in the environment variables.")
openai.api_key = OPENAI_API_KEY

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

def main():
    st.title("Chat with your PDF 💬")
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create the knowledge base object
        knowledge_base = process_text(text)
        
        if 'query' not in st.session_state:
            st.session_state.query = ''
        
        query = st.text_input("Ask a question to the PDF:", key="query")
        
        if st.button('Search'):
            if query:
                docs = knowledge_base.similarity_search(query)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type='stuff')
                
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                
                st.write(f"**Question:** {query}")
                st.write(f"**Answer:** {response}")
                
                # Clear the query input
               

if __name__ == "__main__":
    main()
