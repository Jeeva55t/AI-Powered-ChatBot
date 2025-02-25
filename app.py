import streamlit as st
import pdfplumber
from PyPDF2.errors import PdfReadError
import docx
import fitz
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes 
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






import streamlit as st
import pdfplumber
import docx
import fitz
from PyPDF2.errors import PdfReadError
from pdf2image import convert_from_bytes  # 🆕 Convert PDFs to images
import pytesseract  # 🆕 OCR for extracting text from images

def get_pdf_text(files):
    text = ""
    for file in files:
        file_name = file.name.lower()

        # 📝 Handle PDF files
        if file_name.endswith(".pdf"):
            try:
                pdf_reader = fitz.open(stream=file.read(), filetype="pdf")  
                for page in pdf_reader:
                    text += page.get_text("text")  # Extract text
            except Exception:
                st.warning(f"Warning: {file.name} has issues, trying alternative method...")
                try:
                    with pdfplumber.open(file) as pdf_plumber:
                        for page in pdf_plumber.pages:
                            extracted_text = page.extract_text()
                            if extracted_text:
                                text += extracted_text
                except Exception as e:
                    st.error(f"Failed to read {file.name}: {e}")

            # 🆕 If no text was extracted, apply OCR
            if not text.strip():
                st.warning(f"No text detected in {file.name}, applying OCR...")
                try:
                    images = convert_from_bytes(file.getvalue())  # Convert PDF to images
                    for img in images:
                        text += pytesseract.image_to_string(img)  # Extract text using OCR
                except Exception as ocr_error:
                    st.error(f"OCR failed for {file.name}: {ocr_error}")

        # 📄 Handle DOCX files
        elif file_name.endswith(".docx"):
            try:
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            except Exception as e:
                st.error(f"Failed to read {file.name}: {e}")

        else:
            st.error(f"Unsupported file format: {file.name}. Please upload PDF or DOCX.")

    if not text.strip():
        st.error("Could not extract any text. The document might be unreadable.")

    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the 
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("AI-Powered PDF Chat Application")
    st.write("This is an AI-Powered PDF Chat Application that can answer your questions from the provided PDF Files.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()