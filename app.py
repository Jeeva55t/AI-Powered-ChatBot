import streamlit as st
import pdfplumber
import docx
import fitz
from pdf2image import convert_from_bytes
import pytesseract
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # ✅ Updated Import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# 📝 Function to Extract Text from PDFs and DOCX
def get_pdf_text(files):
    text = ""
    for file in files:
        file_name = file.name.lower()

        # 📝 Handle PDFs
        if file_name.endswith(".pdf"):
            try:
                pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf_reader:
                    text += page.get_text("text")
            except Exception:
                st.warning(f"Warning: {file.name} has issues, trying alternative method...")
                try:
                    with pdfplumber.open(file) as pdf_plumber:
                        for page in pdf_plumber.pages:
                            text += page.extract_text() or ""
                except Exception as e:
                    st.error(f"Failed to read {file.name}: {e}")

            # 🆕 OCR if No Text Extracted
            if not text.strip():
                st.warning(f"No text detected in {file.name}, applying OCR...")
                try:
                    images = convert_from_bytes(file.getvalue())
                    for img in images:
                        text += pytesseract.image_to_string(img)
                except Exception as ocr_error:
                    st.error(f"OCR failed for {file.name}: {ocr_error}")

        # 📄 Handle DOCX
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


# 🛠 Function to Chunk Text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# 📌 Function to Create Vector Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not text_chunks:
        st.error("No text data found to create vector store!")
        return None

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# 💬 Function to Load Conversational Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, respond with: 'Answer is not available in the context'.
    Do NOT generate incorrect responses.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# 🗣 Function to Process User Input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:  # 🛑 Prevents IndexError
            st.error("No relevant data found in the uploaded documents.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"Error retrieving answer: {e}")


# 🚀 Main Streamlit UI
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("AI-Powered PDF Chat Application")
    st.write("This AI can answer your questions from the provided PDF or DOCX files.")

    user_question = st.text_input("Ask a question from the uploaded documents:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF/DOCX files and click 'Submit & Process'", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)

                if vector_store:
                    st.success("Document processing completed!")


if __name__ == "__main__":
    main()
