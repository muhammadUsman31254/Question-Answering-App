# Import necessary libraries
import streamlit as st
from transformers import pipeline

# Function to load the model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to process the uploaded file and display its content
def process_file(uploaded_file):
    file_contents = uploaded_file.read().decode("utf-8")
    st.text(file_contents)
    return file_contents

# Main function to run the Streamlit app
def main():
    st.title("Question Answering App")
    st.markdown("---")

    # Upload file section
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        file_contents = process_file(uploaded_file)

    st.markdown("---")

    # Question and Answer section
    st.subheader("Ask a question about the uploaded text:")
    question = st.text_input("Enter your question here:")
    if st.button("Ask"):
        if uploaded_file is not None and file_contents:
            # Load the model
            qa_model = load_qa_model()

            # Perform question answering
            answer = qa_model(question=question, context=file_contents)

            # Display the answer
            st.write("Answer:", answer["answer"])

# Run the app
if __name__ == "__main__":
    main()
