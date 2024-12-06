from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Step 2: Preprocess text
def preprocess_text(text):
    # Split into sentences for better context
    return text.split(". ")


# Step 3: Create embeddings for the PDF text
def create_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight embedding model
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings, model


# Step 4: Find relevant context for the question
def find_relevant_context(question, sentences, embeddings, model):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, embeddings)
    best_score_idx = scores.argmax().item()
    return sentences[best_score_idx]


# Step 5: Extract the answer from the context
def extract_answer(question, context):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")
    result = qa_pipeline(question=question, context=context)
    return result['answer']


# Main function to ask questions
def ask_question_from_pdf(pdf_path, question):
    # Extract and preprocess text
    pdf_text = extract_text_from_pdf(pdf_path)
    sentences = preprocess_text(pdf_text)

    # Create embeddings
    embeddings, model = create_embeddings(sentences)

    # Find relevant context
    context = find_relevant_context(question, sentences, embeddings, model)

    # Extract the answer
    answer = extract_answer(question, context)
    return answer, context


# Usage example
pdf_path = "sample.pdf"  # Path to your PDF
question = "Just give me the IBAN number?"

answer, context = ask_question_from_pdf(pdf_path, question)
print("Question:", question)
print("Answer:", answer)
print("Context:", context)
