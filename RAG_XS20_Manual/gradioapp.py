# Import Modules
import os  # Interacts with the operating system
import gradio as gr  # Web framework for interactive applications
from langchain_community.document_loaders import PyPDFLoader  # Loads and parses PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain_chroma import Chroma  # Manages vector stores for document embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Generates embeddings using Google’s Generative AI
from dotenv import load_dotenv  # Loads environment variables from .env file
from langchain_google_genai import ChatGoogleGenerativeAI  # For conversational AI
from langchain.chains import create_retrieval_chain  # Creates retrieval chains
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines document processing
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve the Google API key from environment variables
if not api_key:
    raise ValueError("No GOOGLE_API_KEY found in the environment variables.")

# PDF Loader
loader = PyPDFLoader("Fuji_xs20_manual.pdf")  # Initialize the PDF loader with the specified PDF file
data = loader.load()  # Load the entire PDF as a single Document

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # Initialize the text splitter
docs = text_splitter.split_documents(data)  # Split the loaded data into smaller documents
print("Total number of documents: ", len(docs))  # Display the total number of documents

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")  # Initialize the embedding model
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)  # Create a Chroma vector store from the split documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  # Set up the retriever

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)  # Initialize the language model

# Define the system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System-level instructions
        ("human", "{input}"),  # User input placeholder
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  # Create a Retrieval-Augmented Generation (RAG) chain

# Function to handle user queries
def answer_query(query):
    if query:
        response = rag_chain.invoke({"input": query})  # Invoke the RAG chain with the user query
        return response["answer"]  # Return the answer from the assistant

# Gradio Interface
iface = gr.Interface(
    fn=answer_query,  # Function to call for generating the response
    inputs="text",  # Input type for the query
    outputs="text",  # Output type for the answer
    title="RAG Application Built on Gemini Model",
    description="Ask any question about your Fuji X-S20 camera"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()

