
## Import Modules
import os  # Interacts with the operating system
from langchain_community.document_loaders import PyPDFLoader  # Loads and parses PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain_chroma import Chroma  # Manages vector stores for document embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Generates embeddings using Google’s Generative AI
from dotenv import load_dotenv  # Loads environment variables from .env file


load_dotenv()  # Load environment variables from the .env file
api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve the Google API key from environment variables
if not api_key:
    raise ValueError("No GOOGLE_API_KEY found in the environment variables.")



loader = PyPDFLoader("Fuji_xs20_manual.pdf")  # Initialize the PDF loader with the specified PDF file
data = loader.load()  # Load the entire PDF as a single Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # Initialize the text splitter
docs = text_splitter.split_documents(data)  # Split the loaded data into smaller documents

print("Total number of documents: ", len(docs))  # Print the total number of documents


api_key = os.getenv("GOOGLE_API_KEY")  # Retrieve the Google API key from environment variables
# Embedding models: https://python.langchain.com/v0.1/docs/integrations/text_embedding/
embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")  # Initialize the embedding model with the API key and specified model
vector = embeddings.embed_query("hello, world!")  # Generate an embedding vector for a sample query
print(vector[:5])  # Print the first five elements of the embedding vector


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)  # Create a Chroma vector store from the split documents using the embeddings
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  # Set up a retriever to find similar documents based on queries


retrieved_docs = retriever.invoke("what are film simulation modes and how to set it in Xs20 camera?")  # Use the retriever to find documents relevant to the query
print("Number of retrieved documents: ", len(retrieved_docs))  # Print the number of documents retrieved
print(retrieved_docs[5].page_content)  # Print the content of the 6th retrieved document


from langchain_google_genai import ChatGoogleGenerativeAI  # Import the ChatGoogleGenerativeAI class for conversational AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)  # Initialize the language model with specified parameters

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)  # Define the system prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System-level instructions
        ("human", "{input}"),  # User input placeholder
    ]
)  # Create a prompt template

question_answer_chain = create_stuff_documents_chain(llm, prompt)  # Create a question-answering chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  # Create a Retrieval-Augmented Generation (RAG) chain

response = rag_chain.invoke({"input": "what is new in YOLOv9?"})  # Invoke the RAG chain with a query
print(response["answer"])  # Print the assistant's answer

