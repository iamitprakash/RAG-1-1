import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import sys

load_dotenv()

# Option to use local embeddings instead of OpenAI (set USE_LOCAL_EMBEDDINGS=true in .env)
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def get_embedding_model():
    """Get embedding model - either OpenAI or local"""
    if USE_LOCAL_EMBEDDINGS:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("Using local HuggingFace embeddings (sentence-transformers)...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return embedding_model
        except ImportError:
            print("ERROR: HuggingFace embeddings not installed.")
            print("Install it with: pip install sentence-transformers")
            sys.exit(1)
    else:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not found in environment variables.")
            print("Please set it in your .env file or use local embeddings by setting USE_LOCAL_EMBEDDINGS=true")
            sys.exit(1)
        
        print("Using OpenAI embeddings...")
        return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    try:
        embedding_model = get_embedding_model()
        
        # Create ChromaDB vector store
        print("--- Creating vector store ---")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("--- Finished creating vector store ---")
        
        print(f"Vector store created and saved to {persist_directory}")
        return vectorstore
    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        if "RateLimitError" in error_type or "429" in error_msg or "quota" in error_msg.lower():
            print("\n❌ ERROR: OpenAI API quota exceeded or rate limit reached.")
            print("\nOptions to fix this:")
            print("1. Check your OpenAI billing and quota at https://platform.openai.com/account/billing")
            print("2. Use local embeddings instead by setting USE_LOCAL_EMBEDDINGS=true in your .env file")
            print("   Then install: pip install sentence-transformers")
            print("3. Wait and try again later if it's a rate limit issue")
        elif "AuthenticationError" in error_type or "401" in error_msg or "authentication" in error_msg.lower():
            print("\n❌ ERROR: OpenAI API authentication failed.")
            print("Please check your OPENAI_API_KEY in your .env file")
        elif "insufficient_quota" in error_msg.lower():
            print("\n❌ ERROR: Insufficient OpenAI quota.")
            print("Please check your plan and billing details at https://platform.openai.com/account/billing")
            print("\nTo use local embeddings instead, set USE_LOCAL_EMBEDDINGS=true in your .env file")
        else:
            print(f"\n❌ ERROR: {error_type}: {error_msg}")
            print("\nIf you're having API issues, consider using local embeddings:")
            print("Set USE_LOCAL_EMBEDDINGS=true in your .env file")
            print("Then install: pip install sentence-transformers")
        
        raise

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        try:
            embedding_model = get_embedding_model()
            vectorstore = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_model, 
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
            return vectorstore
        except Exception as e:
            print(f"\n❌ ERROR loading existing vector store: {e}")
            print("You may need to recreate it. Consider using local embeddings if you have API issues.")
            raise
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    #  Step 3: Create vector store
    vectorstore = create_vector_store(chunks, persistent_directory)
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()