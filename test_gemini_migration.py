# Test script for Groq API migration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test imports
print("Testing imports...")
try:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("[OK] Successfully imported Groq and HuggingFace classes")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    exit(1)

# Test API key
print("\nChecking API key...")
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"[OK] GROQ_API_KEY found (length: {len(groq_key)})")
else:
    print("[ERROR] GROQ_API_KEY not found in environment")
    exit(1)

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"[OK] OPENAI_API_KEY found (length: {len(openai_key)})")
else:
    print("[INFO] OPENAI_API_KEY not found - using HuggingFace embeddings instead")

# Test LLM initialization
print("\nTesting LLM initialization...")
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=groq_key)
    print("[OK] ChatGroq initialized successfully")
except Exception as e:
    print(f"[ERROR] LLM initialization error: {e}")
    exit(1)

# Test embeddings initialization
print("\nTesting embeddings initialization...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[OK] HuggingFaceEmbeddings initialized successfully")
except Exception as e:
    print(f"[ERROR] Embeddings initialization error: {e}")
    exit(1)

# Test simple LLM call
print("\nTesting simple LLM call...")
try:
    from langchain.schema import HumanMessage
    response = llm.invoke([HumanMessage(content="Say 'Hello from Groq!' in one sentence.")])
    print(f"[OK] LLM response: {response.content}")
except Exception as e:
    print(f"[ERROR] LLM call error: {e}")
    exit(1)

# Test embeddings
print("\nTesting embeddings...")
try:
    test_text = "This is a test for embeddings"
    embedding = embeddings.embed_query(test_text)
    print(f"[OK] Embedding generated successfully (dimension: {len(embedding)})")
except Exception as e:
    print(f"[ERROR] Embeddings error: {e}")
    exit(1)

print("\n" + "="*50)
print("[SUCCESS] All tests passed! Groq API migration successful!")
print("="*50)

