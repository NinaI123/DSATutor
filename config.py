"""
Configuration file for DSA Tutor System
Centralizes all settings and API keys
"""

import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
# ==================== API KEYS ====================
# Get your free Groq API key from: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional: If you want to use OpenAI for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# ==================== MODEL CONFIGURATION ====================
# Groq models (fast inference)
GROQ_MODEL = "mixtral-8x7b-32768"  # Fast and current
# GROQ_MODEL = "llama-3.1-70b-versatile"  # More capable

# If using OpenAI for embeddings
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo"

# Model temperature (creativity vs consistency)
MODEL_TEMPERATURE = 0.7

# ==================== APP CONFIGURATION ====================
APP_NAME = "DSA Tutor AI"
APP_VERSION = "1.0.0"
DEBUG_MODE = True

# Server configuration
SERVER_HOST = "0.0.0.0"  # For local access use "127.0.0.1"
SERVER_PORT = 7860
SHARE_PUBLICLY = False  # Set to True for temporary public link

# ==================== KNOWLEDGE BASE CONFIG ====================
VECTOR_STORE_TYPE = "FAISS"  # Options: "FAISS", "Chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==================== STUDENT DATA CONFIG ====================
# Storage type for student progress
STORAGE_TYPE = "json"  # Options: "json", "sqlite", "memory"
STORAGE_FILE = "student_progress.json"  # For json storage
DATABASE_URL = "sqlite:///dsa_tutor.db"  # For sqlite storage

# ==================== AGENT CONFIGURATION ====================
# Enable/disable specific agents
ENABLE_TEACHER_AGENT = True
ENABLE_QUESTION_GENERATOR = True
ENABLE_HINT_AGENT = True
ENABLE_EVALUATOR_AGENT = True

# Hint agent configuration
MAX_HINT_LEVEL = 3
ENABLE_SOCRATIC_HINTS = True

# Evaluator configuration
MIN_SCORE_PASS = 70  # Minimum score to pass (0-100)
ENABLE_CODE_EXECUTION = False  # WARNING: Executing untrusted code is dangerous

# ==================== UI CONFIGURATION ====================
UI_THEME = "soft"  # Gradio theme
DEFAULT_TOPICS = ["Arrays", "Linked Lists", "Trees"]
DEFAULT_DIFFICULTY = "Medium"

# ==================== LOGGING CONFIG ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "dsa_tutor.log"
ENABLE_CONSOLE_LOG = True

# ==================== VALIDATION ====================
def validate_config():
    """Validate configuration and warn about issues"""
    warnings = []
    
    # Check API key
    if GROQ_API_KEY == "gsk_your_api_key_here" or not GROQ_API_KEY:
        warnings.append("Groq API key not set. Get a free key from: https://console.groq.com/keys")
    
    # Check embeddings
    # Note: Using HuggingFace embeddings, no API key required
    pass
    
    # Check model selection
    if GROQ_MODEL not in ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]:
        warnings.append(f"Unknown Groq model: {GROQ_MODEL}")
    
    return warnings

# ==================== UTILITY FUNCTIONS ====================
def get_model_config():
    """Get model configuration based on available API keys"""
    if GROQ_API_KEY and GROQ_API_KEY != "gsk_your_api_key_here":
        return {
            "provider": "groq",
            "model": GROQ_MODEL,
            "api_key": GROQ_API_KEY,
            "temperature": MODEL_TEMPERATURE
        }
    elif OPENAI_API_KEY:
        return {
            "provider": "openai",
            "model": OPENAI_MODEL,
            "api_key": OPENAI_API_KEY,
            "temperature": MODEL_TEMPERATURE
        }
    else:
        raise ValueError("No valid API key found. Please set GROQ_API_KEY or OPENAI_API_KEY")

def print_config_summary():
    """Print a summary of the current configuration"""
    print("=" * 50)
    print(f"{APP_NAME} v{APP_VERSION} - Configuration Summary")
    print("=" * 50)
    
    config = get_model_config()
    print(f"Model Provider: {config['provider'].upper()}")
    print(f"Model: {config['model']}")
    print(f"Temperature: {MODEL_TEMPERATURE}")
    print(f"Storage: {STORAGE_TYPE}")
    print(f"UI Theme: {UI_THEME}")
    print(f"Debug Mode: {DEBUG_MODE}")
    
    warnings = validate_config()
    if warnings:
        print("\n WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("=" * 50)