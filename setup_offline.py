import os
import ollama
from dotenv import load_dotenv

load_dotenv()

# Configuration for Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def setup_ollama():
    print(f"Checking for Ollama model: {OLLAMA_MODEL}...")
    
    try:
        # Check if model exists
        models = ollama.list()
        model_names = [m['name'] for m in models['models']]
        
        # Ollama names can have :latest suffix or not
        if OLLAMA_MODEL in model_names or f"{OLLAMA_MODEL}:latest" in model_names:
            print(f"Model '{OLLAMA_MODEL}' is already available.")
            return

        print(f"Pulling model '{OLLAMA_MODEL}' from Ollama registry...")
        print("This may take some time depending on your connection.")
        
        # Use pull with progress if possible, but simple pull is fine for a script
        ollama.pull(OLLAMA_MODEL)
        
        print(f"\nSuccessfully pulled {OLLAMA_MODEL}!")
        print("You can now run the research agent with Ollama.")
        
    except Exception as e:
        print(f"\nError using Ollama: {str(e)}")
        print("Ensure Ollama is installed and running (visit https://ollama.com).")

if __name__ == "__main__":
    setup_ollama()
