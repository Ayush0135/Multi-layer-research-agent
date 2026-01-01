
import os
import time
import google.generativeai as genai
from groq import Groq
from anthropic import Anthropic, NotFoundError
from dotenv import load_dotenv
from utils.llm_offline import query_offline_llm

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize Clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# --- Internal Callers ---

def _call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found.")
    
    # Updated to gemini-2.0-flash based on available models
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Simple retry logic for ResourceExhausted or other transient errors
    # Reduced retries for faster failover to other models/offline
    max_retries = 3 
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if not response.text:
                raise ValueError("Gemini returned empty response.")
            return response.text
        except Exception as e:
            # Check if it's a quota error (429/ResourceExhausted)
            if "429" in str(e) or "ResourceExhausted" in str(e) or "QuotaExceeded" in str(e):
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                    print(f"  [Gemini] Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            raise e  # smooth failover to next model if retries exhausted or other error

def _call_groq(prompt):
    if not groq_client:
        raise ValueError("GROQ_API_KEY not found or client init failed.")
    
    # Updated to llama-3.3-70b-versatile
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        if "429" in str(e): # Rate limit
             # For Groq free tier, limits can be long (minutes). 
             # Better to failover immediately unless it's very short, 
             # but we can try a short sleep just in case.
             time.sleep(2) 
        raise e

def _call_anthropic(prompt):
    if not anthropic_client:
        raise ValueError("ANTHROPIC_API_KEY not found or client init failed.")
    
    # Updated: Try Claude 3.5 Sonnet (June version) then Haiku (most widely available)
    model_id = "claude-3-5-sonnet-20240620" 
    
    try:
        message = anthropic_client.messages.create(
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            model=model_id, 
        )
        return message.content[0].text
    except NotFoundError:
        # Fallback to Haiku which is usually available to all tiers
        try:
            message = anthropic_client.messages.create(
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                model="claude-3-haiku-20240307", 
            )
            return message.content[0].text
        except Exception as e:
            raise e

# --- Main Logic ---

# --- Stage Configuration ---

# Format: "stage_name": ["model_id_1", "model_id_2"]
# Model IDs can be: 'groq', 'anthropic', 'gemini', or 'ollama:model_name'
STAGE_CONFIG = {
    "default": ["groq", "anthropic", "ollama:llama3.2"],
    
    # Fast, Logic Heavy
    "topic": ["groq", "anthropic", "ollama:llama3.2"],
    
    # Search filtering (High volume, needs speed)
    "discovery": ["groq", "ollama:llama3.2"], 
    
    # Analysis (Heavy Context, Reasoning)
    "analysis": [
        "groq",                      
        "anthropic",                 
        "ollama:llama3.2"      
    ],
    
    # Scoring (FAST, strict formatting)
    "scoring": ["groq", "ollama:llama3.2"],
    
    # Synthesis & Generation (Creative, high quality)
    "synthesis": ["anthropic", "groq", "ollama:llama3.2"],
    "generation": ["anthropic", "groq", "ollama:llama3.2"]
}

def _resolve_strategy(model_id):
    """
    Returns a callable (function) for a given model_id string.
    """
    if model_id == 'groq':
        return lambda p: _call_groq(p)
    elif model_id == 'anthropic':
        return lambda p: _call_anthropic(p)
    elif model_id == 'gemini':
         return lambda p: _call_gemini(p)
    elif model_id.startswith('ollama:'):
        # Specific offline/cloud model
        model_name = model_id.split(':', 1)[1]
        return lambda p: query_offline_llm(p, model_name=model_name)
    else:
        # Default to offline if unknown
        return lambda p: query_offline_llm(p)

def execute_strategies(strategies, prompt):
    """
    Executes a list of strategy functions in order.
    """
    errors = []
    for i, func in enumerate(strategies):
        try:
             # print(f"  [Strategy {i+1}] Executing...") 
             return func(prompt)
        except Exception as e:
            errors.append(str(e))
            from termcolor import colored
            
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                print(colored(f"  [Limit] Provider rate limited. Switching to backup...", "yellow"))
            elif "not found" in error_msg.lower():
                 print(colored(f"  [Config] Key not found/Model missing. Switching...", "yellow"))
            else:
                print(colored(f"  [Error] Strategy {i+1} failed: {error_msg[:80]}...", "red"))
                
            # print(colored(f"  [Fallback] Transferring context...", "yellow"))
            continue
            
    # Fallback to generic offline if enabled and not already tried
    enable_offline = os.getenv("ENABLE_OFFLINE_FALLBACK", "True").lower() == "true"
    if enable_offline:
        try:
            return query_offline_llm(prompt)
        except Exception as e:
            errors.append(f"Offline Default: {e}")
            
    raise Exception(f"All strategies failed. Errors: {errors}")

def query_stage(stage, prompt):
    """
    Primary Entry Point for Stage-based LLM routing.
    """
    # Get config for stage, or default
    model_chain = STAGE_CONFIG.get(stage, STAGE_CONFIG['default'])
    
    # Resolve to functions
    strategies = [_resolve_strategy(m) for m in model_chain]
    
    return execute_strategies(strategies, prompt)

# --- Deprecated / Compatibility ---

def query_llm_robust(prompt, primary_preference=None, use_heavy_fallback=True):
    """
    Legacy wrapper. Maps to 'default' stage effectively.
    """
    return query_stage("default", prompt)

def query_gemini(prompt, retries=1, delay=0, fallback_to_others=False):
    # Map legacy calls to appropriate stages based on fallback flag
    # If fallback_to_others is False (usually analysis), use 'analysis' stage
    if not fallback_to_others:
        return query_stage("analysis", prompt)
    return query_stage("default", prompt)

def query_groq(prompt, json_mode=False, fallback_to_others=True):
    return query_stage("scoring", prompt)
