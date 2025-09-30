#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides an interactive chat interface using Gemini or ChatGPT
with context from scilpy documentation.
"""
import os
import argparse
import importlib.util

if importlib.util.find_spec("scilpy") is None:
    print("Error: scilpy is not installed.")
    exit(1)

from scilpy import SCILPY_HOME

# Try to import OpenAI
try:
    from openai import OpenAI
    openai_available = True
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
except ImportError:
    print("Warning: openai library not installed. Fix : pip install openai")
    openai_available = False
    client = None

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    gemini_available = True
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        gemini_available = False
except ImportError:
    print("Warning: google-generativeai library not installed. Fix : pip install google.generativeai")
    gemini_available = False


def build_arg_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Interactive chat with AI models using scilpy documentation context."
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument('--gemini', action='store_true', help='Use Gemini Bot')
    group.add_argument('--chatgpt', action='store_true', help='Use ChatGPT Bot')
    return p


def load_context(doc_path):
    """Read all files in doc_path and concatenate as a single string."""
    context = ""
    for f in os.listdir(doc_path):
        file_path = os.path.join(doc_path, f)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as fd:
                    context += fd.read() + "\n"
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    return context


def ask_gpt(prompt):
    """Send prompt to OpenAI GPT and return the response."""
    if not openai_available:
        return "Error: OpenAI library not installed."
    
    if client is None:
        return "Error: OPENAI_API_KEY not set in environment."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for scilpy, a Python library for diffusion MRI processing."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def ask_gemini(prompt):
    """Send prompt to Google Gemini and return the response."""
    if not gemini_available:
        return "Error: google-generativeai library not installed or GOOGLE_API_KEY not set."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


def chat_loop(model_name, context):
    """Interactive chat loop."""
    print(f"Starting {model_name} chat. Type 'exit' or 'quit' to end the session.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        # Build prompt with context
        prompt = f"""Context (scilpy documentation):
{context}

User question: {user_input}

Please answer based on the context provided above."""
        
        # Get response from selected model
        if model_name == "chatgpt":
            reply = ask_gpt(prompt)
        elif model_name == "gemini":
            reply = ask_gemini(prompt)
        else:
            reply = "Error: Unknown model"
        
        print(f"\n{model_name.upper()}: {reply}")


def main():
    """Main function."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Determine which model to use
    if args.chatgpt:
        model = "chatgpt"
        if not openai_available or client is None:
            print("Error: ChatGPT is not available. Install openai library and set OPENAI_API_KEY.")
            exit(1)
    elif args.gemini:
        model = "gemini"
        if not gemini_available:
            print("Error: Gemini is not available. Install google-generativeai library and set GOOGLE_API_KEY.")
            exit(1)
    else:
        # Default to gemini
        model = "gemini"
        if not gemini_available:
            print("Warning: Gemini not available, trying ChatGPT...")
            if openai_available and client:
                model = "chatgpt"
            else:
                print("Error: No AI models available. Install required libraries and set API keys.")
                exit(1)
    
    # Load documentation context
    doc_path = os.path.join(SCILPY_HOME, '.hidden')
    
    if not os.path.exists(doc_path):
        print("Error: scilpy doc not generated.")
        print("Enter this command in the terminal: scil_search_keywords --r hello")
        exit(1)
    
    if not any(os.path.isfile(os.path.join(doc_path, f)) for f in os.listdir(doc_path)):
        print("Error: scilpy doc directory is empty.")
        print("Enter this command in the terminal: scil_search_keywords --r hello")
        exit(1)
    
    # Load context and start chat
    print("Loading documentation context...")
    context = load_context(doc_path)
    print(f"Loaded {len(context)} characters of context.")
    
    chat_loop(model, context)


if __name__ == "__main__":
    main()