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

# Suppress any warnings from Gemini's internal system
# Has to be before the import of google.generativeai
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_ABORT_ON_LEAKS"] = "0"

# Try to import OpenAI and validate API key
try:
    from openai import OpenAI
    openai_available = True
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
except ImportError:
    openai_available = False
    client = None

# Try to import Google Generative AI validate API key
try:
    import google.generativeai as genai
    gemini_available = True
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        gemini_available = False
except ImportError:
    gemini_available = False

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def build_arg_parser():
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="__doc__"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument('--gemini', action='store_true',
                       help='Use Gemini model')
    group.add_argument('--chatgpt', action='store_true',
                       help='Use ChatGPT model')
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


def ask_gpt(user_input, context, conversation_history):
    """Send prompt to OpenAI GPT with conversation history."""
    if not openai_available:
        return "Error: OpenAI library not installed."

    if client is None:
        return "Error: OPENAI_API_KEY not set in environment."

    try:
        # Build messages with system context and conversation history
        messages = [
            {"role": "system", "content": f"You are a helpful assistant \
             for scilpy, a Python library for diffusion MRI processing. \
             Here is the documentation context:\n\n{context}"}
        ]

        # Add conversation history
        messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def ask_gemini(user_input, context, chat_session):
    """Send prompt to Google Gemini using chat session."""
    if not gemini_available:
        return "Error: google-generativeai library not installed \
                or GOOGLE_API_KEY not set."

    try:
        response = chat_session.send_message(user_input)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


def chat_loop(model_name, context):
    """Interactive chat loop with conversation memory."""
    print(f"Starting {model_name} chat. Type 'exit' or 'quit' \
           to end the session.")
    print("-" * 60)

    # Initialize conversation memory
    if model_name == "chatgpt":
        # For ChatGPT, we maintain a list of message dictionaries
        conversation_history = []
    elif model_name == "gemini":
        # For Gemini, we create a chat session with initial context
        model = genai.GenerativeModel('gemini-2.5-flash')
        chat_session = model.start_chat(history=[])
        # Send context as first message
        print("Initializing chat with documentation context...")
        pre_prompt = """
            You are a research assistant and an expert in a Python library \
            called scilpy, which is used for diffusion MRI processing. Your \
            primary goal is to help new users discover and understand the \
            tools available in this library.

        Please adhere to the following guidelines:
        - Base all your answers on the provided scilpy documentation context.
        - Be rigorous, concise, and ensure your answers are easy to \
          understand for users who may be new to the field.
        - Do NOT use markdown formatting (e.g., bolding, lists), as the \
          output is displayed in a terminal.

        The user's questions will follow.
        """

        context_introduction = f"{pre_prompt}\n\nHere is the documentation \
            context that you should use to answer questions:\
            \n\n{context}\n\n If you write scripts name, please do not \
            include the .py file extension."

        chat_session.send_message(pre_prompt + context_introduction)
        print("Context loaded into conversation memory.\n")

    while True:
        try:
            user_input = input(f"\nYOU: {BLUE}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Get response from selected model with conversation memory
        if model_name == "chatgpt":
            reply = ask_gpt(user_input, context, conversation_history)
            # Add to conversation history
            conversation_history.append(
                {"role": "user", "content": user_input}
            )
            conversation_history.append(
                {"role": "assistant", "content": reply}
            )
        elif model_name == "gemini":
            reply = ask_gemini(user_input, context, chat_session)
        else:
            reply = "Error: Unknown model"

        print(f"{RESET}\n{model_name.upper()}: {GREEN}{reply}{RESET}")


def main():
    """Main function."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Determine which model to use
    if args.chatgpt:
        model = "chatgpt"
        if not openai_available or client is None:
            print("Error: ChatGPT is not available. pip install openai \
                   and set OPENAI_API_KEY.")
            exit(1)
    elif args.gemini:
        model = "gemini"
        if not gemini_available:
            print("Error: Gemini is not available. pip install \
                   google.generativeai and set GOOGLE_API_KEY.")
            exit(1)
    else:
        # Default to gemini
        model = "gemini"
        if not gemini_available:
            if openai_available and client:
                model = "chatgpt"
            else:
                print("Error: No AI models available. Install required \
                      libraries and set API keys.")
                exit(1)

    # Load documentation context
    doc_path = os.path.join(SCILPY_HOME, '.hidden')

    if not os.path.exists(doc_path):
        print("Error: scilpy doc not generated.")
        print("Enter this command in the terminal: scil_search_keywords \
               --regenerate_help_files placeholder")
        exit(1)

    if not any(
        os.path.isfile(os.path.join(doc_path, f)) for f in os.listdir(doc_path)
    ):
        print("Error: scilpy doc directory is empty.")
        print("Enter this command in the terminal: scil_search_keywords \
               --regenerate_help_files placeholder")
        exit(1)

    # Load context and start chat
    print("Loading documentation context...")
    context = load_context(doc_path)
    print(f"Loaded {len(context)} characters of context.")

    chat_loop(model, context)


if __name__ == "__main__":
    main()
