# app.py
import os
from flask import Flask, request, jsonify
from agent.mental_agent import process_user_input
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
# Render will inject these as environment variables directly in production.
load_dotenv()

app = Flask(__name__)

@app.route('/health')
def health_check():
    """
    A simple health check endpoint to ensure the server is running.
    """
    return "OK"

@app.route('/ask', methods=['POST'])
def ask_agent():
    """
    API endpoint to receive user input and return agent's response.
    Expects a JSON payload with a 'text' field.
    """
    data = request.get_json()
    user_input = data.get('text', '')
    
    if not user_input:
        # Return an error if no text is provided
        return jsonify({"response": "Please provide a 'text' field in the request."}), 400
        
    # Process the input using the mental agent logic
    response = process_user_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    # Get the port from environment variable (set by Render), default to 5000 for local testing
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)