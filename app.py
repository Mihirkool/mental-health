# app.py
import os
from flask import Flask, request, jsonify
from agent.mental_agent import process_user_input
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# New route to handle GET requests on the root URL
@app.route('/')
def index():
    return "The Mental Health Agent API is running. Send a POST request to the /ask endpoint to interact with it."

@app.route('/health')
def health_check():
    return "OK"

@app.route('/ask', methods=['POST'])
def ask_agent():
    data = request.get_json()
    user_input = data.get('text', '')
    
    if not user_input:
        return jsonify({"response": "Please provide a 'text' field in the request."}), 400
        
    response = process_user_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)