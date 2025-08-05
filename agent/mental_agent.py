# agent/mental_agent.py
import joblib
import random
import os
import sys

# Add the project root to the Python path for imports
# This is crucial for local runs and some deployment environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the NLU service after path is set
from ibm_services.ibm_nlu import get_emotion_analysis

# --- Agentic AI Core: Load Models and Tools ---
# Ensure models are loaded relative to the project root
try:
    # Adjust paths for deployment environment if necessary, but relative paths usually work
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    print("✅ Models and tools loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure you have run preprocess.py and train_model.py first.")
    # In a deployed environment, this would cause a crash, which is what we want.
    # It indicates a critical setup failure.
    raise RuntimeError(f"Model files not found: {e}. Ensure preprocess.py and train_model.py were run.")


def generate_response_by_emotion(emotion_scores):
    """
    Creates an agentic response based on emotion scores, prioritizing negative emotions.
    """
    # First, check for strong negative emotions
    if emotion_scores.get("sadness", 0) > 0.4:
        return "I hear sadness in your words. It's okay to feel this way, and I'm here to listen. You can talk to me about anything."
    
    elif emotion_scores.get("fear", 0) > 0.4:
        return "I'm sensing some fear or worry. Remember to take a deep breath. It's brave to share what you're feeling."
    
    elif emotion_scores.get("anger", 0) > 0.4:
        return "I'm sensing a lot of anger, and I'm sorry you're feeling that way. It might help to take a moment to cool down."
    
    # Then, check for strong positive emotions
    elif emotion_scores.get("joy", 0) > 0.5:
        return "That sounds wonderful! I'm glad to hear you are having a good day."
    
    # Default fallback if no single emotion is dominant or strong
    else:
        fallback_responses = [
            "I'm here to listen. You can always talk to me.",
            "I hear you. Thank you for sharing.",
            "It's okay to not have all the answers. Let's just talk."
        ]
        return random.choice(fallback_responses)

def process_user_input(user_input):
    """
    Main function to process user input and get a response from the agent.
    """
    nlu_result = get_emotion_analysis(user_input)
    if not nlu_result['success']:
        return "I'm sorry, I'm having trouble connecting to my service. Please try again later."
    
    emotions = nlu_result['emotions']
    
    # Generate response using the new, more robust logic
    response = generate_response_by_emotion(emotions)
    
    return response

if __name__ == "__main__":
    print("Mental Health Agent is running in test mode. Type 'quit' to exit.")
    while True:
        user_text = input("You: ")
        if user_text.lower() == 'quit':
            break
        
        response = process_user_input(user_text)
        print(f"Agent: {response}")