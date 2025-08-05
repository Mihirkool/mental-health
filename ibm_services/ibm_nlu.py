# ibm_services/ibm_nlu.py
import os
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("IBM_NLU_APIKEY")
SERVICE_URL = os.getenv("IBM_NLU_URL")

authenticator = IAMAuthenticator(API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(SERVICE_URL)

def get_emotion_analysis(text):
    """
    Analyzes the emotional tone of a given text using IBM Natural Language Understanding.
    Returns a dictionary of emotion scores or an error message.
    """
    try:
        response = nlu.analyze(
            text=text,
            features=Features(
                emotion=EmotionOptions()
            )
        ).get_result()
        
        emotions = response['emotion']['document']['emotion']
        return {"success": True, "emotions": emotions}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    test_text_negative = "I'm so worried and anxious about my future. Everything feels wrong."
    test_text_positive = "This is the best day of my life! I'm so happy and proud."

    print("Analyzing negative text:")
    result_negative = get_emotion_analysis(test_text_negative)
    print(json.dumps(result_negative, indent=2))

    print("\nAnalyzing positive text:")
    result_positive = get_emotion_analysis(test_text_positive)
    print(json.dumps(result_positive, indent=2))