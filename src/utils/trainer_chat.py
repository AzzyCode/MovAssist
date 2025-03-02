import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

class TrainerChat:
    def __init__(self, api_key):
        self.api_key = os.getenv("API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "MovAssist" 
        }
        self.context = "You are a personal trainer specializing in squats and push-ups. Provide concise, helpful fitness advice."
        
    
    def get_response(self, question, app_data=None):
        try:
            messages = [{"role": "system", "content": self.context}]
            if app_data:
                messages.append({"role": "system", "content": f"App Data: {app_data}"})
            messages.append({"role": "user", "content": question})
        
            payload = {
                "model": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                "messages": messages,
                "temperature": 0.7,
            }
            
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: Couldn't reach trainer ({str(e)})"
        
        