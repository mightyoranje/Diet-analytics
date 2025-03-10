import requests
import json
import os

class Generator:
    def __init__(self,nutrition_input:list,ingredients:list=[],params:dict={'n_neighbors':5,'return_distance':False}):
        self.nutrition_input=nutrition_input
        self.ingredients=ingredients
        self.params=params
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:8000')

    def set_request(self,nutrition_input:list,ingredients:list,params:dict):
        self.nutrition_input=nutrition_input
        self.ingredients=ingredients
        self.params=params

    def generate(self):
        request = {
            'nutrition_input': self.nutrition_input,
            'ingredients': self.ingredients,
            'params': self.params
        }
        response = requests.post(url=f'{self.backend_url}/predict/', data=json.dumps(request))
        
        # Add error handling
        try:
            return response.json()
        except json.JSONDecodeError:
            print(f"Failed to decode JSON. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            # Return a fallback or raise a more informative error
            return {"output": [], "error": "Backend error"}