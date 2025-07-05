# Import the generative AI libraryimport google.generativeai as genai
# Call ListModels to see the list of available models and their supported methods
from google import genai
import os
gemini_api_key = os.getenv('GEMINI_API_KEY')
models = genai.Client(api_key=gemini_api_key,
                      http_options={'api_version': 'v1alpha'}).models.list()
for model in models:
    print(f"Model: {model.name}")
    print(f"Supported methods: {model.supported_actions}")
    print()





