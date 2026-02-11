import numpy as np
import requests
from pathlib import Path

class TinyStoriesLoader:
    
    def __init__(self):
        self.url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        
    def load_text(self, max_lines=75000):
        response = requests.get(self.url)
        text = response.text.split('\n')[:max_lines]
        return ' '.join(text)