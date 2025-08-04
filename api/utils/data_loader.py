import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.json')

def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Optionally, cache data in memory for performance
_data_cache = None

def get_data():
    global _data_cache
    if _data_cache is None:
        _data_cache = load_data()
    return _data_cache