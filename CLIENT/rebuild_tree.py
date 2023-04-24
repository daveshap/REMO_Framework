import requests
import time

url = 'http://localhost:8000/rebuild_tree'

response = requests.post(url)

print(response.status_code)
print(response.json())