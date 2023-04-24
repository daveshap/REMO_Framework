import requests
import time

url = 'http://localhost:8000/search'

query = input('search: ')

response = requests.get(url, params={'query': query})

print(response.status_code)
print(response.json())
