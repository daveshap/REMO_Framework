import requests
import time

url = 'http://localhost:8000/add_message'

user = input('User: ')
message = input('Enter your message: ')
timestamp = time.time()

response = requests.post(url, params={
    'message': message,
    'speaker': user,
    'timestamp': timestamp
})

print(response.status_code)
print(response.json())
