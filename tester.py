import requests

url = 'http://localhost:8080/predict'

data = {
    'model': 'one'
}

response = requests.post(url, json=data)

print(response.json())