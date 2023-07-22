import requests

# Assuming you have the input features in a dictionary 
features = {'Date': value1, 'High': value2}

url = 'http://localhost:5000/predict'
response = requests.post(url, json=features)

if response.status_code == 200:
    result = response.json()
    print('Predicted Stock Price:', result['prediction'])
else:
    print('Error:', response.json()['error'])
