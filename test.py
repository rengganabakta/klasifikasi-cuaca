import requests

# Check health first
health_url = "https://klasifikasi-cuaca.onrender.com/health"
health_response = requests.get(health_url)
print("Health check response:")
print(health_response.text)
print("\n")

# Then try to send sensor data
url = "https://klasifikasi-cuaca.onrender.com/sensor-data"
data = {
    "temperature": 25.5,
    "humidity": 60.0,
    "pressure": 1013.0
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=data, headers=headers)
print("Sensor data response:")
print(response.text)