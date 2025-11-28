import requests

URL = "http://127.0.0.1:5000/api/add_student"

students = [
    {"name": "Vikas", "age": 25, "email": "vikas@example.com", "final_score": 85},
    {"name": "Rajat", "age": 24, "email": "rajat@example.com", "final_score": 45},
    {"name": "Abhishek", "age": 23, "email": "abhishek@example.com", "final_score": 70},
    {"name": "Garima", "age": 22, "email": "garima@example.com", "final_score": 90},
    {"name": "Sonu", "age": 26, "email": "sonu@example.com", "final_score": 55}
]

for s in students:
    res = requests.post(URL, json=s)
    print("Added:", s["name"], "| Status:", res.status_code, "| Response:", res.json())
