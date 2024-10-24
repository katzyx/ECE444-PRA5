import requests
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

# config endpoint
url = "http://serve-sentiment-env.eba-redfm7an.us-east-1.elasticbeanstalk.com/predict"

# test cases were generated by chatgpt ---
test_cases = [
    "The United Nations will convene a summit on climate change next week.", # real
    "NASA successfully launched its new Mars rover to explore the planets surface.", # real
    "Aliens have landed in New York and are taking over Times Square!", # fake
    "Drinking bleach cures COVID-19 according to new research." # fake
]
# --- end chatgpt generated code

# test locally - send requests and print the results
# for text in test_cases:
#     response = requests.post(url, json={"text": text})
#     try:
#         response_data = response.json()
#         print(f"Input: {text}\nPrediction: {response_data}\n")
#     except ValueError:
#         print(f"Failed to parse JSON. Response: {response.text}")

# run tests
with open('performance.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Test Case", "Time (s)"]) # test case with time stamp

    for text in test_cases:
        for _ in range(100):
            start = time.time()
            response = requests.post(url, json={"text": text})
            end = time.time()
            writer.writerow([text, end - start])

# Boxplot from CSV Data
data = pd.read_csv('performance.csv', encoding='ISO-8859-1')

test_case_mapping = {
    "The United Nations will convene a summit on climate change next week.": 1,
    "NASA successfully launched its new Mars rover to explore the planets surface.": 2,
    "Aliens have landed in New York and are taking over Times Square!": 3,
    "Drinking bleach cures COVID-19 according to new research.": 4
}
data['Test Case'] = data['Test Case'].map(test_case_mapping)


data.boxplot(by='Test Case', column='Time (s)', grid=False)
plt.title("API Performance Boxplot")
plt.suptitle("") 
plt.xlabel("Test Case")
plt.show()