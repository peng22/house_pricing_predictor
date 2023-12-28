import requests 
url = 'http://localhost:9696/predict_house_price' 

client = {"squarefeet":2126,
"yearbuilt":1969,
"neighborhood":"Rural",
"bathrooms":2,
"bedrooms":4,
}
result=requests.post(url, json=client)

print(result.json())

   