import requests 
url = 'http://peng24.pythonanywhere.com/predict_house_price' 

client = {"squarefeet":2126,
"yearbuilt":1969,
"neighborhood":"Rural",
"bathrooms":3,
"bedrooms":4,
}
result=requests.post(url, json=client).json()

print(result)
   