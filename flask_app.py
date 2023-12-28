#!/usr/bin/python3.10

from flask import Flask, jsonify, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


with open('house_price_prediction.pkl', 'rb') as f_in: 
    model = pickle.load(f_in)


app = Flask('ping') 

def transform_year(house):
    house["house_age"] = 2024 - house["yearbuilt"]
    del house['yearbuilt']
    return house 

def transform_bedroom(house):
    for n in range(2,6):
      if n == house["bedrooms"]:
         house[f"bedrooms_{n}"] = 1
      else:
        house[f"bedrooms_{n}"] = 0
    del house["bedrooms"]    
    return house

def transform_bathrooms(house):
    for n in range(1,4):
      if n == house["bathrooms"]:
         house[f"bathrooms_{n}"] = 1
      else:
        house[f"bathrooms_{n}"] = 0
    del house["bathrooms"]    
    return house

def transform_neighbourhood(house):
    for neighborhood in ["rural","suburb","urban"]:
        if neighborhood == house["neighborhood"].lower():
          house[f"neighborhood_{neighborhood}"] = 1
        else:
          house[f"neighborhood_{neighborhood}"] = 0 
    del house["neighborhood"]
    return house       


def transfrom_house(house):
   house= transform_year(house)
   house= transform_neighbourhood(house)
   house= transform_bathrooms(house)
   house= transform_bedroom(house)
   return house


@app.route('/predict_house_price', methods=['POST'])  ## in order to send the employee information we need to post its data.
def predict():
    house = transfrom_house(request.get_json()) 
    # Fit and transform the scaler on the training set
    sample_data = pd.DataFrame([house])
    house_price = model.predict(sample_data)
    result = {
        'house_price': round(house_price[0],3),
    }

    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) 