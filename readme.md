# Project Scope
This project is for predicting the house prices using housing_price_dataset.csv

https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data

We made the data preparation then we used diffrent models and then chose the best model.
we used the r2_score and mean square error to compare the models.

Using pickle we were able to export the chosen model to be used.

## Probem Description

We have some info about the house like the area, neighbourhood, number of bedrooms and number of bathrooms
using these data we try to predict the house price.

we tried to get the relation between these information and the price.

Then we did some processing to get the best performance.

## Deploying the model locally 

We created a Dockerfile to create an image. \
firstly we can build the model using: \
docker build . -t house_predictor
then we can run the image using: \ 
docker run   -it  -p 9696:9696 house_predictor 

then we can run the predict_local.py after installing requests in our environment
as the url will be:
url = 'http://localhost:9696/predict_house_price' 
and run \
python predict_local.py

# Using the deployed model 
to use the deployed model the url will be: \
url = 'http://peng24.pythonanywhere.com/predict_house_price'  \
then run \
python predict.py


# predicted dictionary  shape
Make sure the predicted dictionary to be in this form: \
bathrooms from 1 to 3 \
bedrooms from to to 5 \

client = {"squarefeet":2126,
"yearbuilt":1969,
"neighborhood":"Rural",
"bathrooms":3,
"bedrooms":4,
}









