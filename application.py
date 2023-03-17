from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import math

app=Flask(__name__)
data=pd.read_csv("PreProcessedData.csv")
cors=CORS(app)
model=pickle.load(open('KNeighborsClassifierModel1.pkl','rb'))

# 'full_name','new_price','year','seller_type','km_driven','owner_type','fuel_type','transmission_type','mileage','Brand'
@app.route('/')
def homePage():
    return render_template('homePage.html')

@app.route('//prediction')
def index():
    car_list={}
    brand = sorted(data['Brand'].unique())
    car_name =  sorted(data['full_name'].unique())
    new_car_price= sorted(data['new-price'].unique())
    year= sorted(data['year'].unique(),reverse=True)
    seller_type= sorted(data['seller_type'].unique())
    # km_driven= sorted(data['km_driven'].unique())
    Owners= sorted(data['owner_type'].unique())
    fuel_type= sorted(data['fuel_type'].unique())
    transmissionMode= sorted(data['transmission_type'].unique())
    # mileage= sorted(data['mileage'].unique())
    for i in car_name:
        key=i.split(' ')[0]
        car_list[key]=i

    return render_template('index1.html',brand=brand,car_name=car_name,year=year)


@app.route('/prediction',methods=['POST'])
@cross_origin()
def predict():
    Owners=request.form.get('Owner')
    brands=request.form.get('brands')
    car_name=request.form.get('car_names')
    year=request.form.get('years')
    mileages=request.form.get('mileage')
    new_car_price=request.form.get('new_car_prices')
    driven=request.form.get('kilo_driven')
    year=int(year)
    mileages=float(mileages)
    new_car_price=int(new_car_price)
    driven=int(driven)
    print(type(new_car_price))
    print(car_name,new_car_price,year,mileages,brands)
    data1=pd.DataFrame([new_car_price,year,driven,mileages])
    print(data1)  
    data1=data1.to_numpy()
    data1=np.array(data1).reshape(1,4)
    print(data1)
    prediction=model.predict(pd.DataFrame(data1,columns=['new-price','year','km_driven','mileage']))
    print(prediction)
    return str(math.floor(prediction[0]))


if __name__=="__main__":
    app.run(debug=True)