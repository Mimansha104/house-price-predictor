from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

house=pd.read_csv("final_df.csv")

location_encoder= pickle.load(open('models/location_encoder (1).pkl','rb'))
area_encoder= pickle.load(open('models/area_encoder (1).pkl','rb'))
ohe= pickle.load(open('models/ohe (1).pkl','rb'))
model= pickle.load(open('models/Rfrmodel.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def index():
    location= sorted(house['location'].unique())
    return render_template("index.html",location=location)

@app.route('/predict',methods=['post'])
def predict():
    location = request.form.get('location')
    total_sqft = request.form.get('total_sqft')
    area = request.form.get('area_type')
    size = request.form.get('size')
    bath = request.form.get('bath')
    balcony = request.form.get('balcony')
    print(location,total_sqft,area,size,bath, balcony)


    area_type = area_encoder.transform(np.array([area]))
    print( area_type)

    location=location_encoder.transform(np.array([location]))
    print(location)


    X=np.array([size,total_sqft,bath,balcony]).reshape(1,4)
    print(X)
    X_trans=np.array([area_type,location]).reshape(1,2)
    print(X_trans)

    X_trans= ohe.transform(X_trans).toarray()
    print(X_trans.shape)
    X= np.hstack((X_trans,X))
    print(X)
    print(X.shape)

    y_pred= model.predict(X)

    print(y_pred)
    print(X.shape)
    return str(y_pred)



if __name__=="__main__":
    app.run(debug=True)