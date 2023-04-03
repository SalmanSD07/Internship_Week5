import numpy as np 
from flask import Flask, request,render_template 
import pickle

app=Flask(__name__)
model=pickle.load(open('model_1.pkl','rb'))

@app.route('/')
def home():
    
    return "Hello World"


@app.route('/predict',methods=['POST'])
def predict():
    a=request.get_json()
    
    
    b=[a["area"],a["bedrooms"],a["bathrooms"],a["stories"]]
    features=[float(x) for x in b]
    final_features=[np.array(features)]
    
    predictions=model.predict(final_features)
    out=predictions[0]
    print(out)

    return str(out)

if __name__=="__main__":
    app.run(port=5000,debug=True)