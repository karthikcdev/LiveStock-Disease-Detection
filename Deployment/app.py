import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "Animal","Age","Temperature","Symptom 1","Symptom 2","Symptom 3"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "Blackleg"
        if output == 2:
            res_val = "Foot And Mouth"
            if output == 3:
                res_val = "Lumpy Virus"
                if output == 4:
                    res_val = "Pneumonia"
    else:
        res_val = "Anthrax"
        

        

    return render_template('index.html', prediction_text='The disease is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
