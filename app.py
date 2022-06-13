from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

covid_model = pickle.load(open('covid_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("prediction_page.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    int_features[0] = (int_features[0]-1)//4
    int_features[1] = ((int_features[1]-12)/17.975) - 1.8
    int_features[2] = ((int_features[2]-40000)/76666.66) - 2.55
    int_features[3] = ((int_features[3] - 0.1) / 0.7425) - 0.5338
    int_features[4] = ((int_features[4] - 2) / 1.76) - 1.66103
    int_features[5] = -0.329
    int_features[6] = ((int_features[6] - 1000) / 2142.8) - 1.4
    int_features[7] = ((int_features[7] - 100) / 163.63) - 2.05
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=covid_model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if(output < str(0.25)) :
        return render_template('prediction_page.html', pred='Home quarantine the patient')
    elif(output < str(0.50)) :
        return render_template('prediction_page.html', pred='Shift patient to general ward')
    else :
        return render_template('prediction_page.html', pred='Shift patient to ICU')

if __name__ == '__main__':
    app.run(debug=True)
