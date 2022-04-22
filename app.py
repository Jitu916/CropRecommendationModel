from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

crop_labels = {0:'apple',1:'banana',2:'blackgram',3:'chickpea',4:'coconut',5:'coffee',6:'cotton',7:'grapes',8:'jute',9:'kidneybeans', 10:'lentil',11:'maize',12:'mango',13:'mothbeans',14:'mungbean',15:'muskmelon',16:'orange',17:'papaya',18:'pigeonpeas',19:'pomegranate',20:'rice',21:'watermelon'}


@app.route('/')
def hello_world():
    return render_template("farmersCorner.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    #print(int_features)
    #print(final)
    prediction=model.predict(final)
    output=crop_labels[int(prediction[0])]

    return render_template('farmersCorner.html',Result='Suitable Crop for You is {}'.format(output))
    

if __name__ == '__main__':
    app.run(debug=True)
