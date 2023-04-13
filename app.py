from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import random


filename = 'spam.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

#loaded_model = load_model('spam.h5')
cv = pickle.load(open('cv1.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/Spam',methods=['POST','GET'])
def prediction():
	return render_template('spam.html')

@app.route('/test')
def test():
    print("test")
    test_msg = ["test message"]
    test_msg = test_msg.toarray().reshape(-1,7162)
    test_pred =loaded_model.predict(test_msg)
    print(test_pred)
    return render_template('result.html',prediction=1)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = message.strip()
        print(data)
     
    new_review = str(data)
    print(new_review)
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()

    print(new_X_test)
    new_y_pred =loaded_model.predict(new_X_test)
    
    new_x_pred = np.where(new_y_pred>0.5,1,0)
    print(new_x_pred)
    
    

    if new_review[0][0]==1:
        return render_template('result.html',prediction=1)
    else :
        return render_template('result.html',prediction=0)
     

if __name__ == '__main__':
    #port=int(os.environ.get('PORT',5000))
    app.run(debug=False)