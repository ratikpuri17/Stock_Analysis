import matplotlib
matplotlib.use('Agg')
import numpy as np
from Stock_Prediction_Final import *
from Trend_Analysis import *
from flask import Flask, request, jsonify, render_template
from flask import Response
import pickle
import glob
import os

app = Flask(__name__)

@app.route('/')
def home():
    
    path=os.getcwd() + '\\static\\images'
    
    for file in glob.glob(path+"\\*.png"):
        f=file.split('\\')
        # print(f[7])
        if(os.path.exists(file) and f[7]!='cloud.png'):
            os.remove(file)


    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    vals = [str(x) for x in request.form.values()]

    present=0
    flag=0
    with open('symbolfile.txt','r') as f:
    
            while(True):

                l=f.readline()

                if(not l):
                    break

                g=l.split('\n')
                if(g[0]==vals[0]):
                    present=1
                    flag=1
                    break
    
    if present==0:
        return render_template('error1.html')

    print("-------------------------------")
    print(vals[3])
    print('--------------------------------')
    
    if(len(vals[3])==0):
        print("DEFAULT")
        s,seed_=recommending(vals[0],vals[1],vals[2],flag)
        
    else: 
        s,seed_=recommending(vals[0],vals[1],vals[2],flag,vals[3])

 
    random.seed(seed_)
    r=random.randint(1,1e9)
    return render_template('index.html', prediction_text=s,img=vals[3],r=r)

@app.route('/fetch',methods=['GET'])
def fetch():
    return render_template('stock_data.html')

@app.route('/stock_symbols',methods=['GET'])
def stock_symbols():
    return render_template('stock_symbolfile.html')

@app.route('/trends',methods=['GET'])
def trends():
    wordcloud=fetch_trend()
    wordcloud.to_file('static\\cloud.png')
    return render_template('trends.html')

@app.route('/tweet',methods=['GET'])
def tweet():
    return render_template('tweets.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html'), 404
@app.errorhandler(403)
def forbidden(error):
    return render_template('error.html'), 403
@app.errorhandler(401)
def gone(error):
    return render_template('error.html'), 401
@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html'), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(debug=False,port=port)