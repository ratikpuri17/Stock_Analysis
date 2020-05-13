import matplotlib
matplotlib.use('Agg')
import numpy as np
from Stock_Prediction_Final import *
from flask import Flask, request, jsonify, render_template
from flask import Response
import pickle

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# @app.after_request
# def add_header(response):
#     # response.cache_control.no_store = True
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response

@app.route('/')
def home():
    l=[]
    # path='C:\\Users\\Ratik\\Desktop\\Project\\static\\images'
    path=os.getcwd() + '\\static\\images'
    for files in os.walk(path):
    #     for name in files:
    #         print(name)
        l.append(files)

    g=l[0][2]

    for i in g:
        # print(path+"\\"+i)
        os.remove(path+"\\"+i)

    return render_template('base2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''s
    For rendering results on HTML GUI
    '''
    vals = [str(x) for x in request.form.values()]
    # prediction = model.predict(final_features)
    # s=recommending('GOOG','3/12/2019','31/3/2020')
    # stock=pd.read_csv('stock_symbol')
    # present=(stock['symbol']==vals[0]).sum()
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

        # Check in nsepy.txt
        print("Checking in nsepy.txt")
        with open('nsepy.txt','r') as f:

            while(True):

                l=f.readline()

                if(not l):
                    break

                g=l.split('\n')
                if(g[0]==vals[0]):
                    present=1
                    flag=2
                    break

        if(present==0):
            return render_template('error1.html')
    print("-------------------------------")
    print(vals[3])
    print('--------------------------------')
    
    if(len(vals[3])==0):
        print("DEFAULT")
        s,seed_=recommending(vals[0],vals[1],vals[2],flag)
        
    else: 
        s,seed_=recommending(vals[0],vals[1],vals[2],flag,vals[3])

   #  return render_template('untitled1.html', name = 'new_plot',
   # url ='/static/images/new_plot.png')
    # output = round(prediction[0], 2)
    random.seed(seed_)
    r=random.randint(1,1e9)
    return render_template('index.html', prediction_text=s,img=vals[3],r=r)

@app.route('/fetch',methods=['GET'])
def fetch():
    return render_template('stock_data.html')

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