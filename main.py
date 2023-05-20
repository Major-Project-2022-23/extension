import marshal, types

from flask import Flask,render_template, request

#create an object of the class Flask
app = Flask(__name__)


with open("code_string.bin", "rb") as file:
    code_string = file.read()
code = marshal.loads(code_string)
func = types.FunctionType(code, globals())

#home url
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    text= "Times Of India Shares Fake Cropped Bungee Jump Fail Video"
    links = func(text,10)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True) 
