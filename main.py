from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods = ["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form

        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

    # Code for inference
        inputfeatures = [fever, pain, age, runnyNose, diffBreath]
        infprob = clf.predict_proba([inputfeatures])[0][1]
        # return 'Hello, World! ' + str(infprob)
        # print(infprob)
        return render_template('show.html', inf = infprob)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)