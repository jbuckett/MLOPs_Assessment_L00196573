import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# load my trained model from the file
model = pickle.load(open("model.pkl", "rb"))
@app.route("/")
def home():
    # show the main web page where users type their info
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    # get the data the user typed into the web form
    schooling = float(request.form["schooling"])
    status = request.form["status"]

    # change the status text into numbers so the model understands it
    if status == "Developed":
        status_num = 1
    else:
        status_num = 0
# put the features into a dataframe
    features = pd.DataFrame([[status_num, schooling]], columns=["Status", "Schooling"])
# make the prediction using my trained model
    prediction = model.predict(features)

    # round the answer to 1 decimal place
    result = round(float(prediction[0]), 1)
    final_text = f"The predicted life expectancy is {result} years."
# show the result page with the final answer
    return render_template("result.html", prediction=final_text)
if __name__ == "__main__":
    # run the app so anyone on the network can access it
    app.run(debug=True, host='0.0.0.0')

