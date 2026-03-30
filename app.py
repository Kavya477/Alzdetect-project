from flask import Flask, render_template, request, redirect, session
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "secret123"

# In-memory user storage
users = {}

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
mri_model = load_model("mri_mobilenet.h5")


# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect("/upload")
        else:
            return render_template("login.html", error="Invalid Login")

    return render_template("login.html")


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users[username] = password
        return redirect("/")

    return render_template("register.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user" not in session:
        return redirect("/")

    if request.method == "POST":

        # Form data
        age = float(request.form["age"])
        educ = float(request.form["educ"])
        mmse = float(request.form["mmse"])
        ses = float(request.form["ses"])
        etiv = float(request.form["etiv"])
        nwbv = float(request.form["nwbv"])
        asf = float(request.form["asf"])

        # Save image
        file = request.files["mri"]
        path = os.path.join("static", file.filename)
        file.save(path)

        # -------- Tabular Model --------
        brain_ratio = nwbv / etiv
        cognitive_index = mmse / age

        features = np.array([[age, educ, ses, mmse, etiv, nwbv, asf,
                              brain_ratio, cognitive_index]])

        features = imputer.transform(features)
        features = scaler.transform(features)

        tabular_prob = model.predict_proba(features)[0][1]

        # -------- MRI Model --------
        img = image.load_img(path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        mri_prob = mri_model.predict(img)[0][0]

        # -------- Combine --------
        final_score = (tabular_prob + mri_prob) / 2 * 100

        result = "High Alzheimer Risk" if final_score > 50 else "Low Alzheimer Risk"

        return render_template("result.html",
                               result=result,
                               score=round(final_score, 2),
                               image=path)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)