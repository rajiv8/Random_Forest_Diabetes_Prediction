from flask import render_template,Flask,request
import pickle

app=Flask(__name__)
file=open("model.pkl","rb")
random_Forest=pickle.load(file)
file.close()

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        myDict = request.form
        age = float(myDict["age"])
        bmi = float(myDict["bmi"])
        pregnancies=float(myDict["Pregnancies"])
        insulin = float(myDict["insulin"])
        glucose = float(myDict["glucose"])
        BP = float(myDict["BP"])
        skin_thickness = float(myDict["thickness"])
        pedigreeFunction=float(myDict["PedigreeFunction"])
        
        pred = [pregnancies,glucose, BP, skin_thickness,insulin,bmi,pedigreeFunction,age]
        Diabetes_Prediction=random_Forest.predict([pred])[0]
        if(Diabetes_Prediction==1):
            res="Diabetic. Consult with Doctor Immediately."
        else:
            res="Non-Diabetic. All good, do exercise regularly."
        return render_template('result.html',age=age,bmi=bmi,Pregnancies=pregnancies,insulin=insulin,
        glucose=glucose,BP=BP,thickness=skin_thickness,PedigreeFunction=pedigreeFunction,res=res)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
