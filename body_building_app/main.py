from flask import Flask,request
import pickle 
import pandas as pd

import numpy as np

# function to round to the nearest 5 (default)
def myround(x, base=5):
    return base * round(x/base)

# class for powerlifting instance
# this will hold prediction method and powerlifter data
class Powerlifter: 
    def __init__(self, name, gender, equipment, age, bodyweightKg, squatKg, deadliftKg):
        self.name = name
        self.gender = gender
        self.equipment = equipment
        self.age = int(age)
        self.bodyweightKg = int(bodyweightKg) / 2.2
        self.squatKg = int(squatKg) / 2.2
        self.deadliftKg = int(deadliftKg) / 2.2
        self.config_file = {}
    
        # load in model 
        with open('./elastic.bin', 'rb') as f_in:
            self.model = pickle.load(f_in)

        # load in scaler
        with open('./scaler.bin', 'rb') as f_in:
            self.scaler = pickle.load(f_in)

    def create_dict(self):
        female = 1 if self.gender[0].lower() == "f" else 0
        male = 1 if self.gender[0].lower() == "m" else 0
        
        multiply_ply = 1 if self.equipment[0].lower() == "m" else 0
        raw = 1 if self.equipment[0].lower() == "r" else 0
        single_ply = 1 if self.equipment[0].lower() == "s" else 0
        wraps = 1 if self.equipment[0].lower() == "w" else 0

        self.config_file["F"] = [female]
        self.config_file["M"] = [male]
        self.config_file["Multi-ply"] = [multiply_ply]
        self.config_file["Raw"] = [raw]
        self.config_file["Single-ply"] = [single_ply]
        self.config_file["Wraps"] = [wraps]
        self.config_file["Age"] = [self.age]
        self.config_file["BodyweightKg"] = [self.bodyweightKg]
        self.config_file["BestSquatKg"] = [self.squatKg]
        self.config_file["BestDeadliftKg"] = [self.deadliftKg]

    def predict_best_bench(self):
        self.create_dict()
        df = pd.DataFrame.from_dict(self.config_file)
        # print(df.head())
        df = df[['Age', 'BodyweightKg', 'BestSquatKg', \
            'BestDeadliftKg','F','M','Multi-ply','Raw',\
            'Single-ply','Wraps']]

        prepared_df = self.scaler.transform(df)
        y_pred = int(self.model.predict(prepared_df) * 2.2)
        return myround(y_pred)


app = Flask(__name__)

@app.route("/")
def index():

    if request.args.get("name", ""):
        powerlifter = Powerlifter(name=request.args.get("name", ""), gender=request.args.get("gender", ""),\
                    equipment=request.args.get("equipment", ""), age=request.args.get("age", ""),\
                    bodyweightKg=request.args.get("bodyweightlb", ""),\
                    squatKg=request.args.get("squatlb", ""),\
                    deadliftKg=request.args.get("deadliftlb", ""))

        max_bench = str(powerlifter.predict_best_bench())
        print("max bench: "+ max_bench)
    else: 
        max_bench = str(0)

    return (
        """<form action="" method="get">
                NAME: <input type="text" name="name">
                </br>
                GENDER: <input type="text" name="gender">
                </br>
                EQUIPMENT: <input type="text" name="equipment"> (Wraps, Multi-ply, Single-ply, Raw)
                </br>
                AGE: <input type="text" name="age">
                </br>
                BODYWEIGHT: <input type="text" name="bodyweightlb"> (LB)
                </br>
                SQUAT: <input type="text" name="squatlb"> (LB)
                </br>
                DEADLIFT: <input type="text" name="deadliftlb"> (LB)
                </br>
                <input type="submit" value="Predict Max Bench">
            </form>"""
        + "Predicted Max Bench: " 
        + max_bench
    )


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
