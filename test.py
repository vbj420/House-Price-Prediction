import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import Entry, Label, Button, StringVar


def preprocess_data(data):
    # Preprocess boolean data
    data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
    data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
    data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
    data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
    data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
    data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})

    data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

    return data

def train_model(X, y):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2)),
        ('regression', LinearRegression())
    ])

    model.fit(X, y)

    return model

def predict_price(model, user_input):
    user_input_df = pd.DataFrame([user_input])
    user_input_poly = model.named_steps['poly'].transform(model.named_steps['preprocessor'].transform(user_input_df))
    predicted_price = model.named_steps['regression'].predict(user_input_poly)
    
    return predicted_price[0]

class HousePricePredictorApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("House Price Predictor")

        self.model = model

        self.input_vars = {}
        self.create_input_widgets()
        self.create_predict_button()
        self.create_output_label()

    def create_input_widgets(self):
        row = 0
        for attribute in X.columns:
            label = Label(self.root, text=f"{attribute}:")
            label.grid(row=row, column=0, padx=10, pady=5)

            entry_var = StringVar()
            entry = Entry(self.root, textvariable=entry_var)
            entry.grid(row=row, column=1, padx=10, pady=5)

            self.input_vars[attribute] = entry_var
            row += 1

    def create_predict_button(self):
        button = Button(self.root, text="Calculate", command=self.predict_price)
        button.grid(row=len(X.columns) + 1, column=0, columnspan=2, pady=10)

    def create_output_label(self):
        self.output_var = StringVar()
        label = Label(self.root, textvariable=self.output_var, font=("Helvetica", 16))
        label.grid(row=len(X.columns) + 2, column=0, columnspan=2, pady=10)

    def predict_price(self):
        user_input = {}
        for attribute in X.columns:
            if attribute in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
                # Convert 'yes' to 1 and 'no' to 0
                user_input[attribute] = 1 if self.input_vars[attribute].get().lower() == 'yes' else 0
            elif attribute == 'furnishingstatus':
                # Map 'unfurnished' to 0, 'semi-furnished' to 1, and 'furnished' to 2
                user_input[attribute] = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}[self.input_vars[attribute].get().lower()]
            else:
                user_input[attribute] = float(self.input_vars[attribute].get())
    
        predicted_price = predict_price(self.model, user_input)
        self.output_var.set(f'Predicted Price: {predicted_price}')

if __name__ == "__main__":
    # Assuming "Housing.csv" contains your dataset
    data = pd.read_csv("Housing.csv")

    # Preprocess data
    data = preprocess_data(data)

    X = data.drop('price', axis=1)
    y = data['price']

    # Train the model
    trained_model = train_model(X, y)

    # Tkinter GUI
    root = tk.Tk()
    app = HousePricePredictorApp(root, trained_model)
    root.mainloop()
