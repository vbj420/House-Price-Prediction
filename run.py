import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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


# Assuming "Housing.csv" contains your dataset
data = pd.read_csv("Housing.csv")

# Preprocess data
data = preprocess_data(data)

X = data.drop('price', axis=1)
y = data['price']

# Train the model
trained_model = train_model(X, y)

# User input
user_input = {}
for attribute in X.columns:
    temp = float(input(f"Enter value for {attribute}: "))
    user_input[attribute] = temp

# Predict price using the trained model and user input
predicted_price = predict_price(trained_model, user_input)

print(f'Predicted Price: {predicted_price}')
