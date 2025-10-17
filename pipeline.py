import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

def handle_missing_values(df):
    cat_columns = df.select_dtypes(include='object').columns
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in cat_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in num_columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df


def encode_categorical(df):
    cat_columns = df.select_dtypes(include='object').columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[cat_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_columns))
    
    df = pd.concat([df.drop(columns=cat_columns), encoded_df], axis=1)
    return df


def normalisation_data(X_train, X_test, column_to_scale):
    scaler = StandardScaler()
    
    # Fit only on the selected columns of X_train
    X_train[column_to_scale] = scaler.fit_transform(X_train[column_to_scale])
    X_test[column_to_scale] = scaler.transform(X_test[column_to_scale])
    
    return X_train, X_test

def feature_selection(X, y, num_features=5):
    selector = SelectKBest(score_func=f_regression, k=num_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return {"MAE": mae, "MSE": mse, "R2": r2}

def main():
    data = "data.csv"
   
    df = pd.read_csv(data)
    df = handle_missing_values(df)
    df = encode_categorical(df)
    target_col = 'Delivery_Time_min'
    col_to_drop = ['Order_ID', 'Courier_Experience_yrs']
    X = df.drop(columns=[target_col]+col_to_drop)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )   
    column_to_scale = ['Distance_km', 'Preparation_Time_min']
    X_train_scaled, X_test_scaled = normalisation_data(X_train.copy(), X_test.copy(), column_to_scale)
    X_train_scaled,X_test_scaled = normalisation_data(X_train, X_test,column_to_scale)
    X_train_selected, selected_features = feature_selection(X_train_scaled, y_train, num_features=5)
    parameters = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None , 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }
    scoring={'MAE':make_scorer(mean_absolute_error ,greater_is_better=False)}
    model=RandomForestRegressor(random_state=0)

    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1, scoring=scoring, refit='MAE')
    grid_search.fit(X_train_selected, y_train)
    best_model = grid_search.best_estimator_ 

    evaluate_model(best_model, X_test_scaled[selected_features], y_test)   




if __name__ == "__main__":
    main()
