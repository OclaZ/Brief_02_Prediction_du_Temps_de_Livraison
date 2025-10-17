import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler


def charge_df(df_path="data.csv"):
    df=pd.read_csv(df_path)
    df=df.drop_duplicates()
    cat_columns = df.select_dtypes(include='object').columns 
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in cat_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in num_columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df

def split_xy(df):
    X=df.drop(columns=['Delivery_Time_min','Order_ID','Courier_Experience_yrs'])
    y=df['Delivery_Time_min']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def  mae_seuil(grid_params,model):
    df=charge_df()
    X_train, X_test, y_train, y_test=split_xy(df)
    num_cols=['Distance_km','Preparation_Time_min']
    cat_cols=['Weather','Traffic_Level','Time_of_Day','Vehicle_Type']
    preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ('col',OneHotEncoder(handle_unknown='ignore'),cat_cols),
    ]
    )
    pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model)
    ])
    scoring={'MAE':make_scorer(mean_absolute_error ,greater_is_better=False)}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=5,
        scoring=scoring,
        refit='MAE',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae