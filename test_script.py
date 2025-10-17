import pytest
from sklearn.svm import SVR
from script import *

# ---------- TEST charge_df ----------
def test_charge_df_real_data():
    df = charge_df("data.csv")

    # ✅ Basic checks
    assert not df.empty, "DataFrame should not be empty"
    assert "Delivery_Time_min" in df.columns, "Target column missing"
    assert not df.duplicated().any(), "There should be no duplicate rows"
    assert not df.isnull().values.any(), "Missing values should be filled"
    print("\n✅ charge_df test passed — data cleaned successfully")


# ---------- TEST mae_seuil ----------
def test_mae_seuil_real_data_svr():
    # SVR hyperparameter grid
    grid_params = {
        "model__kernel": ["rbf", "linear"],
        "model__C": [0.1, 1, 10],
        "model__epsilon": [0.1, 0.2, 0.5],
    }

    model = SVR()

    mae = mae_seuil(grid_params, model)

    # ✅ Sanity checks
    assert isinstance(mae, float), "MAE should be a float value"
    assert mae <= 10, "MAE should be positive"
    print(f"\n✅ mae_seuil SVR test passed — MAE = {mae:.2f}")
