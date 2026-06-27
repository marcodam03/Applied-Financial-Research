from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from settings import SEED

# Collection of models used in the project (all with fixed random seed)
MODELS = {
    # Logistic Regression with L2 regularization
    "Logit_L2": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.5, random_state=SEED)
        ),

    # Logistic Regression without regularization
    "Logit_noReg": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1e9, random_state=SEED)
        ),

    # Random Forest Classifier
    "RF": RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=SEED
        ),

    # Gradient Boosting Classifier
    "GB": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=SEED
        ),

    # Ridge Classifier
    "Ridge": make_pipeline(
        StandardScaler(), RidgeClassifier(alpha=1.0, random_state=SEED)
        ),
}

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    MODELS["XGB"] = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=SEED
        )
except Exception:
    pass
