""" end  to end training  scrpit
usage - > in terminal after activating your vene , python src/models.py

output  - > saved pipeline as pk1,
cleaned data ,csv
the plot, .png
"""


import os
import sys

from sklearn.model_selection import train_test_split
import pandas  as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib

from preprocessing import load_and_clean, get_feature_columns, TARGET, LOG_TARGET
from evaluate import evaluate_model, plot_predictions, print_observations

# add src /to path
sys.path.insert,(os.path.dirname(__file__))

# configuration
RAW_DATA_PATH = 'data/raw/Developer_survey_2025.csv'
PROCESSED_DATA = 'data/cleaned/processed-data.csv'
MODEL_OUTPUT_PATH = 'model/salary_pipeline_v2.pkl' 

RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    'n_estimators':300,
    'max_depth':5,
    'learning_rate':0.05,
    'subsample':0.8,
    'colsample_bytre':0.8,
    'random_state': RANDOM_STATE,
    'tree_method':'hist',
    'verbosity':0,
    'reg_alpha':0.05 #L1 regularization,feature selection
}

def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
        1. num pipeline
        2, cat pipeline
        3. column transformer
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(smooth = 'auto',target_type = 'continuous', random_state = RANDOM_STATE))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])

    return preprocessor

def build_pipeline(cat_cols: list, num_cols: list) -> Pipeline :
    """
     Combine preprocessor + xgboost model into one sklearn pipeline
    """
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = XGBRegressor(**XGBOOST_PARAMS)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

def main():
    # 1. Load and clean data
    df = load_and_clean(RAW_DATA_PATH)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f'csv saved to {PROCESSED_DATA} \n')

    #2. separating features from the target
    X = df.drop(columns=[LOG_TARGET])
    y = df[LOG_TARGET]

    cat_cols, num_cols = get_feature_columns(df)
    print(f'we have cat cols {cat_cols} and num cols {num_cols} \n')

    #3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f'We have split the data {TEST_SIZE * 100}% to testing the rest to training')

    #4. Building and training the pipeline
    print('Building pipeline ...')
    pipeline = build_pipeline(cat_cols, num_cols)

    print('Training XGBoost model...\n')
    pipeline.fit(X_train, y_train)
    print('Training complete. \n')

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    print("training _metrics")

    train_metrics = evaluate_model(y_train, y_pred_train)
    print_observations(train_metrics)
    print("\ntesting metrics\n")


    test_metrics = evaluate_model(y_test,y_pred, title=" test set performance")
    print_observations(test_metrics)

    plot_predictions(y_test.values, y_pred, 'data/predictions_plot.png')

    #6. Save the pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}\n")

    """# 7. Predicting one example
    print('Sample prediction: ')
    sample = pd.DataFrame([
        {
            'Country': 'Kenya',
            'YearsCode': 5.0,
            'EdLevel': "Bachelor's",
            'Employment': "Full-time",
            'LanguageCount': 3
        }
    ])
    pred = pipeline.predict(sample)[0]
    print(f"Input: {sample.to_dict(orient='records')[0]}")
    print(f"predicted salary: ${pred:,.0f}")"""

    print('\n Training script complete. \n')


if __name__ == '__main__':
    main()

    


