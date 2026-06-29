"""Data cleaning utilities for developer salary CSV.
We will import this file in train.py or app.py
"""

import pandas as pd
import numpy as np

# Constants
TARGET = 'ConvertedCompYearly'
TOP_N_COUNTRIES = 15
SALARY_MIN = 10000
SALARY_MAX = 500000

SELECTED_FEATURES = [
    'Country',
    'YearsCode',
    'EdLevel',
    'Employment',
    'LanguageHaveWorkedWith',
    # ADD new features
    'DevType',
    'OrgSize',
    'RemoteWork',#remote/hybrid/in-person
    'WorkExp',
    'Industry',
    'Age',# Age Band(use ordinal 0-6)
    'ICorPM',#Individual,contrubutor,person Manager
    'DatabaseHaveWorkedWith',# dabase count (How many dbs known)
    'PlatformHaveWorkedWith',
    'ToolCountWork'

]
TOP_COUNTRIES = 25
LOG_TARGET = 'log_salary'
EMPLOYMENT_KEEP = ['Full-time','Freelance/Self-employed']
TARGET_ENC_FEATURES = ["Country","DevType","Industry"]

#mappings for featurs that we'll do lable encoding

ED_LEVEL_ORDINAL: dict[str, int] = {
    "Primary school": 0,
    "High school": 1,
    "Some college": 2,
    "Associate's": 3,
    "Bachelor's": 4,
    "Master's": 5,
    "Professional":6,
    "Other": np.nan
}
REMOTE_ORDINAL: dict[str,int] = {
    "in-person":0,
    "Hybrid":1,
    "Remote":2,
    "Other":np.nan
}

AGE_ORDINAL: dict[str, int] = {
    "18-24 years old": 0,
    "25-34 years old": 1,
    "35-44 years old": 2,
    "45-54 years old": 3,
    "55-64 years old": 4,
    "65 years or older": 5,
    "Prefer not to say": np.nan
}

ORGSIZE_ORDINAL: dict[str, int] = {
    'Just me - I am a freelancer, sole proprietor, etc.':0,
    'Less than 20 employees': 1,
    '20 to 99 employees': 2,
    '100 to 499 employees': 3,
    '500 to 999 employees': 4,
    '1,000 to 4,999 employees': 5,
    '5,000 to 9,999 employees': 6,
    "10,000 or more employees": 7,
    "I don't know": np.nan
}


# Cleaning functions
def clean_year_code(series: pd.Series) -> pd.Series:
    """
    Convert the YearsCode column to numeric.
    """

    series = series.copy()
    series = pd.to_numeric(series, errors='coerce')

    return series


def clean_work_exp(series:pd.Series)->pd.Series:
    series = series.copy()
    series = pd.to_numeric(series, errors='coerce')

    return series

def clean_education(series: pd.Series)-> pd.Series:
 """ CLEANING EDUCATION CATEGORIESie 
 standardizing the column
 """

 mapping = {
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)":"Master's",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor's",
        "Some college/university study without earning a degree": "Some college",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "High school",
        "Associate degree (A.A., A.S., etc.)": "Associate's",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional",
        "Primary/elementary school":"Primary school",
        "Other (please specify):": "Other"
}
 incomplete = series.map(mapping).fillna("other")
 return incomplete.map(ED_LEVEL_ORDINAL)

def clean_employment(series: pd.Series) -> pd.Series:
    """ simplifying employment into core categories"""

    def simplify(val):
        if pd.isna(val):
            return np.nan
       
        val = str(val)
        if 'full-time' in val.lower() or 'employed' in val.lower():
            return 'Full-time'
        elif 'Independent contractor, freelancer, or self-employed' in val.lower():
            return 'Freelance/Self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
        
    return series.apply(simplify)

def clean_age(series: pd.Series) -> pd.Series:
    """ Mapping age bands to ordinal intergers"""
    return series.map(AGE_ORDINAL)

def clean_org_size(series:pd.Series)->pd.Series:
    """mapping org-size bands to ordinal integers"""
    return series.map(ORGSIZE_ORDINAL)

def clean_icorpm(series:pd.Series) -> pd.Series:
    """mapping IC OR PM to binary: manager -1,ic - 0"""
    def _map(val):
        if pd.isna(val):
            return np.nan
        v= str(val).lower()
        if"manager"in v or "lead"in v:
            return 1
        return 0
    return series.apply(_map)


def clean_remote_work(series: pd.Series) -> pd.Series:
    """ Map remote work values to ordinal intergers"""
    mapping = {
        "Remote": "Remote",
        "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid",
        "Hybrid (some remote, leans heavy to in-person)": "Hybrid",
        "In-person": "In-person",
        "Your choice (very flexible, you can come in when you want or just as needed)": "Other"
    }
    incomplete = series.map(mapping).fillna("Other")
    return incomplete.map(REMOTE_ORDINAL)


def count_languages(series:pd.Series)-> pd.Series:
    """ CONVER SEMICOLN-SEPARETED 
    LANGUAGE LIST INTO COUNT.
    example:'python;,JavaScrpt;,Sql;HTML' -4
    """
    def _count(val):
       if pd.isna(val) or val== '':
            return np.nan
       return len(str(val).split(';'))

    return series.apply(_count)

def group_rare_countries(series: pd.Series,top_n: int = TOP_N_COUNTRIES)-> pd.Series:
     """Keep only the top n most common countries.Replace all others wit'others'
    """
     top_countries = series.value_counts().head(top_n).index.tolist()
     return series.apply(lambda x: x if x in top_countries else"other")

def clean_industry(series:pd.Series) -> pd.Series:
    "keeping only the top 10 insudtries and replacing all others with other"
    top = series.value_counts().head(10).index.tolist()
    return series.apply(lambda x:x if x in top else "other")


def clean_dev_type(series: pd.Series) -> pd.Series:
    "Picking the primary developer role"
    def _primary(val):
        if pd.isna(val):
            return "Other"
        low = str(val).lower()
        if "full-stack" in low:
            return "Full-stack"
        if "back-end" in low:
            return "Back-end"
        if "front-end" in low:
            return "Front-end"
        if "software" in low:
            return "Software"
        if "desktop" in low:
            return "Desktop"
        if "embedded" in low or "hardware" in low:
            return "Embedded/Hardware"
        if "devops" in low or "cloud" in low or "site reliability" in low:
            return "DevOps/Cloud"
        if "mobile" in low:
            return "Mobile"
        if "data scientist" in low or "machine learning" in low or "ml" in low:
            return "Data/ML"
        if "data engineering" in low or "data analyst" in low:
            return "Data/ML"
        if "manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"
    return series.apply(_primary)
def count_items(series:pd.Series) ->pd.Series:
    """for counting semicoln separeted items in
      a column, we are going to return NaN if balnk"""
    def _count(val):
        if pd.isna(val) or val =="":
            return np.nan
        return len(str(val).split(";"))

    return series.apply(_count)
    
def add_interaction_features(df: pd.DataFrame)-> pd.DataFrame:
        """ why these features help:
        yearcode_sq: diminishing returns on salary grwoth per year
        workExp_sq: sma as above
        Exp_ratio: what fraction of programming /coding time was professional
        Tech_breadth: total tools known = shows seniority.
        """
        df['yearsCode_sq'] = df['YearsCode'] ** 2
        df['WorkExp_sq'] = df['WorkExp'] ** 2
        df['Exp_ratio'] = df['WorkExp']/ (df['YearsCode'] + 1)
        df['Tech_breadth'] = (df['LanguageCount'] + df['DatabaseCount'] + df['PlatformCount']).fillna(0)
        
        return df


def load_and_clean(filepath: str) ->pd.DataFrame:

    #step 0 loading the data
    df= pd.read_csv(filepath,low_memory = False)
    print(f" Raw Shape: {df.shape}\n")


    #keep only rows with a valid salary and do some filtering.
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(SALARY_MIN,SALARY_MAX)]
    print(f"shape after the salary filter:{df.shape}")

# step 1.5 log transforming
    df[LOG_TARGET]= np.log1p(df[TARGET])




    # 2: check whether the feature and target columns exist
    cols_needed = SELECTED_FEATURES + [LOG_TARGET]
    cols_available = [c for c in cols_needed if c in df.columns]
    missing_cols = set(cols_needed)- set(cols_available)
    if missing_cols:
        print(f" you dont have columns:{ missing_cols}in your dataset")

    df = df[cols_available].copy()
    print(f"selected{len(cols_available)} columns, expected 6 columns")

  # 3: Cleaning specific columns

    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_year_code(df['YearsCode'])
    
    if 'WorkExp' in df.columns:
        df['WorkExp'] = clean_work_exp(df['WorkExp'])

    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])

    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])

    if 'Age' in df.columns:
        df['Age'] = clean_age(df['Age'])

    if 'OrgSize' in df.columns:
        df['OrgSize'] = clean_org_size(df['OrgSize'])
    
    if 'ICorPM' in df.columns:
        df['ICorPM'] = clean_icorpm(df['ICorPM'])
    
    if 'RemoteWork' in df.columns:
        df['RemoteWork'] = clean_remote_work(df['RemoteWork'])
    
    if 'DevType' in df.columns:
        df['DevType'] = clean_dev_type(df['DevType'])

    if 'LanguageHaveWorkedWith' in df.columns:
        df['LanguageCount'] = count_languages(df['LanguageHaveWorkedWith'])
        df = df.drop(columns=['LanguageHaveWorkedWith'])
    
    if 'DatabaseHaveWorkedWith' in df.columns:
        df['DatabaseCount'] = count_items(df['DatabaseHaveWorkedWith'])
        df = df.drop(columns=['DatabaseHaveWorkedWith'])

    if 'PlatformHaveWorkedWith' in df.columns:
        df['PlatformCount'] = count_items(df['PlatformHaveWorkedWith'])
        df = df.drop(columns=['PlatformHaveWorkedWith'])
    
    if 'ToolCountWork' in df.columns:
        df['ToolCountWork'] = pd.to_numeric(df['ToolCountWork'], errors='coerce')

    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])



    # step 4: filtering employment
    if'Employment'in df.columns:
        before = len(df)
        df = df[df["Employment"]. isin(EMPLOYMENT_KEEP)]
        df['Employment'] = (df['Employment'] == 'Full-time').astype(int)

        print(f"Employment Filter: {before} -> {len(df)} rows"
                f" we have kept only Full-time and Freelance")
        
    # step 5: Add interaction features (polynomial features)

    df = add_interaction_features(df)

  #drop rows where all features are NaN )Handling thhis edge case)
    
#step 6 drop rows  whose All features  is NaN
    df = df.dropna(how='all')

    print(f"clean data shape: {df.shape}")
    print(f"Missing values per column: \n {df.isna().sum().to_string()}")

    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
        returns categorical columns and numerical columns from the cleaned df.
        This does not include the target variable.
    """
    non_target = [c for c in df.columns if c != LOG_TARGET]
    target_enc= [c for c in TARGET_ENC_FEATURES if c in non_target]

    num_cols = df[non_target].select_dtypes(include=['number']).columns.tolist()
 
    return target_enc, num_cols
