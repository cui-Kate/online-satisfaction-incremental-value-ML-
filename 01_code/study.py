from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_data" / "online_education_review.csv"
RES = ROOT / "03_results"
FIG = ROOT / "04_figures"
EXPECTED_SHA = "747852bcf1a7f3a7be8a2bccc56de224649627899a8f6daa853c7b6c0056e82c"
SEED = 42
FIELDS = [
    ("Gender", "gender", "Gender", "nominal"),
    ("Home Location", "residence", "Home location", "nominal"),
    ("Level of Education", "stage", "Education level", "nominal"),
    ("Age(Years)", "age", "Age", "numeric"),
    ("Number of Subjects", "subjects", "Number of subjects", "numeric"),
    ("Device type used to attend classes", "device", "Attendance device", "nominal"),
    ("Economic status", "economic", "Economic status", "nominal"),
    ("Family size", "family", "Family size", "numeric"),
    ("Internet facility in your locality", "internet", "Local internet rating", "numeric"),
    ("Are you involved in any sports?", "sports", "Sports participation", "nominal"),
    ("Do elderly people monitor you?", "monitoring", "Monitoring by elders", "nominal"),
    ("Study time (Hours)", "study", "Study time", "numeric"),
    ("Sleep time (Hours)", "sleep", "Sleep time", "numeric"),
    ("Time spent on social media (Hours)", "social", "Social media time", "numeric"),
    ("Interested in Gaming?", "gaming", "Gaming interest", "nominal"),
    ("Have separate room for studying?", "room", "Separate study room", "nominal"),
    ("Engaged in group studies?", "group", "Group study", "nominal"),
    ("Average marks scored before pandemic in traditional classroom", "grades", "Prior marks band", "band"),
    ("Your interaction in online mode", "interaction", "Online interaction", "numeric"),
    ("Clearing doubts with faculties in online mode", "clarification", "Instructor help with questions", "numeric"),
    ("Interested in?", "interest", "Study preference", "nominal"),
    ("Performance in online", "performance", "Self-rated online performance", "numeric"),
    ("Your level of satisfaction in Online Education", "outcome", "Overall online satisfaction", "outcome"),
]
LABELS = {key: label for _,key,label,_ in FIELDS}
NOMINAL = [key for _,key,_,kind in FIELDS if kind == "nominal"]
PAIR = ["interaction", "clarification"]
PRIMARY = [key for _,key,_,_ in FIELDS if key not in ("outcome", "performance")]

def load_data():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError("Source CSV differs from the published Mendeley V1 file.")
    raw = pd.read_csv(DATA)
    expected = [row[0] for row in FIELDS]
    if list(raw.columns) != expected or raw.isna().any().any():
        raise ValueError("Unexpected schema or missing observations.")
    df = raw.rename(columns={old:key for old,key,_,_ in FIELDS}).copy()
    for key in NOMINAL:
        df[key] = df[key].str.strip().str.lower()
    bands = ["0-10"] + [f"{i+1}-{i+10}" for i in range(10,100,10)]
    df["grades"] = df["grades"].map({b:i for i,b in enumerate(bands)})
    for key in ("internet", *PAIR):
        if not df[key].isin(range(1,6)).all():
            raise ValueError(f"Out-of-range ratings in {key}")
    if df["grades"].isna().any() or not df["outcome"].isin(["Bad","Average","Good"]).all():
        raise ValueError("Unexpected marks band or outcome.")
    return raw, df

def candidates():
    return {
        "Logistic regression": (LogisticRegression(max_iter=1600, random_state=42), {"C":[0.1,1,10]}),
        "Random forest": (RandomForestClassifier(n_estimators=180, max_features="sqrt", random_state=42,n_jobs=1), {"min_samples_leaf":[1,4,10],"max_depth":[None,8]}),
        "Gradient boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), {"max_depth":[1,2,3],"learning_rate":[0.04,0.1]}),
        "XGBoost": (XGBClassifier(n_estimators=120,learning_rate=0.06,n_jobs=1,tree_method="hist",eval_metric="logloss",random_state=42), {"max_depth":[2,4],"min_child_weight":[1,6]}),
        "AdaBoost": (AdaBoostClassifier(n_estimators=90,algorithm="SAMME",random_state=42), {"learning_rate":[0.15,0.6,1]}),
        "Support vector machine": (SVC(kernel="rbf",probability=True,random_state=42), {"C":[0.5,2,8]}),
        "Nearest neighbors": (KNeighborsClassifier(n_jobs=1), {"n_neighbors":[7,19,35],"weights":["uniform","distance"]}),
    }

def transformer(features):
    numeric = [k for k in features if k not in NOMINAL]
    categorical = [k for k in features if k in NOMINAL]
    return ColumnTransformer([("num",StandardScaler(),numeric),
                              ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)],
                             sparse_threshold=0)

def tune(model_name, features, df, y, train_idx, scoring="average_precision"):
    estimator, params = candidates()[model_name]
    pipe = Pipeline([("transform",transformer(features)),("model",estimator)])
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
    grid = GridSearchCV(pipe,{"model__"+k:v for k,v in params.items()},cv=cv,scoring=scoring,
                        refit=True,n_jobs=1,error_score="raise",return_train_score=False)
    grid.fit(df.loc[train_idx,features],y.loc[train_idx])
    return grid

def prob(fit, x):
    cls = list(fit.classes_)
    return fit.predict_proba(x)[:,cls.index(1)]

def write_json(name, obj):
    (RES/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False, default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+"\n",encoding="utf-8")

