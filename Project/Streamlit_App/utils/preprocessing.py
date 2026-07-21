import pandas as pd
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def load_artifact(name):
    return joblib.load(os.path.join(MODELS_DIR, name))

def build_features_df1(year):
    return pd.DataFrame([[float(year)]], columns=["Year"])

def build_features_df2(year, ftas, age_0_14, age_35_44, age_45_54, age_55_64):
    return pd.DataFrame([[float(year), float(ftas), float(age_0_14),
                          float(age_35_44), float(age_45_54), float(age_55_64)]],
                        columns=["Year", "FTAs",
                                 "% distribution by Age-Group (in Year) - 0-14",
                                 "% distribution by Age-Group (in Year) - 35-44",
                                 "% distribution by Age-Group (in Year) - 45-54",
                                 "% distribution by Age-Group (in Year) - 55-64"])

def build_features_df3(year, arrivals, q1):
    return pd.DataFrame([[float(year), float(arrivals), float(q1)]],
                        columns=["Year", "Arrivals",
                                 "% Distribution by Quarter - 1st Quarter (Jan-Mar)"])

def build_features_df4(year, rank):
    return pd.DataFrame([[float(year), float(rank)]],
                        columns=["Year", "Rank of India"])

def build_features_df5(arrivals, region_cat, country_cat):
    return pd.DataFrame([[float(arrivals), float(region_cat), float(country_cat)]],
                        columns=["Arrivals (in numbers)", "Region_cat",
                                 "Country of Nationality_cat"])

def build_features_df6(domestic_1920, foreign_1920, circle_new, monument_new):
    return pd.DataFrame([[float(domestic_1920), float(foreign_1920),
                          float(circle_new), float(monument_new)]],
                        columns=["Domestic-2019-20", "Foreign-2019-20",
                                 "Circle_new", "Name of the Monument_new"])

def build_features_df7(arr_2017, arr_2018, region_labels, country_freq):
    region_cols = [
        "Region_Australasia", "Region_Central and South America",
        "Region_East Asia", "Region_Eastern Europe",
        "Region_North America", "Region_Not Classified elsewhere",
        "Region_South Asia", "Region_South East Asia",
        "Region_West Asia", "Region_Western Europe",
    ]
    row = [float(arr_2017), float(arr_2018)]
    for rc in region_cols:
        row.append(1 if rc.replace("Region_", "") == region_labels else 0)
    row.append(float(country_freq))
    cols = ["Number of Arrivals - 2017", "Number of Arrivals - 2018"] + region_cols + ["Country of Nationality_new"]
    return pd.DataFrame([row], columns=cols)

def build_features_df8(state_label, total_2019):
    return pd.DataFrame([[float(state_label), float(total_2019)]],
                        columns=["States/UTs_new", "Total_2019"])
