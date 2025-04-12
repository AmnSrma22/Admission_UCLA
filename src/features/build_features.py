def build_features(df):
    return df.drop(columns=['Admit_Chance'], errors='ignore')