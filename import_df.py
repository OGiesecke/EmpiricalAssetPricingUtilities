import pandas as pd
import numpy as np

def import_df(url,name):
    data = import_zip(url,skiprows_v=3).rename(columns={"Unnamed: 0":"date"})
    data = data.loc[:data[data['Mkt-RF'].isna()].index.tolist()[0]-1,:]
    data['date'] = data['date'].apply(lambda x:x.replace(" ",""))
    data["date"] = pd.to_datetime(data['date'],format="%Y%m")
    data = data.set_index("date")
    data.columns = [f"{name}_{col}" for col in data.columns]
    for col in data.columns:
        data[col]=data[col].apply(lambda x:float(x)/100)
    return data
