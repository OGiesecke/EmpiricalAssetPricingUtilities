
def import_zip(url,skiprows_v=0):
    import zipfile
    import io 
    from io import BytesIO
    import pandas as pd
    import requests

    r = requests.get(url)
    zf = zipfile.ZipFile(BytesIO(r.content))
    print("Imported " + zipfile.ZipFile.namelist(zf)[0])
    df = pd.read_csv(zf.open(zipfile.ZipFile.namelist(zf)[0]), skiprows=skiprows_v)
    return df
