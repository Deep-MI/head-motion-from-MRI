import pandas as pd
from glob import glob

for t in ["T1", "T2", "FLAIR"]:
    dfs = []
    for r in f"reports/{t}*.csv":
        df = pd.read_csv(r)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(f"reports/{t}.csv", index=False)  