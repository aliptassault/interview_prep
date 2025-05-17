import numpy as np
import pandas as pd

dict = np.random.rand(2, 2)
df = pd.DataFrame(dict, columns = ["col1","col2"])
print(df)
df["col3"] = [2,3]
df2= pd.DataFrame(columns=["col1","col2","col3"])
print(df)
for idx, row in df.iterrows():
    new_row = {}
    if row["col3"] > .3:
        new_row["col3"] = row["col3"]
    if row["col2"] > .8:
        new_row["col2"] = row["col2"]
    if new_row:
        df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)

print(df2)
        
