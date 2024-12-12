import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

data = { "x": [0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 10.0], "y": [0.1, 0.2, 0.4, 5.0, 5.1, 5.3, 10.0],}
df = pd.DataFrame(data)

epsilon = 0.5  
min_samples = 1 

coordinates = df[['x', 'y']].to_numpy()
db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean').fit(coordinates)

df['group'] = db.labels_
print(df)