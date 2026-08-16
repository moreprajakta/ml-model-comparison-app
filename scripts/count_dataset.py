import pandas as pd
p='c:/BITS Pilani MTECH/1_Semester/ML/Assignment/Assignment_2/ml-model-comparison-app/heart.csv'
df=pd.read_csv(p)
print(df.shape)
print(len(df.columns)-1)
