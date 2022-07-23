import pandas as pa

df = pa.read_csv('lexicon')
# df = df.reset_index()

l = pa.read_csv('lexicon').values.tolist()
d = {}

for index, row in df.iterrows():
    # row = df.iloc[i, :]
    d[row['word']] = row.values.tolist()[1:]
print(d)
