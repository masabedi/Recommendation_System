import numpy as np
import pandas as pd

df1 = pd.read_csv("/Users/masoudabedi/PycharmProjects/Recommendation_System/Data/tmdb_5000_credits.csv")
df2 = pd.read_csv("/Users/masoudabedi/PycharmProjects/Recommendation_System/Data/tmdb_5000_movies.csv")

df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')


df2.head(5)

c = df2['vote_average'].mean()