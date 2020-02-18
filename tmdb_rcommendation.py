# Credit to Ibtesam Ahmed
# https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system


import numpy as np
import pandas as pd

if __name__=="__main__":
    file1 = "/Users/masoudabedi/PycharmProjects/Recommendation_System/Data/tmdb_5000_credits.csv"
    file2 = "/Users/masoudabedi/PycharmProjects/Recommendation_System/Data/tmdb_5000_movies.csv"

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1.columns = ['id', 'tittle', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')

    print(df2.head(5))
    print(df2.shape)
    # the shape is (4803, 23)

    c= df2["vote_average"].mean()
    # values in vote_count column that is higher than 90% of values in this column
    m= df2['vote_count'].quantile(0.9)

    # making a data frame with condition of value in vote_count higher or equal to m
    q_movies = df2.copy().loc[df2['vote_count'] >= m]

    print(q_movies.shape)
    # shape of this df is (481, 23)

    # now we define a new metrics and call it score
    def weighted_rating(x, m=m, C=c):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    #Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)

    #Print the top 10 movies
    q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

    # making a dataframe based on sorted df2 by popularity column
    pop= df2.sort_values('popularity', ascending=False)

    # drawing chart for pop
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))

    plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
            color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularity")
    plt.title("Popular Movies")
    plt.show()

    df2['overview'].head(5)

    from sklearn.feature_extraction.text import TfidfVectorizer


