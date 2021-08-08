import pandas as pd
from basic_footy_model import fit_grid

df = pd.read_csv("E0 2021.csv")
print (list(df.columns))
underlying_ratings_home = []
underlying_ratings_away = []
for x, row in df.iterrows():
 #   print (row)
    values = fit_grid(
        #row["BbAHh"],
        row["AHh"],
                     # row["BbAvAHH"],
        row["AvgAHH"],
                    #  row["BbAvAHA"],
        row["AvgAHA"],
                      2.5,
                    #  row["BbAv>2.5"],
        row["Avg>2.5"],

                     # row["BbAv<2.5"],
        row["Avg<2.5"],
                      )["x"]
    print (values)
    underlying_ratings_home.append(values[0])
    underlying_ratings_away.append(values[1])


df["home_underlying"] = underlying_ratings_home
df["away_underlying"] = underlying_ratings_away

df.to_csv("E0_underlying_2120.csv", index=False)