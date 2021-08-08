import pandas as pd
from basic_footy_model import fit_grid

files = ["MLS2021.csv"]

for file in files:

    in_folder = "op_data/MLS/raw/"
    out_folder = "op_data/MLS/fit/"
    df = pd.read_csv(in_folder + file)
    print (list(df.columns))
    underlying_ratings_home = []
    underlying_ratings_away = []
    for x, row in df.iterrows():
     #   print (row)
        values = fit_grid(
            #row["BbAHh"],
            row["hc_line"],
                         # row["BbAvAHH"],
            row["hc_home"],
                        #  row["BbAvAHA"],
            row["hc_away"],
                          row["ou_line"],
                        #  row["BbAv>2.5"],
            row["over"],

                         # row["BbAv<2.5"],
            row["under"],
                          )["x"]
        print (values)
        underlying_ratings_home.append(values[0])
        underlying_ratings_away.append(values[1])


    df["home_underlying"] = underlying_ratings_home
    df["away_underlying"] = underlying_ratings_away

    df.to_csv((out_folder + file), index=False)