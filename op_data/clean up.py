import pandas as pd
from datetime import datetime

def get_balanced(lines):
    lines = lines.split(";")
    the_line = None
    diff = 10000
    #print (lines)
    for line in lines:
        if line != "":
            points, a, b = line.split(" ")
            line_diff = abs(0.5 - 1/float(a)) + abs(0.5 - 1/float(b))
    #        print(line_diff)
            if line_diff < diff:
                the_line = line
                diff = line_diff
   #             print (the_line)
  #  print ("f", the_line)
    return the_line.split(" ")

def strip_final_score(score_string):
    score_string = score_string.split(" ")[2]
    return score_string.split(":")

files = ["MLS2021.csv"]
for file in files:

    df = pd.read_csv(file, header=None)
    df = df.dropna()
    df = df.drop_duplicates()
    date, home, away, home_score, away_score, hc_line, hc_home, hc_away, over_under_line, over, under = [], [], [], [], [], [], [], [], [], [], []
    games = []
    for x, row in df.iterrows():
     #   print (row)
        dummy_game = []
        home_team, away_team = row[0].split(" - ")

        if "awarded" in row[2]:
            continue
        home_team_score, away_team_score = strip_final_score(row[2])
        hc_lines = get_balanced(row[4])
        ou_lines = get_balanced(row[5])
        the_date = datetime.strptime(row[1].split(",")[1].strip(), "%d %b %Y")

        dummy_game.append(the_date)
        dummy_game.append(home_team)
        dummy_game.append(away_team)
        dummy_game.append(home_team_score)
        dummy_game.append(away_team_score)
        dummy_game.append(hc_lines[0])
        dummy_game.append(hc_lines[1])
        dummy_game.append(hc_lines[2])
        dummy_game.append(ou_lines[0])
        dummy_game.append(ou_lines[1])
        dummy_game.append(ou_lines[2])

        games.append(dummy_game)

    df2 = pd.DataFrame(games,
                       columns=["date",
                                "home_team",
                                "away_team",
                                "home_score",
                                "away_score",
                                "hc_line",
                                "hc_home",
                                "hc_away",
                                "ou_line",
                                "over",
                                "under"])


    df2.sort_values(by=['date'], inplace=True, ascending=True)
    print (df2)
    print (df2.columns)
    df2.to_csv("JLeague/raw/"+ file, index=False)