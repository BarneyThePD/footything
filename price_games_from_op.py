import json
from basic_footy_model import produce_football_grid
folder = "op_data/MLS/"
# Opening JSON file
f = open(folder + 'ratings/ratings.json', )
import pickle
import numpy as np

# returns JSON object as
# a dictionary
data = json.load(f)

games = [
# ["Hokkaido Consadole Sapporo","Urawa Reds"],
# ["Shimizu S-Pulse","Yokohama F. Marinos"],
# ["Vissel Kobe","Kashiwa Reysol"],
# ["Yokohama FC","Nagoya Grampus"],
# ["Avispa Fukuoka", "Sanfrecce Hiroshima"],
# ["Cerezo Osaka","Vegalta Sendai"],
# ["Oita Trinita","Kawasaki Frontale"],
# ["Sagan Tosu","FC Tokyo"],
# ["Shonan Bellmare","Kashima Antlers"],
["Houston Dynamo","Colorado Rapids"],
# ["Chicago Fire","New York Red Bulls"],
# ["Inter Miami","Nashville SC"],
# ["New England Revolution","Philadelphia Union"],
# ["San Jose Earthquakes","Los Angeles FC"],
# ["DC United","Club de Foot Montreal"],
# ["Los Angeles Galaxy","Vancouver Whitecaps"],
# ["San Jose Earthquakes","Vancouver Whitecaps"],
         ]

sup_estimator = pickle.load(open(folder + "ML_models/sup_estimator_pre_lin_5.pkl", 'rb'))
tot_estimator = pickle.load(open(folder + "ML_models/tot_estimator_pre_lin_5.pkl", 'rb'))


for game in games:
    home_team, away_team = game
    print (home_team, away_team)
    ori_ex_sup = data[home_team]["sup"] - data[away_team]["sup"] + data["ints"]["sup"]
    ori_ex_tot = data[home_team]["tot"] + data[away_team]["tot"] + data["ints"]["tot"]

    game_ml_data = [[ori_ex_sup,
                     ori_ex_tot,
                     abs(ori_ex_sup),
                     ori_ex_sup if ori_ex_sup > 0 else 0,
                     -ori_ex_sup if ori_ex_sup < 0 else 0]]

    estim_sup = sup_estimator.predict(np.asarray((game_ml_data)))
    estim_tot = tot_estimator.predict(np.asarray((game_ml_data)))

    ex_home_goals = (estim_sup * estim_tot + estim_tot)/2
    ex_away_goals = (estim_tot - ex_home_goals)
    print (ex_home_goals, ex_away_goals)
    grid = np.asarray(produce_football_grid(ex_home_goals[0], ex_away_goals[0]))
    inds = np.indices(grid.shape)
    home_scores = inds[0]
    away_scores = inds[1]

    home_win = np.where(home_scores > away_scores, grid, 0).sum()
    draw = np.where(home_scores == away_scores, grid, 0).sum()
    away_win = np.where(home_scores < away_scores, grid, 0).sum()
    print (1/home_win, 1/draw, 1/away_win)


    line = -1
    line_home = np.where(home_scores + line - 0.5 >= away_scores, grid, 0).sum()
    line_home += np.where(home_scores + line - 0.25 == away_scores, grid, 0).sum() * 0.5
    line_away = np.where(home_scores + line <= away_scores - 0.5, grid, 0).sum()
    line_away += np.where(home_scores + line == away_scores - 0.25, grid, 0).sum() * 0.5
    print(line, (line_home + line_away) / line_home, (line_home + line_away) / line_away)

    line = -0.5
    line_home = np.where(home_scores + line - 0.5 >= away_scores, grid, 0).sum()
    line_home += np.where(home_scores + line - 0.25 == away_scores, grid, 0).sum() * 0.5
    line_away = np.where(home_scores + line <= away_scores - 0.5, grid, 0).sum()
    line_away += np.where(home_scores + line == away_scores - 0.25, grid, 0).sum() * 0.5
    print(line, (line_home + line_away) / line_home, (line_home + line_away) / line_away)

    line = 0
    line_home = np.where(home_scores + line - 0.5 >= away_scores, grid, 0).sum()
    line_home += np.where(home_scores + line - 0.25 == away_scores, grid, 0).sum() * 0.5
    line_away = np.where(home_scores + line <= away_scores - 0.5, grid, 0).sum()
    line_away += np.where(home_scores + line == away_scores - 0.25, grid, 0).sum() * 0.5
    print(line, (line_home + line_away) / line_home, (line_home + line_away) / line_away)

    line = 0.5
    line_home = np.where(home_scores + line - 0.5 >= away_scores, grid, 0).sum()
    line_home += np.where(home_scores + line - 0.25 == away_scores, grid, 0).sum() * 0.5
    line_away = np.where(home_scores + line <= away_scores - 0.5, grid, 0).sum()
    line_away += np.where(home_scores + line == away_scores - 0.25, grid, 0).sum() * 0.5
    print(line, (line_home + line_away) / line_home, (line_home + line_away) / line_away)

    line = 1
    line_home = np.where(home_scores + line - 0.5 >= away_scores, grid, 0).sum()
    line_home += np.where(home_scores + line - 0.25 == away_scores, grid, 0).sum() * 0.5
    line_away = np.where(home_scores + line <= away_scores - 0.5, grid, 0).sum()
    line_away += np.where(home_scores + line == away_scores - 0.25, grid, 0).sum() * 0.5
    print(line, (line_home + line_away) / line_home, (line_home + line_away) / line_away)

    ou_line = 2.5
    line_o = np.where(home_scores + away_scores > ou_line, grid, 0).sum()
    line_u = np.where(home_scores + away_scores < ou_line, grid, 0).sum()
    print(ou_line, (line_o + line_u) / line_o, (line_o + line_u) / line_u)

    ou_line = 3.5
    line_o = np.where(home_scores + away_scores > ou_line, grid, 0).sum()
    line_u = np.where(home_scores + away_scores < ou_line, grid, 0).sum()
    print(ou_line, (line_o + line_u) / line_o, (line_o + line_u) / line_u)