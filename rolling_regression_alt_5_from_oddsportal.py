import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge, Lasso
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from random import random

def get_team_ranks(teams, coefs):

    ranks = (-coefs).argsort()

    for i in range(len(team_list)):
        print (i+1, teams[ranks[i]], coefs[ranks[i]])

def model_pred(model, games):
    games = np.asarray(games)
    return model.predict(games)

def calculate_ridge(Xs, ys, alpha):#
 #   print ("rolling", len(Xs), len(ys))
  #  print (Xs)
   # print(ys)
    Xs = np.asarray(Xs)
    Ys = np.asarray(ys)
 #   Ys = Ys.reshape((Ys.shape[0], 1))
  #  print (Xs)
  #  print (Ys)
  #  print (Xs.shape, Ys.shape)
    clf = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, Ys)
    return clf


def get_inverse_sup_and_tot(sup_goal, tot_goal, sup_grid, tot_grid, sup_value_grid, tot_value_grid):
    new_grid = np.abs(sup_grid - sup_goal) + np.abs(tot_grid - tot_goal) + np.abs(sup_value_grid - sup_goal) / 100 + np.abs(
                tot_value_grid - tot_goal) / 100 + random() / 10000000

    cell = np.where(new_grid == new_grid.min())
    print(new_grid.min())
    print(np.abs(sup_grid - sup_goal)[cell], np.abs(tot_grid - tot_goal)[cell], np.abs(sup_value_grid - sup_goal)[cell], np.abs(tot_value_grid - tot_goal)[cell])
    if sup_value_grid[cell] == 0.79 and tot_value_grid[cell] == 3.16:
        pass
        i = 0
    return sup_value_grid[cell], tot_value_grid[cell]

class BetMaker:
    def __init__(self):
        self.hc_bets = 0
        self.ou_bets = 0
        self.hc_th = 0.25
        self.ou_th = 0.25
        self.hc_pl = 0
        self.ou_pl = 0

import pickle

folder = "op_data/MLS/"
file = "MLS2021.csv"

sup_estimator = pickle.load(open(folder + "ML_models/sup_estimator_pre_lin_5.pkl", 'rb'))
tot_estimator = pickle.load(open(folder + "ML_models/tot_estimator_pre_lin_5.pkl", 'rb'))

sup_list = []
tot_list = []
sup_value_list = []
tot_value_list = []
for s in np.arange(-80, 81):
    #print(s)
    sup_temp = []
    tot_temp = []
    sup_value_temp = []
    tot_value_temp = []
    for t in np.arange(200, 501):
        sup = s / 100
        tot = t / 100



        value = sup_estimator.predict(np.asarray([[sup, tot, abs(sup), sup if sup > 0 else 0, -sup if sup <0 else 0]]))[0]
        value2 = tot_estimator.predict(np.asarray([[sup, tot, abs(sup), sup if sup > 0 else 0, -sup if sup <0 else 0]]))[0]

        sup_temp.append(value)
        tot_temp.append(value2)
        sup_value_temp.append(sup)
        tot_value_temp.append(tot)

    sup_list.append(sup_temp)
    tot_list.append(tot_temp)
    sup_value_list.append(sup_value_temp)
    tot_value_list.append(tot_value_temp)

sup_list = np.asarray(sup_list)
tot_list = np.asarray(tot_list)
sup_value_list = np.asarray(sup_value_list)
tot_value_list = np.asarray(tot_value_list)

print ("build grids")


df = pd.read_csv(folder + "fit/" + file)

df["sup_per_goal"] = (df["home_underlying"] - df["away_underlying"])/(df["home_underlying"] + df["away_underlying"])
df["ex_total"] = (df["home_underlying"] + df["away_underlying"])

#need to roll x games/days
#need to build_features

#date, teama, teamb, scorea, scoreb, sup per goal, ex total

games_details = []

for x, row in df.iterrows():
    sup, tot = get_inverse_sup_and_tot(row["sup_per_goal"], row["ex_total"], sup_list, tot_list, sup_value_list, tot_value_list)
    print (row["sup_per_goal"], row["ex_total"], sup, tot )
    # this_row = [row["Date"],
    #             row["Time"],
    #             row["HomeTeam"],
    #             row["AwayTeam"],
    #             row["FTHG"],
    #             row["FTAG"],
    #             row["AHh"],
    #             row["AvgAHH"],
    #             row["AvgAHA"],
    #             row["Avg>2.5"],
    #             row["Avg<2.5"],
    #             row["sup_per_goal"],
    #             row["ex_total"],
    #             sup[0],
    #             tot[0]] #row["home_underlying"], row["away_underlying"]]

    this_row =[row["date"],
     "",  # row["Time"],
     row["home_team"],
     row["away_team"],
     row["home_score"],
     row["away_score"],
     row["sup_per_goal"],
     row["ex_total"],
     sup[0],
     tot[0]]

    print (this_row)
    games_details.append(this_row)

#print (games_details)

team_list = []
for game in games_details:
    if game[2] not in team_list:
        team_list.append(game[2])
    if game[3] not in team_list:
        team_list.append(game[3])



def run_league(args, should_print=False, write_csv=False):

    bm = BetMaker()
    total_adjust = [0] * len(team_list)
    sup_adjust = [0] * len(team_list)

    csv_rows = []
    rolling_game_list = []
    rolling_game_list2 = []
    rolling_y_list = []
    rolling_y_list_2 = []
    next_fixtures_list = []

    start_date = None
    the_alpha = 0.001
    game_carry = 0.0125
    required_games = 50
    the_alpha = args[0]
    required_games = int(args[1])
    differences = []
    differences2 = []
    for game in games_details:

      #  date_as_dt = datetime.strptime(game[0], "%d/%m/%Y")

        if True:#date_as_dt < datetime(2022, 3, 2):
            if game[0] != start_date and len(rolling_game_list) > required_games:
                #trigger calcs

                model = calculate_ridge(rolling_game_list, rolling_y_list, the_alpha)
                model2 = calculate_ridge(rolling_game_list2, rolling_y_list_2, the_alpha)
          #      if should_print:
          #          print ("ha", model.coef_[-1])
                #    get_team_ranks(team_list, model.coef_)

                start_date = game[0]

                for game2 in games_details:
                    if game2[0] == start_date:
                        fake_row = [0] * (len(team_list)) #one for ha
                        fake_row_2 = [0] * (len(team_list))  # one for ha
                        home_ind = team_list.index(game2[2])
                        away_ind = team_list.index(game2[3])
                        fake_row[home_ind] = 1
                        fake_row[away_ind] = -1
                        #fake_row[-1] = 1

                        fake_row_2[home_ind] = 1
                        fake_row_2[away_ind] = 1
               #         fake_row_2[-1] = -1

                        predicted_value = model_pred(model, [fake_row])[0] + sup_adjust[home_ind] - sup_adjust[away_ind]
                        predicted_value_2 = model_pred(model2, [fake_row_2])[0] + total_adjust[home_ind] + total_adjust[
                            away_ind]
                        actual_sup_prd = sup_estimator.predict(np.asarray([[predicted_value,
                                                                                            predicted_value_2,
                                                                                            abs(predicted_value),
                                                                                            predicted_value if predicted_value > 0 else 0,
                                                                                            -predicted_value if predicted_value < 0 else 0]]))[0]
                        actual_tot_prd = \
                        tot_estimator.predict(np.asarray([[predicted_value,
                                                                                            predicted_value_2,
                                                                                            abs(predicted_value),
                                                                                            predicted_value if predicted_value > 0 else 0,
                                                                                            -predicted_value if predicted_value < 0 else 0]]))[0]




                        if should_print:
                            print (game2[0], ",",
                                   game2[1], ",",
                                   game2[2],  ",",
                                   game2[3],  ",",
                                   game2[4],  ",",
                                   game2[5],  ",",
                                   predicted_value, ",",
                                   predicted_value_2, ",",
                                   actual_sup_prd, ",",
                                   actual_tot_prd,  ",",
                                   game2[-2], ",",
                                   game2[-1], ",",
                                   game2[-4],  ",",
                                   game2[-3])
                         #   print (model.coef_[home_ind], model.coef_[away_ind], model.intercept_)


                        # sup update
                        if game2[4] + game2[5] == 0:
                            sup_adjust[home_ind] = 0
                            sup_adjust[away_ind] = 0
                        else:
                            sup_adjust[home_ind] = ((game2[4] - game2[5]) / (game2[4] + game2[5]) - game2[
                                -4]) * game_carry
                            sup_adjust[away_ind] = ((-game2[4] + game2[5]) / (game2[4] + game2[5]) + game2[
                                -4]) * game_carry

                        # tot_update
                        total_adjust[home_ind] = ((game2[4] + game2[5]) - game2[-3]) * game_carry
                        total_adjust[away_ind] = ((game2[4] + game2[5]) - game2[-3]) * game_carry

            fake_row = [0] * (len(team_list))  # one for ha
            fake_row_2 = [0] * (len(team_list))  # one for ha
            home_ind = team_list.index(game[2])
            away_ind = team_list.index(game[3])
            fake_row[home_ind] = 1
            fake_row[away_ind] = -1

            fake_row_2[home_ind] = 1
            fake_row_2[away_ind] = 1

            rolling_game_list.append(fake_row)
            rolling_game_list2.append(fake_row_2)
            rolling_y_list.append(game[-2])
            rolling_y_list_2.append(game[-1])
            rolling_y_list = rolling_y_list[-(required_games + 1):]
            rolling_y_list_2 = rolling_y_list_2[-(required_games + 1):]
            rolling_game_list = rolling_game_list[-(required_games +1 ):]
            rolling_game_list2 = rolling_game_list2[-(required_games + 1):]

    model = calculate_ridge(rolling_game_list, rolling_y_list, the_alpha)
    model2 = calculate_ridge(rolling_game_list2, rolling_y_list_2, the_alpha)

    rating_dict = {}
    print ("******************")
    for x, team in enumerate(team_list):
        print (team, model.coef_[x] + sup_adjust[x], model2.coef_[x] + total_adjust[x])
        rating_dict[team] = {"sup": model.coef_[x] + sup_adjust[x],
                             "tot": model2.coef_[x] + total_adjust[x]}

    print ("sup int", model.intercept_)
    print("tot int", model2.intercept_)
    rating_dict["ints"] = {"sup": model.intercept_,
                             "tot": model2.intercept_}
    import json
    with open(folder + "ratings/ratings.json", "w") as outfile:
        json_object = json.dump(rating_dict, outfile)
    return None

#optim = minimize(run_league, np.asarray([1, 40]), method="BFGS")

#print (optim)

run_league([0.0000001, 50], should_print=True, write_csv=True)
