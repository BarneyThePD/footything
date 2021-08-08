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

def calculate_ridge(Xs, ys, alpha):
    Xs = np.asarray(Xs)
    Ys = np.asarray(ys)
 #   Ys = Ys.reshape((Ys.shape[0], 1))
  #  print (Xs)
  #  print (Ys)
  #  print (Xs.shape, Ys.shape)
    clf = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, Ys)
    return clf


def get_inverse_sup_and_tot(sup_goal, tot_goal, sup_grid, tot_grid, sup_value_grid, tot_value_grid):
    new_grid = np.abs(sup_grid - sup_goal) + np.abs(tot_grid - tot_goal) + np.abs(sup_value_grid - sup_goal) / 10000 + np.abs(
                tot_value_grid - tot_goal) / 10000 + random() / 10000000

    cell = np.where(new_grid == new_grid.min())
    print(new_grid.min())
    print(np.abs(sup_grid - sup_goal)[cell], np.abs(tot_grid - tot_goal)[cell], np.abs(sup_value_grid - sup_goal)[cell], np.abs(tot_value_grid - tot_goal)[cell])
    if sup_value_grid[cell] == -0.5 and tot_value_grid[cell] == 3.96:
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

folder = "E0_stuff"
sup_estimator = pickle.load(open(folder + "/sup_estimator_pre_lin.pkl", 'rb'))
tot_estimator = pickle.load(open(folder + "/tot_estimator_pre_lin.pkl", 'rb'))

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
    for t in np.arange(200, 401):
        sup = s / 100
        tot = t / 100



        value = sup_estimator.best_estimator_.predict(np.asarray([[sup, tot]]))[0]
        value2 = tot_estimator.best_estimator_.predict(np.asarray([[sup, tot]]))[0]

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

df = pd.read_csv("E0_underlying_2120.csv")

df["sup_per_goal"] = (df["home_underlying"] - df["away_underlying"])/(df["home_underlying"] + df["away_underlying"])
df["ex_total"] = (df["home_underlying"] + df["away_underlying"])

#need to roll x games/days
#need to build_features

#date, teama, teamb, scorea, scoreb, sup per goal, ex total

games_details = []

for x, row in df.iterrows():
    sup, tot = get_inverse_sup_and_tot(row["sup_per_goal"], row["ex_total"], sup_list, tot_list, sup_value_list, tot_value_list)
    print (row["sup_per_goal"], row["ex_total"], sup, tot )
    this_row = [row["Date"],
                row["Time"],
                row["HomeTeam"],
                row["AwayTeam"],
                row["FTHG"],
                row["FTAG"],
                row["AHh"],
                row["AvgAHH"],
                row["AvgAHA"],
                row["Avg>2.5"],
                row["Avg<2.5"],
                row["sup_per_goal"],
                row["ex_total"],
                sup[0],
                tot[0]] #row["home_underlying"], row["away_underlying"]]

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


    csv_rows = []
    rolling_game_list = []
    rolling_game_list2 = []
    rolling_y_list = []
    rolling_y_list_2 = []
    next_fixtures_list = []

    start_date = None
    the_alpha = 0.001
    required_games = 50
    the_alpha = args[0]
    required_games = int(args[1])
    differences = []
    differences2 = []
    for game in games_details:

        date_as_dt = datetime.strptime(game[0], "%d/%m/%Y")

        if date_as_dt < datetime(2022, 3, 2):
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

                        predicted_value = model_pred(model, [fake_row])[0]
                        predicted_value_2 = model_pred(model2, [fake_row_2])[0]
                        actual_sup_prd = sup_estimator.best_estimator_.predict(np.asarray([[predicted_value, predicted_value_2]]))[0]
                        actual_tot_prd = \
                        tot_estimator.best_estimator_.predict(np.asarray([[predicted_value, predicted_value_2]]))[0]

                        date_as_dt2 = datetime.strptime(game2[0], "%d/%m/%Y")
                        if date_as_dt2 < datetime(2021, 5, 1):
                            #hc bet
                            if ((actual_sup_prd* actual_tot_prd) - game2[-4] * game[-3]) > bm.hc_th:
                                bm.hc_bets += 1
                                if game[4] - game[5] + game[6] >= 0.5:
                                    bm.hc_pl += 100 * (game[7] -1)
                                elif game[4] - game[5] + game[6] == 0.25:
                                    bm.hc_pl += 100 * (game[7] -1)/2
                                elif game[4] - game[5] + game[6] == 0:
                                    pass
                                elif game[4] - game[5] + game[6] == -0.25:
                                    bm.hc_pl -= 100/2
                                else:
                                    bm.hc_pl -= 100
                            elif ((actual_sup_prd* actual_tot_prd) - game2[-4] * game[-3]) < -bm.hc_th:
                                bm.hc_bets += 1
                                if game[5] - game[4] - game[6] >= 0.5:
                                    bm.hc_pl += 100 * (game[8] - 1)
                                elif game[5] - game[4] - game[6] == 0.25:
                                    bm.hc_pl += 100 * (game[8] - 1) / 2
                                elif game[5] - game[4] - game[6] == 0:
                                    pass
                                elif game[5] - game[4] - game[6] == -0.25:
                                    bm.hc_pl -= 100 / 2
                                else:
                                    bm.hc_pl -= 100


                        if should_print:
                            print (game2[0], ",",
                                   game2[1], ",",
                                   game2[2],  ",",
                                   game2[3],  ",",
                                   game2[4],  ",",
                                   game2[5],  ",",
                                   actual_sup_prd, ",",
                                   actual_tot_prd,  ",",
                                   game2[-4],  ",",
                                   game2[-3])

                        differences.append([predicted_value, game2[-2]])

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
    #
    # for d in differences:
    #     print (d[0], d[1], d[2])
    differences = np.asarray(differences)
 #   print(differences)

    differences = np.abs(differences[:,0] - differences[:,1])
 #   print (args)

    print (np.mean(differences), np.max(differences))
    print (bm.hc_pl, bm.hc_bets)

    return np.mean(differences)

#optim = minimize(run_league, np.asarray([1, 40]), method="BFGS")

#print (optim)

run_league([0.0000001, 50], should_print=True, write_csv=True)
