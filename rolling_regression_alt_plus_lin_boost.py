import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge, Lasso
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from sklearn.model_selection import GridSearchCV

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

#df = pd.read_csv("E0_underlying_1918.csv")
#df2 = pd.read_csv("E0_underlying_2019.csv")
df = pd.read_csv("E0_underlying_2120.csv")

#df = df.append(df2)

df["sup_per_goal"] = (df["home_underlying"] - df["away_underlying"])/(df["home_underlying"] + df["away_underlying"])
df["ex_total"] = (df["home_underlying"] + df["away_underlying"])

#need to roll x games/days
#need to build_features

#date, teama, teamb, scorea, scoreb, sup per goal, ex total

games_details = []

for x, row in df.iterrows():

    this_row = [row["Date"], row["Time"], row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"], row["sup_per_goal"], row["ex_total"]] #row["home_underlying"], row["away_underlying"]]
    games_details.append(this_row)

#print (games_details)

team_list = []
for game in games_details:
    if game[2] not in team_list:
        team_list.append(game[2])
    if game[3] not in team_list:
        team_list.append(game[3])



def run_league(args, should_print=False, write_csv=False, optim=True):
    csv_rows = []
    rolling_game_list_sup = []
    rolling_game_list_tot = []

    rolling_y_list_sup = []
    rolling_y_list_tot = []
    next_fixtures_list = []
    new_data_set = []

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
            if game[0] != start_date and len(rolling_game_list_sup) > required_games:
                #trigger calcs

                model_sup = calculate_ridge(rolling_game_list_sup, rolling_y_list_sup, the_alpha)
                model_tot = calculate_ridge(rolling_game_list_tot, rolling_y_list_tot, the_alpha)
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

                        fake_row_2[home_ind] = 1
                        fake_row_2[away_ind] = 1

                        predicted_value_sup = model_pred(model_sup, [fake_row])[0]
                        predicted_value_total = model_pred(model_tot, [fake_row_2])[0]

                        if should_print:
                            print (game2[0], ",",
                                   game2[1], ",",
                                   game2[2],  ",",
                                   game2[3],  ",",
                                   game2[4],  ",",
                                   game2[5],  ",",
                                   predicted_value_sup, ",",
                                   0,  ",",
                                   game2[-2],  ",",
                                   game2[-1],",",)
                        game_dict = {"date": game2[0],
                                     "time": game2[1],
                                     "home": game2[2],
                                     "away": game2[3],
                                     "home_score": game[4],
                                     "away_score": game[5],
                                     "home_team_quality": model_sup.coef_[home_ind],
                                     "away_team_quality": model_sup.coef_[away_ind],
                                     "ex_sup_per_goal": predicted_value_sup,
                                     "home_team_goalie": model_tot.coef_[home_ind],
                                     "away_team_goalie": model_tot.coef_[away_ind],
                                     "ex_tot": predicted_value_total,
                                     "actual_sup_per_goal": game2[-2],
                                     "actual_tot": game2[-1]
                                     }
                        new_data_set.append(game_dict)
                        differences.append([predicted_value_sup, game2[-2]])

            fake_row = [0] * (len(team_list))  # one for ha
            fake_row_2 = [0] * (len(team_list))  # one for ha
            home_ind = team_list.index(game[2])
            away_ind = team_list.index(game[3])
            fake_row[home_ind] = 1
            fake_row[away_ind] = -1

            fake_row_2[home_ind] = 1
            fake_row_2[away_ind] = 1

            rolling_game_list_sup.append(fake_row)
            rolling_y_list_sup.append(game[-2])
            rolling_game_list_tot.append(fake_row_2)
            rolling_y_list_tot.append(game[-1])
            rolling_y_list_sup = rolling_y_list_sup[-(required_games + 1):]
            rolling_game_list_sup = rolling_game_list_sup[-(required_games +1 ):]
            rolling_y_list_tot = rolling_y_list_tot[-(required_games + 1):]
            rolling_game_list_tot = rolling_game_list_tot[-(required_games + 1):]
    #

    differences = np.asarray(differences)
    differences = np.abs(differences[:,0] - differences[:,1])

    print (np.mean(differences), np.max(differences))

    if optim:
        return np.mean(differences)
    else:
        print ("done")
        return new_data_set

#optim = minimize(run_league, np.asarray([1, 40]), method="BFGS")

#print (optim)

new_data_set = run_league([0.0000001, 50], should_print=True, write_csv=True, optim=False)
print (new_data_set)

#get handicap adjustment model
Xs = []
ys = []
ys2 = []

for game in new_data_set:
    Xs.append([game["ex_sup_per_goal"],
               abs(game["ex_sup_per_goal"]),
               (game["ex_sup_per_goal"] > 0),
               game["home_team_quality"],
               game["away_team_quality"],
               game["ex_tot"],
               game["home_team_goalie"],
               game["away_team_goalie"]])

    ys.append([game["actual_sup_per_goal"]])
    ys2.append([game["actual_tot"]])

Xs = np.asarray(Xs)
ys = np.asarray(ys).reshape((len(ys),))
ys2 = np.asarray(ys2).reshape((len(ys2),))

load_models = True

if load_models:
    sup_estimator = pickle.load(open("sup_estimator_lin.pkl", 'rb'))
else:

    sup_estimator = RidgeCV(
                          cv = 5, scoring="neg_mean_absolute_error")
    sup_estimator.fit(Xs, ys)

sup_predictions = sup_estimator.predict(Xs)


#get total adjustment model

if load_models:
    tot_estimator = pickle.load(open("tot_estimator_lin.pkl", 'rb'))
else:

    tot_estimator = RidgeCV(
                          cv = 5, scoring="neg_mean_absolute_error")
    tot_estimator.fit(Xs, ys2)

tot_predictions = tot_estimator.predict(Xs)


for x, game in enumerate(new_data_set):
 #   print (game )
    print (sup_predictions[x], tot_predictions[x])

pickle.dump(sup_estimator, open("sup_estimator_lin.pkl", 'wb'))
pickle.dump(tot_estimator, open("tot_estimator_lin.pkl", 'wb'))

# print (sup_estimator.best_score_)
# print (sup_estimator.best_params_)
# print (tot_estimator.best_score_)
# print (tot_estimator.best_params_)