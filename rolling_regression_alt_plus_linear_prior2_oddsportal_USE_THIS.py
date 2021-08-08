import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge, Lasso
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
from sklearn.metrics import make_scorer, mean_poisson_deviance


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
    print (Xs.shape, Ys.shape)
    clf = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, Ys)
    return clf

games_details = []

rolling_game_list_sup = []
rolling_game_list_tot = []

rolling_y_list_sup = []
rolling_y_list_tot = []

in_folder = "op_data/JLeague/fit/"
out_folder = "op_data/JLeague/ML_Models/"
files = ["JL2017.csv", "JL2018.csv", "JL2019.csv"]

df = pd.read_csv(in_folder + files[0])
df["NewHomeTeam"] = df["home_team"] + "_0"
df["NewAwayTeam"] = df["away_team"] + "_0"

for x, file in enumerate(files[1:]):
    df2 = pd.read_csv(in_folder + file)
    df2["NewHomeTeam"] = df2["home_team"] + "_" + str(x + 1)
    df2["NewAwayTeam"] = df2["away_team"] + "_" + str(x + 1)

    df = df.append(df2)

df = df.drop_duplicates()
df["sup_per_goal"] = (df["home_underlying"] - df["away_underlying"])/(df["home_underlying"] + df["away_underlying"])
df["ex_total"] = (df["home_underlying"] + df["away_underlying"])
print (df[df.sup_per_goal == -1])
print (df["sup_per_goal"].max())
print (df["sup_per_goal"].min())
print (df["ex_total"].min())
print (df["ex_total"].max())

for x, row in df.iterrows():
    this_row = [row["date"],"", row["NewHomeTeam"], row["NewAwayTeam"], row["home_score"], row["away_score"],
                row["sup_per_goal"], row["ex_total"]]  # row["home_underlying"], row["away_underlying"]]
    games_details.append(this_row)

team_list = []
for game in games_details:
    if game[2] not in team_list:
        team_list.append(game[2])
    if game[3] not in team_list:
        team_list.append(game[3])


csv_rows = []

next_fixtures_list = []
new_data_set = []

differences = []
differences2 = []
for game in games_details:

    date_as_dt = game[0]

    if True: #date_as_dt < datetime(2020, 4, 2):

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

#these models build essentially a season rating for all the teams
first_model_sup = calculate_ridge(rolling_game_list_sup, rolling_y_list_sup, 0)
first_model_tot = calculate_ridge(rolling_game_list_tot, rolling_y_list_tot, 0)
print ("ha", first_model_sup.intercept_)
for x,i in enumerate(team_list):
    print (i, first_model_sup.coef_[x])
#need a dataset whereby we take the teams avareage season ratings and use that to build the expected game sup_per_goal and tot
# might need an inverse as well

new_game_rows_sup = []
new_game_ys_sup = []
new_game_rows_tot = []
new_game_ys_tot = []

print (games_details)
for game in games_details:
    fake_row = [0] * (len(team_list))  # one for ha
    fake_row_2 = [0] * (len(team_list))  # one for ha
    home_ind = team_list.index(game[2])
    away_ind = team_list.index(game[3])
    fake_row[home_ind] = 1
    fake_row[away_ind] = -1

    fake_row_2[home_ind] = 1
    fake_row_2[away_ind] = 1

    fake_row = [(first_model_sup.coef_[home_ind] - first_model_sup.coef_[away_ind] + first_model_sup.intercept_), #sup
                first_model_tot.coef_[home_ind] + first_model_tot.coef_[away_ind] + first_model_tot.intercept_, # tot
                abs((first_model_sup.coef_[home_ind] - first_model_sup.coef_[away_ind] + first_model_sup.intercept_)), # abs diff
                (first_model_sup.coef_[home_ind] - first_model_sup.coef_[away_ind] + first_model_sup.intercept_) if (first_model_sup.coef_[home_ind] - first_model_sup.coef_[away_ind] + first_model_sup.intercept_) > 0 else 0, #home suo
               abs(first_model_sup.coef_[home_ind] - first_model_sup.coef_[away_ind] + first_model_sup.intercept_) if (
                            first_model_sup.coef_[home_ind] - first_model_sup.coef_[
                        away_ind] + first_model_sup.intercept_) < 0 else 0 #away sup

                ]
    new_game_rows_sup.append(fake_row)
    new_game_rows_tot.append(fake_row)
    new_game_ys_sup.append(game[-2])
    new_game_ys_tot.append(game[-1])

load_models = False
print (new_game_rows_sup)
print (new_game_ys_sup)

def my_custom_loss_func(X_train_scaled, Y_train_scaled):
    error, M = 0, 0
  #  print (X_train_scaled)
    for i in range(0, len(Y_train_scaled)):
        if X_train_scaled[i] <= 0:
            X_train_scaled[i] = 100000
        if Y_train_scaled[i] <= 0:
            Y_train_scaled[i] = 100000
 #   print(X_train_scaled)

    print(np.min(X_train_scaled))
    print (np.max(X_train_scaled))
    print ("m", np.min(Y_train_scaled))
    print("m", np.max(Y_train_scaled))
    error = mean_poisson_deviance(X_train_scaled, Y_train_scaled)
    dev = np.max(np.abs(Y_train_scaled - X_train_scaled))
  #  print (np.where(X_train_scaled==100000))
    print (dev)
  #  print (error)
    return error

my_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)

if load_models:
    sup_estimator = pickle.load(open("sup_estimator.pkl", 'rb'))
else:
    param_grid = {'ml_mod__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'poly__degree': [1, 2,3,4,5]}
    #
    # param_grid = {'ml_mod__alpha': [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #               'ply__degree': [1, 2, 3]},
    pipe = Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()), ('ml_mod', Ridge())])
    sup_estimator = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                 cv=5, n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")
    sup_estimator.fit(new_game_rows_sup, new_game_ys_sup)

print (sup_estimator.predict(new_game_rows_sup))
#get total adjustment model

if load_models:
    tot_estimator = pickle.load(open("tot_estimator.pkl", 'rb'))
else:
    # param_grid = {'grad_boost__n_estimators': [10, 100, 1000],
    #               'grad_boost__learning_rate': [0.2, 0.1, 0.05, 0.02],
    #               'grad_boost__max_depth': [1, 2, 4, 6],
    #               'grad_boost__min_samples_leaf': [3, 5, 9, 17, 25],
    #               'grad_boost__max_features': [1.0, 0.3, 0.1]}

    param_grid = {'ml_mod__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'poly__degree': [1, 2,3,4,5]}
    pipe = Pipeline([('poly', PolynomialFeatures()),('scaler', StandardScaler()), ('ml_mod', Ridge())])
    tot_estimator = GridSearchCV(estimator = pipe, param_grid = param_grid,
                          cv = 5, n_jobs = -1, verbose = 2, scoring=my_scorer)
    tot_estimator.fit(new_game_rows_tot, new_game_ys_tot)


pickle.dump(sup_estimator.best_estimator_, open(out_folder+"sup_estimator_pre_lin_5.pkl", 'wb'))
pickle.dump(tot_estimator.best_estimator_, open(out_folder+"tot_estimator_pre_lin_5.pkl", 'wb'))

print (sup_estimator.best_score_)
print (sup_estimator.best_params_)
print (tot_estimator.best_score_)
print (tot_estimator.best_params_)

#these tell me master rating to game ratings
#but i need game ratings to master ratings

#this model essentially teanslates aggegates to games
#so when runnign through games we need to get the implied aggregated differences and then that is what we build the rating from and then reapply this model