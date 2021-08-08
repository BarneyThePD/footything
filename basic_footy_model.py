import csv
import numpy

from scipy import optimize
from scipy import stats


def fit_grid(hc_line, hc_price_home, hc_price_away, ou_line, ou_price_over, ou_price_under):
    print(hc_line, hc_price_home, ou_line, ou_price_over)

    hc_price_home = ((1/hc_price_home) + (1/hc_price_away))/(1/hc_price_home)
    ou_price_under = ((1 / ou_price_over) + (1 / ou_price_under)) / (1 / ou_price_under)

    def football_grid(lines, return_grid=False):
 #       print (lines)
        home_goal_line = lines[0]
        away_goal_line = lines[1]
        correlation = 0.1  # 0.1
        cs_grid = [[0 for i in range(20)] for j in range(20)]
        for home_score in range(20):
            for away_score in range(20):
                new_away_line = away_goal_line + (home_score - home_goal_line) / (
                        home_goal_line ** 0.5) * away_goal_line * correlation
                cs_grid[home_score][away_score] = stats.distributions.poisson.pmf(home_score,
                                                                                  home_goal_line) * stats.distributions.poisson.pmf(
                    away_score, new_away_line)

        if return_grid:
            return cs_grid

        our_hc_prob_home = 0
        our_hc_prob_away = 0
        our_ou_prob_under = 0
        our_ou_prob_over = 0
        for home_score in range(20):
            for away_score in range(20):
                if home_score - away_score + hc_line == 0.25:
                    our_hc_prob_home += cs_grid[home_score][away_score] * 0.5
                elif home_score - away_score + hc_line == -0.25:
                    our_hc_prob_away += cs_grid[home_score][away_score] * 0.5
                elif home_score - away_score + hc_line > 0:
                    our_hc_prob_home += cs_grid[home_score][away_score]
                elif home_score - away_score + hc_line < 0:
                    our_hc_prob_away += cs_grid[home_score][away_score]

                if home_score + away_score - ou_line == 0.25:
                    our_ou_prob_over += cs_grid[home_score][away_score] * 0.5
                elif home_score + away_score - ou_line == -0.25:
                    our_ou_prob_under += cs_grid[home_score][away_score] * 0.5
                elif home_score + away_score - ou_line > 0:
                    our_ou_prob_over += cs_grid[home_score][away_score]
                elif home_score + away_score - ou_line < 0:
                    our_ou_prob_under += cs_grid[home_score][away_score]


        our_hc_prob_home = our_hc_prob_home / float(our_hc_prob_home + our_hc_prob_away)

        our_ou_prob_under = our_ou_prob_under / float(our_ou_prob_under + our_ou_prob_over)

        #	print (our_ou_prob_under)

        how_out = ((1 / float(hc_price_home)) - our_hc_prob_home) ** 2 + ((1 / float(ou_price_under)) - our_ou_prob_under) ** 2
        #	print lines
        #	print hc_line,hc_price,our_hc_prob_home,ou_line,ou_price,our_ou_prob_under
        return how_out

    optimized_values = optimize.minimize(football_grid,
                                         numpy.array([(-hc_line + ou_line) / 2.0, (hc_line + ou_line) / 2.0]),
                                         bounds=((0.01, 6), (0.01, 6)))

    return optimized_values

#print (fit_grid(2.5, 1.73, 2.15, 2.5, 1.49, 2.61))


def produce_football_grid(home_goal_line, away_goal_line):
    #       print (lines)

    correlation = 0.1  # 0.1
    cs_grid = [[0 for i in range(20)] for j in range(20)]
    for home_score in range(20):
        for away_score in range(20):
            new_away_line = away_goal_line + (home_score - home_goal_line) / (
                    home_goal_line ** 0.5) * away_goal_line * correlation
            cs_grid[home_score][away_score] = stats.distributions.poisson.pmf(home_score,
                                                                              home_goal_line) * stats.distributions.poisson.pmf(
                away_score, new_away_line)


    return cs_grid