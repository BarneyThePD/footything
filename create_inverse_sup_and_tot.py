import pickle
import numpy as np

folder = "E0_stuff"
sup_estimator = pickle.load(open(folder + "/sup_estimator_pre.pkl", 'rb'))
tot_estimator = pickle.load(open(folder + "/tot_estimator_pre.pkl", 'rb'))

for s in np.arange(-80, 81):
    for t in np.arange(200, 401):
        sup = s/100
        tot = t/100

        print(sup, tot)

        value = sup_estimator.best_estimator_.predict(np.asarray([[sup, tot]]))
        value2 = tot_estimator.best_estimator_.predict(np.asarray([[sup, tot]]))
        print (value, value2)
