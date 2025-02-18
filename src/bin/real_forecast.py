import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import importlib
import regex as re
import os
import multiprocessing
from multiprocessing import Pool
import logging

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'forecast_params.yaml'))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print(os.getcwd())
from package import auxiliary_functions as aux
from package import tensorpca

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(process)s %(levelname)s %(message)s',
    filename='results.log',
    filemode='a'
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)





def simulate_forecast(df, convergence_criteria = 0.01, max_iters = 1000):

    # First we break down the sample size, the assumption is  that the time factor is in first position
    # We want to have the sample size represent the number of observed periods, so we add max_horizon to the
    # sample size to get the total amount of data that we need to generate.
    # I also assume that the number of countries is the second factor
    row_dict = {}
    T = sample_size[0]
    new_sample = list(sample_size)
    new_sample[0] += max_horizon
    new_sample = tuple(new_sample)

    row_dict["ar_coefficients"] = dgp.ar_coefficients[0][0]
    row_dict["repetition"] = rep

    # Generate the data
    M, s = dgp.generate_data(new_sample)
    Y = dgp_classes.get_Y(M, s, noise_scale) # We have modified the shape in the generate data call

    # Break the data down into observed and unobserved
    observed_Y = Y[:T, :, :]
    unobserved_Y = Y[T:, :, :]

    #baseline_prediction = aux.predict_var(data = M[0], steps = max_horizon)
    #baseline_prediction_factors = [baseline_prediction, M[1], M[2]]
    #baseline_predicted_Y = dgp_classes.get_Y(baseline_prediction_factors, s, noise_scale = 0)

    #basesline_prediction_error = [np.sqrt(np.sum((baseline_predicted_Y[:,i,:] - unobserved_Y[:, i, :])**2, axis = 1)) for i in range(np.shape(Y)[1])]
    #basesline_prediction_error = np.mean(basesline_prediction_error, axis = 0)
    #row_dict["baseline_error"] = basesline_prediction_error

    # Now to flatten to get a point of comparison.
    # Note that we must flatten like the tensorPCA unfolding or we do not recover the factors at all

    flattened_Y = aux.unfold(observed_Y, 0)
    flattened_unobserved_Y = aux.unfold(unobserved_Y,0) # .reshape(unobserved_Y.shape[0], -1)
    s_hat, M_hat = aux.tpca(flattened_Y, (dgp.lags+1)*dgp.n_factors)
    #print(f"Mhat {np.shape(M_hat[0])}")
    #print(f"M {np.shape(M[0])}")
    print(f"How well the flattened recovers F {aux.get_cos_errors(M[0][:T,:], M_hat[0])}")
    print(f"How well the flattened reconstructs Y {np.sum((M_hat[0] @ M_hat[1].T - flattened_Y)**2)}")
    F_hat_prediction = aux.predict_var(data = M_hat[0], steps = max_horizon)
    factor_prediction = F_hat_prediction @ M_hat[1].T
    factor_prediction_errors = np.sqrt( np.sum((factor_prediction - flattened_unobserved_Y)**2, axis = 1) )
    row_dict["flattened_forecast_error"] = factor_prediction_errors
    
    # Now for tensorPCA
    s_hat, M_hat = aux.tpca(observed_Y, (dgp.lags+1)*dgp.n_factors)
    print(f"How well the direct tPCA recovers F {aux.get_cos_errors(M[0][:T,:], M_hat[0])}")
    
    # I have no idea why, but the code does't work if I don't have the following line
    M_hat = [M_hat[i] for i in range(len(M_hat))]
    #print(f"New M_hat {type(M_hat)}")
    #M_hat[0] = M_hat[0][:T,:]
    M_hat_ALS, counter = aux.alternating_least_squares(observed_Y, M_hat,
            convergence_criteria = convergence_criteria, max_iters = max_iters)
    print(f"ALS took {counter} iterations to converge")
    s_hat_ALS = np.ones(shape = len(s))
    for i in range(len(M_hat_ALS)):
        temp = np.linalg.norm(M_hat_ALS[i], axis = 0)
        s_hat_ALS = s_hat_ALS*temp
        M_hat_ALS[i] = M_hat_ALS[i]/temp
    print(f"How well the tPCA recovers F {aux.get_cos_errors(M[0][:T,:], M_hat_ALS[0])}")
    print(f"How well the tPCA reconstructs Y {np.sum((dgp_classes.get_Y(M_hat_ALS,s_hat_ALS, 0) - observed_Y)**2)}")
    F_hat_prediction = aux.predict_var(data = M_hat_ALS[0], steps = max_horizon)
    M_hat_ALS[0] = F_hat_prediction
    ALS_pred_Y = dgp_classes.get_Y(M_hat_ALS, s_hat_ALS, noise_scale = 0)

    prediction_error = np.sqrt(np.sum( (ALS_pred_Y - unobserved_Y)**2, axis = (1,2) ) )
    row_dict["tPCA_forecast_error"] = prediction_error

    return row_dict



if __name__ == "__main__":
    # Load parameters from config file
    
    logging.debug("Starting real forecast")


    df = pl.read_csv(f"{data_path}/cleaned_all.csv")
    # For the first test I only use the dataframe without the fixed values.
    # I want to compare predicting GDP for a single country using a normal factor model to predicting
    # for a single country using a tensor factor model.
    df = df.filter(~pl.col("geo").is_null()) 
    tensor = df.sort(["TIME_PERIOD", "geo", "indic"], descending = False).to_numpy()
    n_periods = len(df["TIME_PERIOD"].unique())
    n_countries = len(df["geo"].unique())
    n_indic = len(df["indic"].unique())
    #tensor = tensor.reshape((n_periods, n_countries, n_indic))
    test_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape(3,2,2)
    print(f"This is test_list \n{test_list}")

    print(np.shape(tensor))

    aaaa



    # First single country forecasts
    countries = df["geo"].unique()
    print(f"These are the countryies {countries}")
    for count, country in enumerate(countries):
        print(f"current country {country}")
        temp = df.filter(pl.col("geo") == f"Netherlands")
        temp = temp.pivot(index = "TIME_PERIOD", values = "OBS_VALUE", columns = "indic").sort("TIME_PERIOD")
        temp = temp.drop("TIME_PERIOD").to_numpy()
        print(temp)
        obs_data = temp[:-4,:]
        unobs_data = temp[-4:,:]
        s_hat, M_hat = aux.tpca(obs_data, 4)
        F_hat_prediction = aux.predict_var(data = M_hat[0], steps = 4)
        print(f"This is s_hat {s_hat}")
        print(f"This is M_hat {M_hat}")
        factor_prediction = F_hat_prediction @ M_hat[1].T
        factor_prediction_errors = np.sqrt( np.sum((factor_prediction - unobs_data)**2, axis = 1) )
        print(f"These are the prediction error {factor_prediction_errors}")
        # Itertative averge updating
        row_dict["flattened_forecast_error"] = row_dict["flattened_forecast_error"] + \
            (factor_prediction_errors-row_dict["flattened_forecast_error"])/(count+2)
        count += 1
