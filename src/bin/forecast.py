import numpy as np
import pandas as pd
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'forecast_params.yaml'))

print(os.getcwd())
from package import auxiliary_functions as aux
from package import dgp_classes
from package import tensorpca

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(process)s %(levelname)s %(message)s',
    filename='results.log',
    filemode='a'
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

"""
The idea is to do a comparison between three different ways of predicting. The first is to just run a AR model on the series.
The second is to run a single country factor model, the third is to run a tensor factor model on the entire dataset.

he way that I carry out prediction is that I generate a model of say size (T,I,J), I want to predict new entries in the T direction,
 so for finding the loadings I only use floor( T*x ), x in (0,1) observations. With the final 1-T*x observations serving as the set that I wish
 to predict over. Prediction requires two steps. First simulating the path of the factors using an estimated AR model, then plugging those into the 
 equation for y at a future date.   
 Naturally if y_{t} = Lambda1 * F_{t} + Lambda_2 * F_{t-1} + e_{t}
 Then E[y_{t+1}] = Lambda1 E[F_{t+1}] + Lambda_2 F_{t} + E[e_{t+1}] = Lambda1 AF_{t} + Lambda2 F_{t}

 In the same way 
 E[y_{t+h}] = Lambda1E[f_{t+h}] + lambda2 E[F_{t+h-1}] = Lambda1 A^{h}f_{t} + Lambda2 A^{h-1}f_{t}?


When we do prediction, we want to compare multiple different appraoches. The approaches implemented are


1) TensorPCA factor model
2) One factor model for each country
3) VAR on true factor as a baseline predictability meaure


For tensorPCA I take that we know the number of factors.

I also keep the number of time series in a country and the number of countries constant, while increasing the number of periods.


"""
def parse_range_yaml_dict(range_dict):
    """
        I assume that the range dict is two level, first has the ranges and n_values, the second has start and stop for each range.
        {ranges: [{start:5, stop: 10}, {start:1, stop:1}], n_vals: 5 }
        Note that I assum that time dimension is first, country dimension is second and there are three dimensions
    """

    ranges = []
    for item in range_dict["ranges"]:
        ranges.append(np.linspace(item["start"], item["stop"], range_dict["n_vals"], dtype = int))
    return list(zip(*ranges))


def zip_with_arbitrary_middle(tuples_list, middle_items, repetitions):
    res = []
    for tup in tuples_list:
        for i in range(repetitions):
            res.append([tup] + middle_items + [i])
    return res


def load_class_from_string(class_string):
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



def simulate_forecast(sample_size, dgp, noise_scale, max_horizon, rep, convergence_criteria = 0.01, max_iters = 1000):

    with multiprocessing.Lock():
        logging.debug(f"Simulating forecast with {sample_size} (T,I,J), rep {rep}")
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
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logging.debug(f"Starting code with parameters {config}.")
   
    reps = config["repetitions"]
    range_dict = config["range_dict"]
    range_list = list(parse_range_yaml_dict(range_dict))
    noise_scale = config["noise_scale"]
    dgp = load_class_from_string(config["dgp"])
    dgp_kwargs = config["dgp_kwargs"]
    print(f"dgp kwards {dgp_kwargs}")
    max_horizon = config["max_horizon"]
    workers = config["workers"]
  
    logging.debug(f"We have {workers} workers available to us.")
    print(dgp_kwargs)
    
    dgp_instances = [dgp(**arg) for arg in dgp_kwargs]

    mid = [dgp_instances, noise_scale, max_horizon]
    full_zip = []

    for item in dgp_instances:
        for rep in range(reps):
            full_zip.append([range_list[0], item, noise_scale, max_horizon, rep])
    
    logging.debug("Starting the simulation")

    with Pool(workers) as pool:
        result = pool.starmap(simulate_forecast, full_zip)

    err_df = pd.DataFrame(result) 
    err_df = err_df.explode(["tPCA_forecast_error", "flattened_forecast_error"])#.reset_index().rename(columns={"index":"horizon"})
    print(err_df)
    err_df["horizon"] = err_df.groupby(by = ["ar_coefficients", "repetition"]).cumcount() 
    
    logging.debug("The simulation has ended")   
    
    #err_df["sample_size"] = err_df["sample_size"].apply(lambda x: x[0]) # splat the sample size.
    
    pattern = r"_error"
    matching_columns = err_df.filter(regex=pattern).columns
    print(f"Err df \n{err_df}")
    print(f"The matcign columns are {matching_columns}")
    df_melted = pd.melt(err_df, id_vars=['ar_coefficients', 'repetition',"horizon"], 
                    value_vars=matching_columns,
                    var_name='method', value_name='error')
    df_melted.to_csv("./results/forecast_df.csv",index = False)
    print(f"Melted df \n{df_melted}")

    sns.boxplot(x = "ar_coefficients", y = "error", data = df_melted.query("horizon == 0"), hue = "method")
    plt.xlabel("AR coefficient")
    plt.ylabel("RMSE")
    plt.title("RMSE of one step ahead forecasts of Y")   
    plt.savefig("./results/prediction_error.png", dpi = 300, bbox_inches = "tight")
    logging.debug("Everything is done.")