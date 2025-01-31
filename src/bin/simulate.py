import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import importlib
import regex as re
import os
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import logging
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'simulate_params.yaml'))

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

def parse_range_yaml_dict(range_dict):
    """
        I assume that the range dict is two level, first has the ranges and n_values, the second has start and stop for each range.
        {ranges: [{start:5, stop: 10}, {start:1, stop:1}], n_vals: 5 }
    """

    ranges = []
    for item in range_dict["ranges"]:
        ranges.append(np.linspace(item["start"], item["stop"], range_dict["n_vals"], dtype = int))
    return list(zip(*ranges))

def load_class_from_string(class_string):
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def zip_with_arbitrary_middle(tuples_list, middle_items, repetitions):
    result = []
    for tup in tuples_list:
        for i in range(repetitions):
            result.append([tup] + middle_items + [i])
    return result


def simulate_3d(sample_size, noise_scale, dgp, rep, convergence_criteria = 0.01, max_iters = 1000):
    with multiprocessing.Lock():
        logging.debug(f"Simulating with {sample_size}, rep {rep}")
    row_dict = {f"factor_{i}_err": "" for i in range(len(sample_size))}
    row_dict["sample_size"] = sample_size
    row_dict["repetition"] = rep


    #row_dict["s_err"]
    #print(sample_size)
    M, s = dgp.generate_data(sample_size)
    Y = dgp_classes.get_Y(M, s, noise_scale) # We have modified the shape in the generate data call
    
    s_hat, M_hat = aux.tpca(Y, (dgp.lags+1)*dgp.n_factors)
    
    M_hat = [M_hat[i] for i in range(len(M_hat))]
    M_hat_ALS, counter = aux.alternating_least_squares(Y, M_hat,
            convergence_criteria = convergence_criteria, max_iters = max_iters)
    s_hat_ALS = np.ones(shape = len(s))

    # Let's make the factors have the same scale
    for i in range(len(M_hat_ALS)):
        temp = np.linalg.norm(M_hat_ALS[i], axis = 0)
        s_hat_ALS = s_hat_ALS*temp
        M_hat_ALS[i] = M_hat_ALS[i]/temp
    print(f"ALS took {counter} iterations to converge")
    print(f"How well the direct no rotation recovers F {aux.get_L21_errors(M[0], M_hat[0])}")
    print(f"How well rotation recovers F {aux.get_L21_errors(M[0], M_hat_ALS[0])}")
    #errors = 
    for i in range(len(M)):
        print(f"arcos {aux.get_cos_errors(M[i], M_hat_ALS[i])}")
        row_dict[f"factor_{i}_err"] = np.max(np.arccos(aux.get_idividual_cos_error(M[i], M_hat_ALS[i])))
        #temp = #aux.get_idividual_cos_error(M[i], M_hat_ALS[i])
        #with multiprocessing.Lock():
        #    logging.debug(f"factor {sample_size} cos {temp}")
        
        # = np.mean(temp)
    s_hat_ALS = sorted(s_hat_ALS, reverse = True)
    s = np.array(sorted(s, reverse = True))
    with multiprocessing.Lock():
        logging.debug(f"s{s}, s_hat_ALS {s_hat_ALS}")
    
    row_dict["s_error"] = np.linalg.norm(np.array(s_hat_ALS) - s)/np.sqrt(np.prod(sample_size))
    
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
    dgp_kwargs = config["dgp_kwargs"][0]
    workers = config["workers"]
    



    logging.debug(f"We have {workers} workers available to us.")
    #print(dgp_kwargs)
    dgp_instance = dgp(**dgp_kwargs)
    
    mid = [noise_scale, dgp_instance]
    full_zip = zip_with_arbitrary_middle(range_list, mid, reps)

    logging.debug("Starting the simulation")
    
    with Pool(workers) as pool:
        result = pool.starmap(simulate_3d, full_zip)
    
    err_df = pd.DataFrame(result) 
    logging.debug("The simulation has ended")   
    #print(f"This is the result{result}")
    print(dgp)
    #print(f"kwargs{dgp_kwargs}")

    err_df["sample_size"] = err_df["sample_size"].apply(lambda x: np.prod(x)) # splat the sample size.
    print(f"This is the resulting err df \n{err_df}")

    s_error = err_df[["sample_size","repetition","s_error"]]
    pattern = r'factor_\d+_err'

    matching_columns = err_df.filter(regex=pattern).columns

    df_melted = pd.melt(err_df, id_vars=['sample_size', 'repetition'], 
                    value_vars=matching_columns,
                    var_name='factor', value_name='error')
    #print(df_melted)
    #print(df_melted.query("factor == 'factor_0_err'"))
    sns.boxplot(x='sample_size', y='error', hue='factor', data=df_melted)
    plt.title(f"Boxplot of factor estimation errors")
    plt.savefig(f'./results/simulate_boxplot.png', dpi=300, bbox_inches='tight')
    plt.clf()

    sns.boxplot(x = "sample_size", y = "s_error", data = s_error)
    plt.title("Boxplot plot of s errors")
    plt.savefig("./results/s_error_boxplot.png", dpi = 300, bbox_inches = "tight")
    logging.debug("Everything is done.")