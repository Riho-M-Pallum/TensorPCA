import numpy as np
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

N = 500

norms_list = []
ar_params = [1, -0.8]

norms_dict = {"rep":250,"T": 250, "val":100}
for T in range(N):
	for rep in range(100):
		ar_process = sm.tsa.arma_generate_sample(ar = ar_params, ma = [1], nsample = T, burnin = 0, scale = 1)  # second value is the lag polynomial for the MA bit of the process
		norms_dict["rep"] = rep
		norms_dict["T"] = T
		norms_dict["val"] = np.linalg.norm(ar_process)
		norms_list.append(norms_dict.copy())


N_values = np.arange(1, N)  # Define N values from 1 to 100

sqrt_N = np.sqrt(N_values)#/(1-ar_params[1]**2)     # Compute sqrt(N)

# Step 3: Plot sqrt(N) on the existing Seaborn plot

df = pd.DataFrame(norms_list)
sns.violinplot(x = "T", y = "val", data = df)
plt.plot(N_values, sqrt_N, color='red', label=r'$\sqrt{N}$', lw=2)
plt.title("Violin plot of s errors")
plt.savefig("tst.png", dpi = 300, bbox_inches = "tight")
    
