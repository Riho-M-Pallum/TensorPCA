from TensorPCA import estimation_classes, dgp_classes, transformation_classes
import numpy as np
import statsmodels.tsa.api as sm
import TensorPCA.auxiliary_functions as aux


""" 
The way that I carry out prediction is that I generate a model of say size (I,N,T), I want to predict new entries in the T direction,
 so for finding the loadings I only use floor( T*x ), x in (0,1) observations. With the final 1-T*x observations serving as the set that I wish
 to predict over. Prediction requires two steps. First simulating the path of the factors using an estimated AR model, then plugging those into the 
 equation for y at a future date.	
 Naturally if y_{t} = Lambda1 * F_{t} + Lambda_2 * F_{t-1} + e_{t}
 Then E[y_{t+1}] = Lambda1 E[F_{t+1}] + Lambda_2 F_{t} + E[e_{t+1}] = Lambda1 AF_{t} + Lambda2 F_{t}

 In the same way 
 E[y_{t+h}] = Lambda1E[f_{t+h}] + lambda2 E[F_{t+h-1}] = Lambda1 A^{h}f_{t} + Lambda2 A^{h-1}f_{t}?


When we do prediction, we want to compare multiple different appraoches. The approaches implemented are


1) TensorPCA factor model
2) Unfold + VAR 
3) VAR on true factor as a baseline predictability meaure



In this script I will follow way 2 for no particular reason other than the fact that I worked out the rotation for F first.

For VAR I estimate country by country and calculate the average country error.
For tensorPCA I take that we know the number of factors.

"""



def fit_var(data):
        model = sm.VAR(data)
        try:
            lag_order = model.select_order(maxlags = 5)
        except:
            lag_order = 1 
            #print(f"Model lag order {lag_order}")
            fitted_model = model.fit(lag_order) 
            return fitted_model, lag_order
        print(f"Model lag order {lag_order}")
        fitted_model = model.fit(lag_order.aic) 
        return fitted_model, lag_order.aic

def predict(data, steps):
	# So for some reason sometimes on the rotated data we get that the VAR is just a constant
    var_model, lag_order = fit_var(data)
    print(lag_order)
    forecast = var_model.forecast(data[-lag_order:,:], steps = steps)
    return forecast
    
# _______________ SETTING EVERYTHING UP ___________________
seed = 57# np.random.randint(0,10000) # 2024, 4258, 5356 is good 6124, 57 is bad 
print(f"Seed: {seed}")
T = 4*15 # Time periods, 4 quarters over 15 years
I = 5 # Say we have 5 countries
J = 8 # And 8 time series for each country

factors = 2
lags = 0

max_shape = (T, I, J)

ar_coefficients = np.array([[0.8, 0.5]])

AR_DGP = dgp_classes.base_F_AR_DGP(shape = max_shape, factors = factors, lags = lags, ar_coefficients = ar_coefficients,
	ar_scale = 0.001)
tensor_transformer = transformation_classes.TensorTransform()
tensorpca_estimator = estimation_classes.TensorPCA(factors*(lags+1))

# Generate the factors and scale parameters
M, s = AR_DGP.generate_data(seed)
print(f"Size of M[0] {np.shape(M[0])}, size of M[1], {np.shape(M[1])}, size of M2 {np.shape(M[2])}" )
# Generate the data
Y = tensor_transformer.transform(raw_data = M, scale_params = s, noise_scale = 0)
print(f"The shape of Y {np.shape(Y)}")
# Break the data down into observed and unobserved
observed_T = int(T*0.8)
observed_Y = Y[:observed_T, :, :]
unobserved_Y = Y[observed_T:, :, :]

baseline_prediction = predict(data = M[0], steps = T-observed_T)
temp = [baseline_prediction, M[1], M[2]]
baseline_predicted_Y = tensor_transformer.transform(raw_data = temp, scale_params = s, noise_scale = 0)
basesline_prediction_error = (baseline_predicted_Y - Y[observed_T:, :, :])**2

baseline_average_prediction_error_by_horizon = np.mean(np.sum(basesline_prediction_error, axis = 2), axis = 1)
print(f"Baseline average prediction error by horizon {baseline_average_prediction_error_by_horizon}")
print(f"The shape of observed_Y {np.shape(observed_Y)}")
print(f"Steps to predict {T - observed_T}")

# _______________ Now for the estimation and prediction _________________

# First we do VAR
countries = [Y[:,i,:] for i in range(np.shape(Y)[1])]
Var_prediction_errors = []
for country in countries:	
	#print(f"Shape of country {np.shape(country)}")
	prediction = predict(data = country[:observed_T, :], steps = T-observed_T)
	#print(f"Shape of prediction {np.shape(prediction)}")
	prediction_error = np.sum((prediction - country[observed_T:, :])**2, axis = 1)
	#print(f"The prediction error by country {prediction_error}")
	Var_prediction_errors += [prediction_error]
	#print(prediction)

average_var_pred_error_by_horizon = np.mean(Var_prediction_errors, axis = 0)

print(f"average_var_pred_error_by_horizon{average_var_pred_error_by_horizon/baseline_average_prediction_error_by_horizon}")

# Now to do tensorPCA

unrotated_data, unrotated_s, rotated_data, s_hat = tensorpca_estimator.estimate(Y[:observed_T, :, :])
print(f"Whether the unrotated recovers the span {aux.get_cos_errors(M[0][:observed_T,:], unrotated_data[0])},\
	second {aux.get_cos_errors(M[1], unrotated_data[1])} third {aux.get_cos_errors(M[2], unrotated_data[2])}")
print(f"Whether we recover the span F {aux.get_cos_errors(M[0][:observed_T,:], rotated_data[0])},\
	second {aux.get_cos_errors(M[1], rotated_data[1])} third {aux.get_cos_errors(M[2], rotated_data[2])}")

print(f"This is s_hat {s}")
print(f"rotated data shape {np.shape(rotated_data[0])}, {np.shape(rotated_data[1])}, {np.shape(rotated_data[2])}")
F_hat = rotated_data[0]
print(f"This is the observed F_hat {F_hat}")
F_hat_prediction = predict(data = F_hat, steps = T-observed_T)
print(f"This is the f hat prediction shape, should be (T-observed_T, factors*(lags+1)) {np.shape(F_hat_prediction)}")
unobserved_rotated_data = [F_hat_prediction, rotated_data[1], rotated_data[2]]
Y_forecast = tensor_transformer.transform(raw_data = unobserved_rotated_data, scale_params = s_hat, noise_scale = 0)

prediction_error = (Y_forecast - Y[observed_T:, :, :])**2

average_PCA_prediction_error_by_horizon = np.mean(np.sum(prediction_error, axis = 2), axis = 1)
print(f"Average tensorPCA prediction error by horizon {average_PCA_prediction_error_by_horizon/baseline_average_prediction_error_by_horizon}")

print(f"Var prediction error divided by tensorPCA prediction error First PCA error by increasing horizon\n{average_var_pred_error_by_horizon /average_PCA_prediction_error_by_horizon}")

#print(f"Error between true y and estimated Y as a fraction of Frobenius norm of Y{np.sum((Y_estimated - Y)**2)/np.sum(Y**2)}")

"""
I compare the following approaches
1) TensorPCA factor model
2) Unfold + VAR 
3) Unfold + factor model
"""



#print(f"Y estimated {Y_estimated}")

