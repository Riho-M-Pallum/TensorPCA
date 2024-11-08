class PredictionProcess:

    def __init__(self, params):
        self.params = params

    def fit_var(self):
        model = sm.VAR(self.params)
        try:
            lag_order = model.select_order(maxlags = 5)
        except:
            lag_order = 1 
            print(f"Model lag order {lag_order}")
            fitted_model = model.fit(lag_order) 
            return fitted_model, lag_order
        print(f"Model lag order {lag_order}")
        fitted_model = model.fit(lag_order.aic) 
        return fitted_model, lag_order.aic

    def predict(self, steps, components):
        # Assume estimated_params = (weight, bias)
        F_var_model, lag_order = fit_var(self.params)
        forecast = F_var_model.forecast(Y[-lag_order:,:], steps = steps)
        return forecast
    

