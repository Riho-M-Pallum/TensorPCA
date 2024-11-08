from TensorPCA import auxiliary_functions as aux

class Simulation:
    def __init__(self, start_size, max_size, size_jump, dgp, transformer, noise_scale, estimator=None, predictor=None, n_simulations=100):
        self.dgp = dgp  # DataGeneratingProcess instance
        self.transformer = transformer  # TransformingProcess instance
        self.estimator = estimator  # EstimationProcess instance
        self.predictor = predictor  # PredictionProcess instance
        self.n_simulations = n_simulations
        self.max_size = max_size
        self.noise_scale = noise_scale
        



    def run_simulation(self, factors,  tensor_shape, simulate_estimation=True, simulate_prediction=False):
        """
            Currently takes the number of factors as given. 

        """

        results = []
        for _ in range(self.n_simulations):
            # Step 1: Generate raw data
            M, s = self.dgp.generate_data(self.max_size, factors)


            # Step 2: Transform raw data into tensors of the desired shape
            Y = self.transformer.transform(raw_data = M, scale_params = s, noise_scale = self.noise_scale)
            if simulate_estimation and self.estimator is not None:
                # Step 3: Estimate parameters from transformed tensor data
                unrotated_M_hat, M_hat, s_hat = self.estimator.estimate(factors)
                results.append({"params": estimated_params})

            if simulate_prediction and self.predictor is not None:
                # Step 4: Predict outcomes based on estimated parameters
                unrotated_M_hat, M_hat, s_hat = self.estimator.estimate(factors)


                


        return results

# Example usage:
linear_dgp = LinearDGP(slope=2.0, intercept=1.0, noise_level=0.1)
tensor_transformer = TensorTransform()
linear_estimator = LinearEstimator()
linear_predictor = LinearPredictor()

# Define the simulation
sim = Simulation(dgp=linear_dgp, transformer=tensor_transformer, estimator=linear_estimator, predictor=linear_predictor, n_simulations=10)

# Run the simulation for estimation
estimation_results = sim.run_simulation(n_samples=100, tensor_shape=(-1, 1), simulate_estimation=True, simulate_prediction=False)
print("Estimation Results:", estimation_results)

# Run the simulation for prediction based on previously estimated parameters
prediction_results = sim.run_simulation(n_samples=100, tensor_shape=(-1, 1), simulate_estimation=False, simulate_prediction=True)
print("Prediction Results:", prediction_results)
