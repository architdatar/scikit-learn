"""
Generalized Non-Linear Models. 
Wrapper function around the scipy non-linear regression functionality. 
In subsequent versions, will enable support for 1. Constrain optimization. 2. sparse matrices, respectively. 

Things to do: 
1. Make regression summary function inspired by statsmodels. (done)
2. Clean up code to remove references to unnecessary imports and functions. (done)
3. Enable generic support for regularization, constraints, and multioutput. (will do in later iterations)
4. Write additional tests to ensure that the function works properly.  
5. Document the code properly. 
"""

# Author: Archit Datar <architdatar@gmail.com>
# License: BSD 3 clause

import warnings

#import numpy as np

from scipy import optimize
from scipy import sparse
from joblib import Parallel

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin

from scipy.optimize import least_squares
import autograd.numpy as np
from autograd import grad
import scipy
import datetime
from ..metrics import r2_score
import pandas as pd

class NonLinearRegression(RegressorMixin, BaseEstimator): 
    """
    Ordinary least squares non-Linear Regression.
    """

    def __init__(self, 
        model,
        p0_length,
        p0 = None,
        param_names=None,
        model_kwargs_dict = {},
        least_sq_kwargs_dict = {}, #optional arguments for least sq
        normalize_x = False, 
        copy_X = True
        ):
        self.normalize_x = normalize_x
        self.copy_X = copy_X
        self.model  = model
        self.p0_length = p0_length
        self.model_kwargs_dict = model_kwargs_dict
        self.least_sq_kwargs_dict = least_sq_kwargs_dict
        self.p0 = p0
        self.param_names = param_names

    def fit(self, X, y, p0 = None):
        """
        Fit the non-linear model. 
        """

        def _model_residuals(params, X, y):
            return y - self.model(X, params, **self.model_kwargs_dict)

        if p0 is None:
            if self.p0 is None:
                self.p0 = np.repeat(1, self.p0_length)
        else:
            self.p0 = p0

        res_ls = least_squares(_model_residuals, x0=p0, 
             args=(X, y), 
            kwargs=self.model_kwargs_dict, 
            **self.least_sq_kwargs_dict)

        self.coef_ = res_ls.x
        self.jac = res_ls.jac
        self.fitted_lsq_object = res_ls
        self.dfe = res_ls.fun.shape[0] - res_ls.x.shape[0]
        self.RSS = res_ls.fun.T @ res_ls.fun / self.dfe
        self.y = y
        self.X = X

        try:
            self.get_parameter_errors()
        except:
            warnings.warn("Parameter errors could not be estimated. \
            Methods depending on these will not work.")
            pass

    def get_parameter_errors(self):
        """
        Gets the summary for parmeters errors. 
        """
        #Tried to get summary directly from statsmodels, but was unable to do so. Will write a custom
        #summary function. 

        try:
            self.fitted_lsq_object        
        except: 
            raise NameError("Fitted object not found. Please fit the model first and then call the function.")

        self.pcov = self.RSS * np.linalg.inv(self.jac.T @ self.jac)
        self.perr = np.sqrt(np.diag(self.pcov))
        #return pcov

    def summarize_fit(self, side="both", percent_interval=95):
        """
        """
        #Will create this function later, like statsmodels does. 

        n_obs = self.fitted_lsq_object.fun.shape[0]
        p = self.fitted_lsq_object.x.shape[0] 
        r_squared = r2_score(self.y, self.predict(self.X))
        adj_r_squared = 1-(1-r_squared)*(n_obs-1) /(n_obs - p - 1)
        t_stats = (self.coef_ - 0)/self.perr
        t_stats_adj = np.copy(t_stats) #New variable to ensure that we set very high values to a number that doesn't give overflow warning. 
        t_stats_adj[t_stats_adj>50]=50

        print("                 Regression results                ")
        print("Method: Least squares")
        print(f"Date: {datetime.date.today()}")
        print(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"No. observations: {n_obs:.0f}")
        print(f"Df residuals: {self.dfe:.0f}")
        print(f"Df models: {p-1:.0f}")
        print(f"R-squared: {r_squared:.3f}")
        print(f"Adj. R-squared: {adj_r_squared:.3f}")

        df = pd.DataFrame(columns=["parameter", "coef", "std err", "t", "P>|t|", \
        ]) #f"[{alpha:.3f}", f"{1-alpha:.3f}]"
        if self.param_names is None:
            df["parameter"] = [f"p_{num}" for num in range(self.p0_length)]
        else:
            df["parameter"] = self.param_names

        df["coef"] = list(self.coef_)
        df["std err"] = list(self.perr)
        df["t"] = list(t_stats)
        df["P>|t|"] = list(scipy.stats.t.pdf(t_stats_adj, self.dfe))

        if side == "both":
            significance_level = (1 - percent_interval/100) /2 
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            df[f"[{significance_level:.3f}"] = self.coef_ - t_val * self.perr
            df[f"{1-significance_level:.3f}]"] = self.coef_ + t_val * self.perr

        elif side == "lower":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            df[f"[{1-significance_level:.3f}"] = self.coef_ - t_val * self.perr

        elif side == "upper":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            df[f"{significance_level:.3f}]"] = self.coef_ + t_val * self.perr
        return df

    def predict(self, X):
        """
        Predicts the optimal function. 
        """
        return self.model(X, self.coef_, **self.model_kwargs_dict)

    def predict_standard_errors_of_mean(self, X, X_errs = None):
        """
        Provides standard errors of E(y|X_0). 
        Currently, we don't consider the errors in the X values. 
        """

        def _separate_args_wrapper(*X_and_param_args):
            X_ = np.array(X_and_param_args[:X.shape[1]]).reshape((1,X.shape[1]))
            params = np.array(X_and_param_args[X.shape[1]:])
            return self.model(X_, params, **self.model_kwargs_dict)

        grad_matrix = np.zeros((X.shape[0], self.coef_.shape[0]))
        for i in range(X.shape[0]):
            X_and_param_args = list(np.concatenate((X[i, :], self.coef_)))
            for j in range(self.coef_.shape[0]):
                #print(epsilon, parameter)
                grad_matrix[i, j] = grad(_separate_args_wrapper, argnum=X.shape[1]+j)(*X_and_param_args)

        var_confidence_params =  np.diag(grad_matrix @ self.pcov @ grad_matrix.T)                  

        if X_errs is None:
            total_variance = var_confidence_params
            df_ = self.coef_.shape[0] - 1 #model parameters. 

        else:
            if X_errs.shape == X.shape:
                pass
            else:
                raise ValueError("Shape of X_errs and X are not the same.")

            if type(X_errs) == "numpy.ndarray":
                pass
            else:
                raise TypeError("X_errs is not an numpy array.")

            grad_matrix_X = np.zeros(X.shape)
            var_confidence_X = []

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    #print(epsilon, parameter)
                    grad_matrix_X[i, j] = grad(_separate_args_wrapper, argnum=j)(*list(X[i, :]), *self.coef_,)

                var_confidence_X.append(np.diag(grad_matrix_X[i, :] @ np.diag(X_errs[i, :]) @ grad_matrix_X[i, :].T))                  

            var_confidence_X = np.array(var_confidence_X)
            total_variance = var_confidence_params + var_confidence_X
            df_ = self.coef_.shape[0] -1 + X.shape[1]- 1 #model parameters + X values.
            #There is a small problem here. If we only want to consider the errors in certain X values, 
            # we are unnecessarily increasing the degrees of freedom. But this is a niche problem which can be addressed in later versions of the software.  

            #Even for parameters, degrees of freedom are DFE. Same for confidence intervals.

        se_confidence = np.sqrt(total_variance)
        return [se_confidence, df_]
        
    def get_intervals(self, X, X_errs = None, int_type="confidence", side="both", percent_interval=95):
        """
        """
        y_hat = self.predict(X)
        [se_confidence, df_] = self.predict_standard_errors_of_mean(X, X_errs=X_errs)

        if int_type == "confidence":
            se_confidence = se_confidence
        elif int_type == "prediction":
            se_confidence = (se_confidence**2 + self.RSS)**0.5

        if side == "both":
            significance_level = (1 - percent_interval/100) /2 
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            return [y_hat - t_val * se_confidence, y_hat + t_val * se_confidence]
        elif side == "lower":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            return [y_hat - t_val * se_confidence, np.repeat(np.inf, X.shape[0])]
        elif side == "upper":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            return [np.repeat(-np.inf, X.shape[0]), y_hat - t_val * se_confidence]


