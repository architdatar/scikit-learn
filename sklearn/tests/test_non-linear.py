#%%
#import numpy as np
import autograd.numpy as np
import numpy

import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#from sklearn.linear_model import LinearRegression
from sklearn.non_linear_model import NonLinearRegression
import statsmodels.api as sm

np.random.seed(1)

# def linear_model(X, beta):
#     """
#     Issue: There is some issue in which autograd interacts with 
#     np.exp. Thus, it is important to use autograd.np instead of np. 
#     https://stackoverflow.com/questions/67195689/typeerror-loop-of-ufunc-does-not-support-argument-0-of-type-arraybox-which-has?msclkid=401b5a84b51011eca7eb03de2291dd60
#     """
#     a = beta[0]
#     b = beta[1]
#     x_mod = b * X[:,0]
#     return  a+ b*X[:,0]

def linear_model(X, beta):
    """
    """
    return X @ beta

def test_linear_fit():
    """
    Linear with known answer. 
    """
    X = np.linspace(1,10, 500).reshape((-1, 2)) 

    correct_parameter_vector = np.array([0, 0])
    #y = correct_parameter_vector[0] + correct_parameter_vector[1] * X + 0.0*np.random.random(X.shape[0]).reshape(-1, 1)
    y = linear_model(X, correct_parameter_vector) #+ 0.0*np.random.random(X.shape[0]).reshape(-1, 1)
    
    #y = y.reshape((-1, ))

    nlr =  NonLinearRegression(linear_model, 2)
    p0 = np.repeat(1,2).reshape(-1, )

    nlr.fit(X, y, p0)

    numpy.testing.assert_almost_equal(nlr.coef_[0], correct_parameter_vector[0], decimal = 1, err_msg="First parameter is wrong")
    numpy.testing.assert_almost_equal(nlr.coef_[1], correct_parameter_vector[1], decimal = 1, err_msg="Second parameter is wrong")

def test_linear_standard_error():
    """Linear case with standard errors compared with statsmodules. 
    """

    X = np.linspace(1,10, 500).reshape((-1, 2)) #X 
    correct_parameter_vector = [1, 2]
    #y = correct_parameter_vector[0] + correct_parameter_vector[1] * X + 0.5*np.random.random(X.shape[0]).reshape(-1, 1)
    y = linear_model(X, correct_parameter_vector)
    
    y = y.reshape((-1, ))
    #Comparing this with regression output from statsmodels. 
    #X_sm = sm.add_constant(X)
    X_sm = X
    model = sm.OLS(y, X_sm)
    results = model.fit()
    param_std_error_sm = np.sqrt(np.diag(results.cov_params())) #parameter error
    pred_sm_ints = results.get_prediction(X_sm).summary_frame(alpha=0.05)[["obs_ci_lower",	"obs_ci_upper"]].values[:, 0] 

    nlr =  NonLinearRegression(linear_model, 2)
    p0 = np.repeat(1,2).reshape(-1, )
    nlr.fit(X, y, p0)
    pred_ints = nlr.get_intervals(X, int_type="prediction", percent_interval=95, side="both")[0]

    parameter_error = np.linalg.norm(nlr.perr - param_std_error_sm)
    prediction_intervals_error = np.linalg.norm(pred_ints - pred_sm_ints) #Compare for lower bounds of the confidence interval. 

    assert parameter_error<1e-5, "Coefficients from non-linear regression don't match those from statsmodels."
    assert prediction_intervals_error<1e-5, "Prediction intervals from non-linear regression don't match those from statsmodels."

def test_non_linear_fit():
    """Test to check that a non-linear fit is executed properly. 
    """
    def non_linear_model(X, params):
        a, b, c = params
        return a + b * X[:,0] + c * np.exp(X[:,0]) 

    true_params = [1,2,3]
    X = np.linspace(1,10, 500).reshape((-1, 2))
    y = non_linear_model(X, true_params) 

    nlr =  NonLinearRegression(non_linear_model, 3)
    p0 = np.repeat(1,3).reshape(-1, )
    nlr.fit(X, y, p0)

    parameter_error = np.linalg.norm(nlr.coef_ - np.array(true_params))
    assert parameter_error<1e-5, "Coefficients from non-linear regression don't match the true ones for non-linear model."

if __name__=="__main__":
    test_linear_fit()
    test_linear_standard_error()
    test_non_linear_fit()
# %%

"""
#Testubg for 2D X data with 1D Y data and linear model.
#np.random.seed(1000)
#X = np.random.random(size=(500, 20))
X_train = np.array([[1.6464405 , 2.145568  , 1.80829   , 1.6346495 , 1.2709644], [1.9376824 , 1.3127615 , 2.675319  , 2.4868202 , 0.01408643]]).T
#np.random.seed(1)
#y = np.random.random(500)
y_train = np.array([1.1, 1.2, 2.4, 0.7, 2.8])
#nlr = NonLinearRegression(linear_model_for_test, 20)
nlr = NonLinearRegression(linear_model_for_test, 5)
#np.random.seed(1)
p0 = np.random.random(2)
print(f"X shape={X_train.shape}")
print(f"y shape={y_train.shape}")
print(f"p0 shape={p0.shape}")
#nlr.fit(X, y, p0=np.random.random(size=(X.shape[1],)))
#nlr.fit(X, y, p0=np.random.random(20)*0 + 1)
nlr.fit(X_train, y_train, p0=p0)
#plt.scatter(X[:, 0], y)
"""