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

"""
Tests to be written: 
1. Basic regression: linear with known answer. 
2. Non-linear: compare with statsmodules in terms of errors and answer.  
2. Edge case: flat line. see errors and warnings. 
3. Wrong input: exponential issue. 
4. Wrong input type: 
"""

def linear_model(X, beta):
    """
    Issue: There is some issue in which autograd interacts with 
    np.exp. Thus, it is important to use autograd.np instead of np. 
    https://stackoverflow.com/questions/67195689/typeerror-loop-of-ufunc-does-not-support-argument-0-of-type-arraybox-which-has?msclkid=401b5a84b51011eca7eb03de2291dd60
    """
    a = beta[0]
    b = beta[1]
    x_mod = b * X[:,0]
    return  a+ b*X[:,0]

def test_linear_case():
    """
    Linear with known answer. 
    """
    X = np.linspace(1,10, 500).reshape((-1, 1)) #X 

    correct_parameter_vector = [1, 2]
    y = correct_parameter_vector[0] + correct_parameter_vector[1] * X + 0.0*np.random.random(X.shape[0]).reshape(-1, 1)
    y = y.reshape((-1, ))

    nlr =  NonLinearRegression(linear_model, 2)
    p0 = np.repeat(1,2).reshape(-1, )

    nlr.fit(X, y, p0)

    numpy.testing.assert_almost_equal(nlr.coef_[0], correct_parameter_vector[0], decimal = 1, err_msg="First parameter is wrong")
    numpy.testing.assert_almost_equal(nlr.coef_[1], correct_parameter_vector[1], decimal = 1, err_msg="Second parameter is wrong")

def test_linear_standard_error():
    """
    """

    X = np.linspace(1,10, 500).reshape((-1, 1)) #X 
    correct_parameter_vector = [1, 2]
    y = correct_parameter_vector[0] + correct_parameter_vector[1] * X + 0.5*np.random.random(X.shape[0]).reshape(-1, 1)
    y = y.reshape((-1, ))

    #Comparing this with regression output from statsmodels. 
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm)
    results = model.fit()
    param_std_error_sm = np.sqrt(np.diag(results.cov_params())) #parameter error

    nlr =  NonLinearRegression(linear_model, 2)
    p0 = np.repeat(1,2).reshape(-1, )
    nlr.fit(X, y, p0)

    parameter_error = np.linalg.norm(nlr.perr - param_std_error_sm)
    assert parameter_error<1e-5, "Coefficients from non-linear regression don't match those from statsmodels."


    #X_sm_test = np.array([1, 2.5])  # "1" refers to the intercept term
    #X_sm_test = sm.add_constant(X_new)
    #pred_sm = results.get_prediction(X_sm_test).summary_frame(alpha=0.05)["mean"] # alpha = significance level for confidence interval
    #pred_sm_ints = results.get_prediction(X_sm_test).summary_frame(alpha=0.05)[["obs_ci_lower",	"obs_ci_upper"]].values # alpha = significance level for confidence interval


    #return a + X[:,0] + np.exp(-b*X[:,0])
    #return a + X[:,0] + b * np.exp(x_mod) #+ np.exp(-b*X[:,0])

    #y = correct_parameter_vector[0] + correct_parameter_vector[1] * X + 0.0*np.random.random(X.shape[0]).reshape(-1, 1)
    #X_new = np.linspace(1, X[-1, 0], 50).reshape(-1, 1)
    #pred = nlr.predict(X_new)
    

    #pred_se = nlr.predict_standard_errors_of_mean(X_new)
    #pred_ints = nlr.get_intervals(X_new, int_type="prediction", percent_interval=95)

if __name__=="__main__":
    test_linear_case()
    test_linear_standard_error()

"""
#plot.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], y, color=plt.cm.Set1(0))
ax.plot(X_new[:, 0], pred, zorder=0.5, color=plt.cm.Set1(1))
ax.fill_between(X_new[:,0], pred_ints[0], pred_ints[1], 
    color=plt.cm.Set1(1), alpha=0.2, zorder=0)


if np.linalg.norm(nlr.perr - param_std_error_sm)<1e-5:
    print("Test successful. Results from non-linear regression match those from statsmodels.")
else:
    print("Test unsuccessful. Results from non-linear regression don't match those from statsmodels.")

#summarize the fit. 
nlr.summarize_fit()

plt.figure()
plt.scatter(pred, pred_sm)

plt.figure()
plt.scatter(pred_ints[0], pred_sm_ints[:,0]-pred_ints[0])
"""



#Make the regression summary table. Find existing methods. 
# Clean up the code and write tests.  
#Add regularization. 
#Account for correlated parameters when considering prediction errors. 
#Write a file to include tests. Write one with complicated exponential models.
#or ODEs, log models. 
#Test 1: types of things
#Test 2: compare with statsmodels. 
#Make figures. 
# %%

