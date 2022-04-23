"""
Generalized Non-Linear Models. Something else.
"""

# Author: Archit Datar <architdatar@gmail.com>
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod
import numbers
import warnings

#import numpy as np

import scipy.sparse as sp
from scipy import linalg
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from joblib import Parallel

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
from ..preprocessing._data import _is_constant_feature
from ..utils import check_array
from ..utils.validation import FLOAT_DTYPES
from ..utils import check_random_state
from ..utils.extmath import safe_sparse_dot
from ..utils.extmath import _incremental_mean_and_var
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..utils._seq_dataset import ArrayDataset32, CSRDataset32
from ..utils._seq_dataset import ArrayDataset64, CSRDataset64
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils.fixes import delayed

from scipy.optimize import least_squares
import autograd.numpy as np
from autograd import grad
import scipy

# TODO: bayesian_ridge_regression and bayesian_regression_ard
# should be squashed into its respective objects.

SPARSE_INTERCEPT_DECAY = 0.01
# For sparse data intercept updates are scaled by this decay factor to avoid
# intercept oscillation.


# FIXME in 1.2: parameter 'normalize' should be removed from linear models
# in cases where now normalize=False. The default value of 'normalize' should
# be changed to False in linear models where now normalize=True
def _deprecate_normalize(normalize, default, estimator_name):
    """Normalize is to be deprecated from linear models and a use of
    a pipeline with a StandardScaler is to be recommended instead.
    Here the appropriate message is selected to be displayed to the user
    depending on the default normalize value (as it varies between the linear
    models and normalize value selected by the user).

    Parameters
    ----------
    normalize : bool,
        normalize value passed by the user

    default : bool,
        default normalize value used by the estimator

    estimator_name : str
        name of the linear estimator which calls this function.
        The name will be used for writing the deprecation warnings

    Returns
    -------
    normalize : bool,
        normalize value which should further be used by the estimator at this
        stage of the depreciation process

    Notes
    -----
    This function should be updated in 1.2 depending on the value of
    `normalize`:
    - True, warning: `normalize` was deprecated in 1.2 and will be removed in
      1.4. Suggest to use pipeline instead.
    - False, `normalize` was deprecated in 1.2 and it will be removed in 1.4.
      Leave normalize to its default value.
    - `deprecated` - this should only be possible with default == False as from
      1.2 `normalize` in all the linear models should be either removed or the
      default should be set to False.
    This function should be completely removed in 1.4.
    """

    if normalize not in [True, False, "deprecated"]:
        raise ValueError(
            "Leave 'normalize' to its default value or set it to True or False"
        )

    if normalize == "deprecated":
        _normalize = default
    else:
        _normalize = normalize

    pipeline_msg = (
        "If you wish to scale the data, use Pipeline with a StandardScaler "
        "in a preprocessing stage. To reproduce the previous behavior:\n\n"
        "from sklearn.pipeline import make_pipeline\n\n"
        "model = make_pipeline(StandardScaler(with_mean=False), "
        f"{estimator_name}())\n\n"
        "If you wish to pass a sample_weight parameter, you need to pass it "
        "as a fit parameter to each step of the pipeline as follows:\n\n"
        "kwargs = {s[0] + '__sample_weight': sample_weight for s "
        "in model.steps}\n"
        "model.fit(X, y, **kwargs)\n\n"
    )

    if estimator_name == "Ridge" or estimator_name == "RidgeClassifier":
        alpha_msg = "Set parameter alpha to: original_alpha * n_samples. "
    elif "Lasso" in estimator_name:
        alpha_msg = "Set parameter alpha to: original_alpha * np.sqrt(n_samples). "
    elif "ElasticNet" in estimator_name:
        alpha_msg = (
            "Set parameter alpha to original_alpha * np.sqrt(n_samples) if "
            "l1_ratio is 1, and to original_alpha * n_samples if l1_ratio is "
            "0. For other values of l1_ratio, no analytic formula is "
            "available."
        )
    elif estimator_name in ("RidgeCV", "RidgeClassifierCV", "_RidgeGCV"):
        alpha_msg = "Set parameter alphas to: original_alphas * n_samples. "
    else:
        alpha_msg = ""

    if default and normalize == "deprecated":
        warnings.warn(
            "The default of 'normalize' will be set to False in version 1.2 "
            "and deprecated in version 1.4.\n"
            + pipeline_msg
            + alpha_msg,
            FutureWarning,
        )
    elif normalize != "deprecated" and normalize and not default:
        warnings.warn(
            "'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n"
            + pipeline_msg
            + alpha_msg,
            FutureWarning,
        )
    elif not normalize and not default:
        warnings.warn(
            "'normalize' was deprecated in version 1.0 and will be "
            "removed in 1.2. "
            "Please leave the normalize parameter to its default value to "
            "silence this warning. The default behavior of this estimator "
            "is to not do any normalization. If normalization is needed "
            "please use sklearn.preprocessing.StandardScaler instead.",
            FutureWarning,
        )

    return _normalize


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    check_input=True,
):
    """Center and scale data.

    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output

        X = (X - X_offset) / X_scale

    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    fit_intercept=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).

    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype

    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
        If normalize is True, then X_out is rescaled (dense and sparse case)
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Likely performed inplace on input y.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        The standard deviation per column of input X.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(
                    X,
                    last_mean=0.0,
                    last_variance=0.0,
                    last_sample_count=0.0,
                    sample_weight=sample_weight,
                )
            else:
                X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset

        if normalize:
            X_var = X_var.astype(X.dtype, copy=False)
            # Detect constant features on the computed variance, before taking
            # the np.sqrt. Otherwise constant features cannot be detected with
            # sample weights.
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            X_scale = np.sqrt(X_var, out=X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


# TODO: _rescale_data should be factored into _preprocess_data.
# Currently, the fact that sag implements its own way to deal with
# sample_weight makes the refactoring tricky.


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight because

        (y - X w)' S (y - X w)

    with S = diag(sample_weight) becomes

        ||y_rescaled - X_rescaled w||_2^2

    when setting

        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}

    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y, sample_weight_sqrt


class LinearModel(RegressorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0

    def _more_tags(self):
        return {"requires_y": True}



class SparseCoefMixin:
    """Mixin for converting coef_ to and from CSR format.

    L1-regularizing estimators should inherit this.
    """

    def densify(self):
        """
        Convert coefficient matrix to dense array format.

        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self
            Fitted estimator.
        """
        msg = "Estimator, %(name)s, must be fitted before densifying."
        check_is_fitted(self, msg=msg)
        if sp.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        return self

    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.

        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.

        The ``intercept_`` member is not converted.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.

        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.
        """
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        self.coef_ = sp.csr_matrix(self.coef_)
        return self


class NonLinearRegression(RegressorMixin, BaseEstimator): #LinearModel
    """
    Ordinary least squares non-Linear Regression.


    """

    def __init__(self, 
        model,
        p0_length,
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


    def fit(self, X, y, p0 = None):
        """
        Fit the non-linear model. 
        """

        def _model_residuals(params, X, y):
            return y - self.model(X, params, **self.model_kwargs_dict)

        if p0 is None:
            p0 = np.repeat(1, self.p0_length)

        res_ls = least_squares(_model_residuals, x0=p0, 
             args=(X, y), 
            kwargs=self.model_kwargs_dict, 
            **self.least_sq_kwargs_dict)

        self.coef_ = res_ls.x
        self.jac = res_ls.jac
        self.fitted_lsq_object = res_ls
        self.dfe = res_ls.fun.shape[0] - res_ls.x.shape[0]
        self.RSS = res_ls.fun.T @ res_ls.fun / self.dfe

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
            raise NameError("Model wasn't fit successfully. Please fit the model first and then call the function.")

        self.pcov = self.RSS * np.linalg.inv(self.jac.T @ self.jac)
        self.perr = np.sqrt(np.diag(self.pcov))
        #return pcov

    def summarize_fit(self):
        """
        """
        #Will create this function later, like statsmodels does. 

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
        elif side == "upper":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            return [y_hat - t_val * se_confidence, np.repeat(np.inf, X.shape[0])]
        elif side == "upper":
            significance_level = percent_interval/100
            t_val = scipy.stats.t.ppf(q=1-significance_level, df=self.dfe)
            return [np.repeat(-np.inf, X.shape[0]), y_hat - t_val * se_confidence]

'''
def _check_precomputed_gram_matrix(
    X, precompute, X_offset, X_scale, rtol=1e-7, atol=1e-5
):
    """Computes a single element of the gram matrix and compares it to
    the corresponding element of the user supplied gram matrix.

    If the values do not match a ValueError will be thrown.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data array.

    precompute : array-like of shape (n_features, n_features)
        User-supplied gram matrix.

    X_offset : ndarray of shape (n_features,)
        Array of feature means used to center design matrix.

    X_scale : ndarray of shape (n_features,)
        Array of feature scale factors used to normalize design matrix.

    rtol : float, default=1e-7
        Relative tolerance; see numpy.allclose.

    atol : float, default=1e-5
        absolute tolerance; see :func`numpy.allclose`. Note that the default
        here is more tolerant than the default for
        :func:`numpy.testing.assert_allclose`, where `atol=0`.

    Raises
    ------
    ValueError
        Raised when the provided Gram matrix is not consistent.
    """

    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


def _pre_fit(
    X,
    y,
    Xy,
    precompute,
    normalize,
    fit_intercept,
    copy,
    check_input=True,
    sample_weight=None,
):
    """Function used at beginning of fit in linear models with L1 or L0 penalty.

    This function applies _preprocess_data and additionally computes the gram matrix
    `precompute` as needed as well as `Xy`.

    Parameters
    ----------
    order : 'F', 'C' or None, default=None
        Whether X and y will be forced to be fortran or c-style. Only relevant
        if sample_weight is not None.
    """
    n_samples, n_features = X.shape

    if sparse.isspmatrix(X):
        # copy is not needed here as X is not modified inplace when X is sparse
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=False,
            check_input=check_input,
            sample_weight=sample_weight,
        )
    else:
        # copy was done in fit if necessary
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # Rescale only in dense case. Sparse cd solver directly deals with
        # sample_weight.
        if sample_weight is not None:
            # This triggers copies anyway.
            X, y, _ = _rescale_data(X, y, sample_weight=sample_weight)

    # FIXME: 'normalize' to be removed in 1.2
    if hasattr(precompute, "__array__"):
        if (
            fit_intercept
            and not np.allclose(X_offset, np.zeros(n_features))
            or normalize
            and not np.allclose(X_scale, np.ones(n_features))
        ):
            warnings.warn(
                "Gram matrix was provided but X was centered to fit "
                "intercept, or X was normalized : recomputing Gram matrix.",
                UserWarning,
            )
            # recompute Gram
            precompute = "auto"
            Xy = None
        elif check_input:
            # If we're going to use the user's precomputed gram matrix, we
            # do a quick check to make sure its not totally bogus.
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)

    # precompute if n_samples > n_features
    if isinstance(precompute, str) and precompute == "auto":
        precompute = n_samples > n_features

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order="C")
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, "__array__"):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, "__array__") and Xy is None:
        common_dtype = np.find_common_type([X.dtype, y.dtype], [])
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order="C")
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order="F")
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy
'''
