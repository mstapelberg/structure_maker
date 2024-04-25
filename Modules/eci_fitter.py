import random
import numpy as np
from monty.serialization import loadfn, dumpfn
from pymatgen.core.structure import Structure
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion, RegressionData

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error


# this code will be for training the ECI terms 
# takes in the wrangler and subspace objects

# Set fit_intercept to False because we already do this using
# the empty cluster.
def eci_fitter(wrangler, subspace):
    """
    Fits a cluster expansion model to the given data using the provided wrangler and subspace.

    Args:
        wrangler (Wrangler): The data wrangler object containing the feature matrix and property vector.
        subspace (Subspace): The subspace object representing the basis functions for the cluster expansion.

    Returns:
        ClusterExpansion: The fitted cluster expansion model.

    """
    estimator = LinearRegression(fit_intercept=False)
    estimator.fit(wrangler.feature_matrix, wrangler.get_property_vector('energy'))
    coefs = estimator.coef_

    train_predictions = np.dot(wrangler.feature_matrix, coefs)

    rmse = mean_squared_error(
        wrangler.get_property_vector('energy'), train_predictions, squared=False
    )
    maxer = max_error(wrangler.get_property_vector('energy'), train_predictions)

    print(f'RMSE {1E3 * rmse} meV/prim')
    print(f'MAX {1E3 * maxer} meV/prim')

    # use the reg data, subspace, and wrangler to create the cluster expansion 
    reg_data = RegressionData.from_sklearn(
        estimator, wrangler.feature_matrix,
        wrangler.get_property_vector('energy')
    )

    expansion = ClusterExpansion(
        subspace, coefficients=coefs, regression_data=reg_data
    )

    structure = random.choice(wrangler.structures)
    prediction = expansion.predict(structure, normalized=True)

    print(
        f'The predicted energy for a structure with composition '
        f'{structure.composition} is {prediction} eV/prim.\n'
    )
    print(f'The fitted coefficients are:\n{expansion.coefs}\n')
    print(f'The effective cluster interactions are:\n{expansion.eci}\n')
    print(expansion)

    return expansion
