import glob
import pickle
import sys
import numpy as np
import numpy.lib.recfunctions as np_recfunctions

# assign constants
components = 15
window = 20000
normalize_chrom = 22

# input and output
input_bins = glob.glob('cnv_train_data/pca/*[0-9][0-9][0-9][0-9].npz') + glob.glob('cnv_train_data/*[0-9][0-9][0-9][0-9].npz')
input_pca = 'data/pca_c20.pck'
output_bins = [filepath[:-4] + '_pca.npz' for filepath in input_bins]


def load_pca(pca_dill):
    """
    Loads stored pca components.
    :param pca_dill: str - file with pca components.
    :return: pca component object
    """
    with open(pca_dill, "rb") as f:
        if sys.version_info > (3, 0):
            pca = pickle.load(f, encoding='latin1')
        else:
            pca = pickle.load(f)

    return pca


def normalize_by_pca(pca, bins, max_comps=None, min_bin_value=0.001):
    """
    Normalize the bin count by PCA normalization.
    :param pca: PCA object - contains all data for normalization
    :type bins: np.ndarray
    :param bins: 1D-ndarray - fragment counts per bin for a single sample
    :param max_comps: int - number of components to remove
    :param min_bin_value: float - minimal bin value to keep
    :return: 1D-ndarray - fragment counts per bin without first max_comps components
    """
    # check max_comps
    if max_comps is None:
        max_comps = pca.components_.shape[0]

    # check if there is extracted mean and transform to component space
    if pca.mean_ is not None:
        bins_transf = np.dot((bins - min_bin_value) - pca.mean_, pca.components_[:max_comps, :].T)
    else:
        bins_transf = np.dot(bins - min_bin_value, pca.components_[:max_comps, :].T)

    # transform back to bin space
    bins_transf = np.dot(bins_transf, pca.components_[:max_comps, :])

    # subtract and return
    bins = bins - bins_transf
    bins += min_bin_value
    return bins


def pca_sample(bins_all, window, components, pca):
    """
    PCA normalization. Takes values and creates a new column "PCA_weights" with PCA normalized values.
    :param bins_all: structured ndarray - npy file with information about reads per bin
    :param window: int - window size
    :param components: int - how many components to remove
    :param pca: pickle - dill/pickle with pca components created with pca normalization script
    :return: structured ndarray - structured array about bins
    """
    # create bins
    ind = bins_all['chromosome'] < normalize_chrom
    bins = np.array(bins_all['bins_loess'][ind])
    bins_all['bins_PCA'] = bins_all['bins_loess']

    # normalize to 1.000.000
    norm_coef = sum(bins) / 1000000
    bins /= norm_coef

    # normalize by pca
    bins_normalized = normalize_by_pca(pca, bins, max_comps=components, min_bin_value=0.001)

    # return
    bins_all['bins_PCA'][ind] = bins_normalized.astype('f2')
    return bins_all


def add_pca_column(input_bins, output_bins, pca_object):
    """
    Read input_bins, add the PCA column and return it as output_bins.
    :param input_bins: str - input bin numpy file
    :param output_bins: str - output bin numpy file
    :param pca_object: pickle - pickle of pca object
    """
    # load data
    sample = np.load(input_bins)['values']

    # run pca
    bins = pca_sample(sample, window, components, pca_object)

    # save bins for cnv
    np.savez_compressed(output_bins, values=bins)


# load PCA pickle
pca = load_pca(input_pca)

for i, (input_bin, output_bin) in enumerate(zip(input_bins, output_bins)):
    print(i, '/', len(input_bins), input_bin, output_bin)
    add_pca_column(input_bin, output_bin, pca)
