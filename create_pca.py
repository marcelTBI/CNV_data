import glob
import dill
import numpy as np
from sklearn.decomposition import PCA

# which chromosomes to normalize
normalize_chrom = 22
# number of components:
components = 20

# inputs and outputs
input_bins = glob.glob('cnv_train_data/pca/*[0-9][0-9][0-9][0-9].npz')
sids = [int(inp.split('/')[-1][:-4]) for inp in input_bins]
output_pca_dill = 'data/pca_c{comps}.pck'.format(comps=components)
output_sids = 'data/samples_c{comps}.txt'.format(comps=components)


def pca_it(bins, components=None):
    """
    Extract PCA components from bins.
    :param bins: 2D-ndarray - (samples, bins) fragment counts per bin
    :param components: int - number of components to keep, None - all of them
    :return: PCA object
    """
    if components is not None:
        pca = PCA(n_components=components, svd_solver="full")
    else:
        pca = PCA(svd_solver="full")

    pca.fit(bins)

    return pca


# load all values into an array
all_bins = []
for i, bin_file in enumerate(input_bins):
    print(i, '/', len(input_bins))
    # load file
    bins_all = np.load(bin_file) ['values']

    # select only autosomal chromosomes
    ind = bins_all['chromosome'] < normalize_chrom
    bins = np.array(bins_all[ind]['bins_loess'])

    # normalize to 1.000.000
    norm_coef = sum(bins) / 1000000
    bins /= norm_coef

    # save to one array
    all_bins.append(bins)

# convert into numpy array
all_bins = np.array(all_bins)

# PCA it!
pca = pca_it(all_bins, components)

# save the pca
with open(output_pca_dill, 'wb') as f:
    dill.dump(pca, f, 2)

# save sids that generated this
np.savetxt(output_sids, np.array(sorted(sids), dtype=int), fmt='%d')
