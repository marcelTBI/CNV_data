import glob
import pickle
import sys
import pandas as pd
import numpy as np

window = 20000

# inputs and outputs
input_bins = glob.glob('cnv_train_data/pca/*[0-9][0-9][0-9][0-9]_pca.npz') + glob.glob('cnv_train_data/*[0-9][0-9][0-9][0-9]_pca.npz')
sids = [int(inp.split('/')[-1][:-8]) for inp in input_bins]
output_sids = 'data/samples_means.txt'
gc_bins_path = 'data/gc_bins_20000.pck'
nn_bins_path = 'data/nn_bins_20000.pck'
means_npy_path = 'data/means.npy'
means_tsv_path = 'data/means.tsv'


def get_bins_from_file(gc, nn):
    """
    Load the GC content and N content of the reference.
    :param gc: str - filename of the GC content pickle
    :param nn: str - filename of the N content pickle
    :return: 2x dict{window:dict{chromsome:ndarray}} - directory of window and chromsomes to GC/N contents of those windows
    """
    with open(gc, "rb") as f:
        if sys.version_info > (3, 0):
            gc_bins_all = pickle.load(f, encoding='latin1')
        else:
            gc_bins_all = pickle.load(f)
    with open(nn, "rb") as f:
        if sys.version_info > (3, 0):
            nn_bins_all = pickle.load(f, encoding='latin1')
        else:
            nn_bins_all = pickle.load(f)

    return gc_bins_all, nn_bins_all


# load data
gc_bins, nn_bins = get_bins_from_file(gc_bins_path, nn_bins_path)

# read data
all_table_PCA = None
all_table_loess = None

# asses normalization factors
autosome_norm = 1000000.0
# good bins: bins greater than 1.0 coverage
x_good_bins = 7107
y_good_bins = 701
auto_good_bins = 137816 - x_good_bins - y_good_bins
x_norm = x_good_bins / auto_good_bins * autosome_norm
y_norm = y_good_bins / auto_good_bins * autosome_norm

# go through every sample
for i, (sid, input_bin) in enumerate(zip(sids, input_bins)):
    print(i, '/', len(input_bins))
    # load sample bins
    sample_bins = np.load(input_bin)['values']
    # get chromosomes
    chroms = np.sort(np.unique(sample_bins['chromosome']))
    # create table:
    table_full = None
    for chr in chroms:
        # create table row
        ind = (sample_bins['chromosome'] == chr)
        bins_loess = sample_bins['bins_loess']
        bins_PCA = sample_bins['bins_PCA']
        table = pd.DataFrame({'chromosome': np.ones(len(bins_loess[ind]), dtype=int) * chr, 'position': np.arange(0, window * len(bins_loess[ind]), window), 'GC_content': gc_bins[chr],
                              'N_content': nn_bins[chr], 'read_count_loess': bins_loess[ind], 'read_count_PCA': bins_PCA[ind]})
        if table_full is not None:
            table_full = pd.concat([table_full, table])
        else:
            table_full = table
    #
    ind_autosomes = table_full['chromosome'] < 22
    ind_x = table_full['chromosome'] == 22
    ind_y = table_full['chromosome'] == 23
    table_full.loc[ind_autosomes, 'read_count_PCA'] = table_full.loc[ind_autosomes, 'read_count_PCA'] / (sum(table_full['read_count_PCA'][ind_autosomes]) / autosome_norm)
    table_full.loc[ind_x, 'read_count_PCA'] = table_full.loc[ind_x, 'read_count_PCA'] / (sum(table_full['read_count_PCA'][ind_x]) / x_norm)
    table_full.loc[ind_y, 'read_count_PCA'] = table_full.loc[ind_y, 'read_count_PCA'] / (sum(table_full['read_count_PCA'][ind_y]) / y_norm)
    table_full.loc[ind_autosomes, 'read_count_loess'] = table_full.loc[ind_autosomes, 'read_count_loess'] / (sum(table_full['read_count_loess'][ind_autosomes]) / autosome_norm)
    table_full.loc[ind_x, 'read_count_loess'] = table_full.loc[ind_x, 'read_count_loess'] / (sum(table_full['read_count_loess'][ind_x]) / x_norm)
    table_full.loc[ind_y, 'read_count_loess'] = table_full.loc[ind_y, 'read_count_loess'] / (sum(table_full['read_count_loess'][ind_y]) / y_norm)
    if all_table_PCA is None:
        all_table_PCA = pd.DataFrame()
        all_table_PCA['position'] = table_full['position']
        all_table_PCA['chromosome'] = table_full['chromosome']
    if all_table_loess is None:
        all_table_loess = pd.DataFrame()
        all_table_loess['position'] = table_full['position']
        all_table_loess['chromosome'] = table_full['chromosome']

    all_table_PCA[sid] = table_full['read_count_PCA']
    all_table_loess[sid] = table_full['read_count_loess']


# save whole table
all_table_PCA = all_table_PCA.set_index(['chromosome', 'position'])
# all_table_PCA.to_csv(output.whole_table_PCA, sep='\t')
all_table_loess = all_table_loess.set_index(['chromosome', 'position'])
# all_table_loess.to_csv(output.whole_table_loess, sep='\t')

# get means and variance:
means_PCA = all_table_PCA.mean(axis=1)
var_PCA = all_table_PCA.var(axis=1, ddof=0)
means_loess = all_table_loess.mean(axis=1)
var_loess = all_table_loess.var(axis=1, ddof=0)
means_pd = pd.DataFrame({'bins_PCA': means_PCA, 'var_PCA': var_PCA, 'bins_loess': means_loess, 'var_loess': var_loess})

# and save them
means_npy = np.empty(len(means_pd), dtype=[('chromosome', 'u1'), ('position', 'u4'), ('bins_PCA', 'f4'),('var_PCA', 'f4'),('bins_loess', 'f4'),('var_loess', 'f4')])
means_npy['chromosome'] = [x[0] for x in means_pd.index.values]
means_npy['position'] = [x[1] for x in means_pd.index.values]
means_npy['bins_PCA'] = means_PCA
means_npy['var_PCA'] = var_PCA
means_npy['bins_loess'] = means_loess
means_npy['var_loess'] = var_loess
print('saving to %s' % means_npy_path)
np.save(means_npy_path, means_npy)
means_pd.to_csv(means_tsv_path, sep='\t', float_format='%.4f')

# save sids that generated this
np.savetxt(output_sids, np.array(sorted(sids), dtype=int), fmt='%d')
