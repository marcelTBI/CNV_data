# CNV_data
Data for training of CNV caller

The training is two fold:
1. PCA components are trained from a subset of the data (samples with higher read count (>10M)). These samples reside in the "pca" directory.
2. Means of bin counts are trained from PCA normalized bincounts from all samples (from those in the "pca" dir, too). 
More info in the article. 

Each file is a single comressed numpy array with following dtype in the only 'values' key:
```python 
dtype=[('chromosome', 'i1'), ('bins_loess', '<f2'), ('bins_PCA', '<f2')] 
```
and each array has length of 154794, since the used bin size is 20,000 (hg19 reference). The "columns" are:
1. chromosome - chromosome number (0-based, 22 for X, 23 for Y)
2. bins_loess - loess normalized bin counts (bin size is 20,000) - used for training of PCA normalization
2. bins_PCA - loess and subsequently PCA normalized bin counts (bin size is 20,000) - used for training of bin count means

Thus, the loess corrected read count for the chromosome 1 can be obtained as:
```python
import numpy as np
sample_npy = np.load('/path/to/sample/values.npy')['values']
read_count_chr1 = sum(sample_npy[sample_npy['chromosome']==0]['bins_loess'])
```
