import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import cKDTree as KDTree
import geopandas as gpd
import json
from joblib import Parallel, delayed
import clhs as cl

cov_cols = ['be-30y-81', 'be-30y-82', 'be-30y-83', 'be-30y-84', 'be-30y-85', 'be-30y-86', 'be-30y-87', 'be-30y-88',
            'Rad2016K1', 'water-851', '3dem_mag1', '3dem_mag2', '3dem_mag3', 'ceno_euc1', 'Clim_Pre1', 'dem_fill1',
            'Dose_2011', 'Gravity_1', 'MvrtpLL_1', 'national1', 'Potassiu1', 'Rad2016T1', 'Rad2016U1', 'Rad2016U2',
            'relief_e1', 'SagaWET91', 'si_geol11', 'slope_fi1', 'Thorium_1', 'Uranium_1']

major_elements = ['Fe_log10', 'Si0_log', 'Ca0_log', 'K_log', 'MGO_log', 'MNO_log', 'NA20_log', 'Al203_log']


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape
    assert (d == dy)
    print(n, m)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=1e5, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=1e5, p=2)[0]

    is_finite = np.isfinite(r) & np.isfinite(s) & (r > 1e-10) & (s > 1e-10)
    r = r[is_finite]
    s = s[is_finite]
    kldiv = np.sum(rel_entr(r, s))
    print(y.shape[0], kldiv)
    return kldiv


def avg_kl_div(no_samples):
    print(f"==========simulation for N_SAMPLES={int(no_samples)}========")
    div = {}
    divs_list = Parallel(n_jobs=50)(delayed(__parallel_clhs)(no_samples) for r in range(2))
    for i, d in enumerate(divs_list):
        div[i] = d
    return div


def __parallel_clhs(no_samples):
    sampled = cl.clhs(df[cov_cols], int(no_samples), max_iterations=1000)
    clhs_sample = df.iloc[sampled["sample_indices"]]
    return KLdivergence(df[cov_cols], clhs_sample[cov_cols])


if __name__=='__main__':
    # kl_div = defaultdict(dict)
    # kl_div_mean = defaultdict()
    #
    # df = gpd.read_file("/home/sudipta/Documents/nci/Sudipta_data_chem_v1_cleaned_cleaned.shp")
    #
    # # design_space = np.logspace(8, 13.6, 40, base=2)
    # design_space = np.logspace(2, 4, 3, base=2)
    #
    # for d in design_space:
    #     print(int(d))
    #
    # for n_samples in design_space:
    #     kl_div[int(n_samples)] = avg_kl_div(n_samples)
    #     kl_div_mean[int(n_samples)] = np.mean(list(kl_div[int(n_samples)].values()))
    #
    # print(kl_div)
    # print(kl_div_mean)
    # with open("convergence_n.json", 'w') as f:
    #     json.dump(kl_div, f, sort_keys=True, indent=4)
    #
    # with open("convergence_mean_n.json", 'w') as f:
    #     json.dump(kl_div_mean, f, sort_keys=True, indent=4)

    with open("convergence_mean_n.json", 'r') as f:
        kl_div_mean = json.load(f)

    plt.plot(kl_div_mean.keys(), kl_div_mean.values())
    plt.savefig("kl_div_mean.png")
