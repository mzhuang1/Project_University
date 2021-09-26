'''
=================================================================================
The raw csv dataset includes 47 fields sepecified in t_fields, k_fields
and other_fields. Each field in t_fields represents the temperature of
a ammeter at a time point, and the same goes to k_fields and v_fields.
In this case, temperature, power and voltage at twelve time points are
sampled for analysis.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data.
=================================================================================
'''

print(__doc__)

from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def do_pca(d, n):
    pca = PCA()
    X = pca.fit_transform(d)
    eigvals = pca.explained_variance_ratio_
    pca = PCA(n_components=n)
    X = pca.fit_transform(d)
    print 'eigen value list', eigvals
    print 'chosen ones', pca.explained_variance_ratio_
    return X, eigvals

def draw(dataset, title):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    for index, data in enumerate(dataset):
        eigvals, df = data
        sing_vals = np.arange(len(eigvals)) + 1
        plt.subplot(2, 2, index * 2 + 1)
        plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.subplot(2, 2, index * 2 + 2)
        plt
        color = '#5cb85c'
        plt.scatter(df['x'], df['y'], c=color)

# temperture t0, t1, ... t1
t_fields = ['t' + str(i) for i in xrange(0, 12)]
# kwh k0, k1, ... k11
k_fields = ['k' + str(i) for i in xrange(0, 12)]
# voltage v0, v1, ... v11
v_fields = ['v' + str(i) for i in xrange(0, 12)]

# other fields
other_fields = ['meter', 'year', 'month', 'day', 'town', 'amiModel']
all_fields = other_fields + t_fields + k_fields + v_fields

csv1000_filename = 'output_1000.csv'
csv10000_filename = 'output_10000.csv'
csv_filename = 'output.csv'
raw_df = pd.read_csv(csv10000_filename)
raw_df1000 = pd.read_csv(csv1000_filename)
raw_df10000 = pd.read_csv(csv10000_filename)


tx1000 = raw_df1000[t_fields].values
tx1000_pca, tx1000_eigvals = do_pca(tx1000, 2)
tx1000_pca_df = pd.DataFrame({'x': tx1000_pca[:, 0], 'y': tx1000_pca[:, 1]})
tx10000 = raw_df10000[t_fields].values
tx10000_pca, tx10000_eigvals = do_pca(tx10000, 2)
tx10000_pca_df = pd.DataFrame({'x': tx10000_pca[:, 0], 'y': tx10000_pca[:, 1]})
draw([(tx1000_eigvals, tx1000_pca_df), (tx10000_eigvals, tx10000_pca_df)], 'PCA - Temperature')

kx1000 = raw_df1000[k_fields].values
kx1000_pca, kx1000_eigvals = do_pca(kx1000, 2)
kx1000_pca_df = pd.DataFrame({'x': kx1000_pca[:, 0], 'y': kx1000_pca[:, 1]})
kx10000 = raw_df10000[k_fields].values
kx10000_pca, kx10000_eigvals = do_pca(kx10000, 2)
kx10000_pca_df = pd.DataFrame({'x': kx10000_pca[:, 0], 'y': kx10000_pca[:, 1]})
draw([(kx1000_eigvals, kx1000_pca_df), (kx10000_eigvals, kx10000_pca_df)], 'PCA - Power')

vx1000 = raw_df1000[t_fields].values
vx1000_pca, vx1000_eigvals = do_pca(vx1000, 2)
vx1000_pca_df = pd.DataFrame({'x': vx1000_pca[:, 0], 'y': vx1000_pca[:, 1]})
vx10000 = raw_df10000[t_fields].values
vx10000_pca, vx10000_eigvals = do_pca(vx10000, 2)
vx10000_pca_df = pd.DataFrame({'x': vx10000_pca[:, 0], 'y': vx10000_pca[:, 1]})
draw([(vx1000_eigvals, vx1000_pca_df), (vx10000_eigvals, vx10000_pca_df)], 'PCA - Voltage')

x1000 = raw_df1000[t_fields].values
x1000_pca, x1000_eigvals = do_pca(x1000, 2)
x1000_pca_df = pd.DataFrame({'x': x1000_pca[:, 0], 'y': x1000_pca[:, 1]})
x10000 = raw_df10000[t_fields].values
x10000_pca, x10000_eigvals = do_pca(x10000, 2)
x10000_pca_df = pd.DataFrame({'x': x10000_pca[:, 0], 'y': x10000_pca[:, 1]})
draw([(x1000_eigvals, x1000_pca_df), (x10000_eigvals, x10000_pca_df)], 'PCA - Temperature, Power and Voltage')