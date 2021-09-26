'''
The raw csv dataset includes 47 fields sepecified in t_fields, k_fields
and other_fields. Each field in t_fields represents the temperature of
a ammeter at a time point, and the same goes to k_fields and v_fields.
In this case, temperature, power and voltage at twelve time points are
sampled for analysis.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data.

Fig <PCA - Temperature>, we plot the tempature samples on the 2 first principal components.
Fig <PCA - Power>, we plot the power samples on the 2 first principal components.
Fig <PCA - Voltage>, we plot the voltage samples on the 2 first principal components.
Fig <PCA - Temperature, Power and Voltage>. Both temperature, power and voltage at certain time point are sampled for plotting.
'''

print(__doc__)

from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def do_pca(d, n):
    pca = PCA(n_components=n)
    X = pca.fit_transform(d)
    print 'pca.explained_variance_ratio_', pca.explained_variance_ratio_
    return X

def draw(df, title):
    color = '#5cb85c'
    plt.figure(figsize=(400, 400))
    plt.title(title)
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
raw_df = pd.read_csv(csv1000_filename)

tx = raw_df[t_fields].values
tx_pca = do_pca(tx, 2)
tx_pca_df = pd.DataFrame({'x': tx_pca[:, 0], 'y': tx_pca[:, 1]})
draw(tx_pca_df, 'PCA - Tempature')

kx = raw_df[k_fields].values
kx_pca = do_pca(kx, 2)
kx_pca_df = pd.DataFrame({'x': kx_pca[:, 0], 'y': kx_pca[:, 1]})
draw(kx_pca_df, 'PCA - Power')

vx = raw_df[v_fields].values
vx_pca = do_pca(vx, 2)
vx_pca_df = pd.DataFrame({'x': vx_pca[:, 0], 'y': vx_pca[:, 1]})
draw(vx_pca_df, 'PCA - Voltage')

x = raw_df[['t0', 'k0', 'v0']].values
x_pca = do_pca(x, 2)
x_pca_df = pd.DataFrame({'x': x_pca[:, 0], 'y': x_pca[:, 1]})
draw(x_pca_df, 'PCA - Tempature, Power and Voltage')