# File: nbaPlayer_shotRate_lgcp_new2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

import plot_court
import sklearn.model_selection
from sklearn.decomposition import NMF


# Load data
full_DatFrame = pd.read_csv('../dat/joined_shots_2013.csv')
df = pd.DataFrame(full_DatFrame, 
                  columns = ['PLAYER_ID.1', 'PLAYER_NAME', 
                             'MATCHUP', 'LOCATION', 'TEAM_ID', 
                             'SHOT_DISTANCE', 
                             'PTS_TYPE', 'LOC_X', 'LOC_Y', 
                             'ACTION_TYPE', 'SHOT_TYPE',
                             'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG'])

# Add shooter's team column
teamID_dict = plot_court.out_teamsDict()
def out_teamAbbrev(teamID):
    teamID_dict = plot_court.out_teamsDict()
    return teamID_dict[teamID]['abbreviation']
df['TEAM_ABBREV'] = pd.Series(map(out_teamAbbrev, df.TEAM_ID), index=df.index)


#%%

# Create bins
bins, binRange = ([25,18], [[-250,250], [-47.5,312.5]])
hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df.LOC_X, df.LOC_Y, 
                                                                    df.SHOT_MADE_FLAG,
                                                                    statistic='count',
                                                                    bins=bins, 
                                                                    range=binRange)
XX, YY = np.meshgrid(xedges, yedges)
binX_flat = XX.T[:-1,:-1].flatten()
binY_flat = YY.T[:-1,:-1].flatten()
binXY = np.column_stack((binX_flat.T, binY_flat.T))

dist_matrix = scipy.spatial.distance_matrix(binXY, binXY)
def cov_func(dist_matrix, sigma2, phi2):
    return sigma2 * np.exp( -(dist_matrix**2) / (2 * phi2) )

phi2 = 25.**2
sigma2 = 1e3

cov_K = cov_func(dist_matrix, sigma2, phi2)
sign, logdet_cov_K = np.linalg.slogdet(cov_K)
inv_cov_K = np.linalg.inv(cov_K)
nbins = np.prod(bins)

#%%

num_players = 300
top_players_shotNum = df.PLAYER_NAME.value_counts()[:num_players]
top_players_nameList = top_players_shotNum.index.tolist()


train_df = {}
test_df = {}
randSeed = 348098
for i, player in enumerate(set(top_players_nameList)):  
    temp = df[df.PLAYER_NAME == player]
    train_df[player], test_df[player] = sklearn.model_selection.train_test_split(temp, test_size=0.2, random_state=randSeed)

    
player_shotHist_train = {}
for i, player in enumerate(set(top_players_nameList)):  
    temp = train_df[player]
    hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                        temp.SHOT_MADE_FLAG,
                                                                        statistic='count',
                                                                        bins=bins, 
                                                                        range=binRange)
    player_shotHist_train[player] = hist2d.flatten()
    

player_shotHist_test = {}
for i, player in enumerate(set(top_players_nameList)):  
    temp = test_df[player]
    hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                        temp.SHOT_MADE_FLAG,
                                                                        statistic='count',
                                                                        bins=bins, 
                                                                        range=binRange)
    player_shotHist_test[player] = hist2d.flatten()
    

#%%
# Likelihood-max for shot count distribution (Log Gaussian Cox Process)

def ln_prior(nbins, zn_v, logdet_cov_K, inv_cov_K):
    part1 = - (nbins/2.) * np.log(2 * np.pi) - (0.5 * logdet_cov_K)
    part2 = -0.5 * np.dot(zn_v, np.dot(inv_cov_K, zn_v))
    return part1 + part2

def lambdaN_func(z0, zn_v):
    return np.exp(z0 + zn_v)

def ln_lambdaN_func(z0, zn_v):
    return z0 + zn_v

def ln_factorial(n):
    temp = scipy.misc.factorial(n)
    return np.log(temp)

def ln_likelihood(z0, zn_v, Xn_v):
    part1 = -lambdaN_func(z0, zn_v)
    part2 = Xn_v * ln_lambdaN_func(z0, zn_v)
    part3 = -ln_factorial(Xn_v)
    return np.sum(part1 + part2 + part3)

def ln_postprob(z, Xn_v, logdet_cov_K, inv_cov_K, nbins):
    z0 = z[0]
    zn_v = z[1:]
    return ln_prior(nbins, zn_v, logdet_cov_K, inv_cov_K) + ln_likelihood(z0, zn_v, Xn_v)
    
#%%
    
import matplotlib.colors as colors

def plot_player_x6(player, norm_lambdaN_v, histN_v, norm_lambdaN_v_NMF, add_title=['','','']):
    LAMBDA_v = np.reshape(norm_lambdaN_v, bins)
    HIST_v = np.reshape(histN_v, bins)
    NMF_v = np.reshape(norm_lambdaN_v_NMF, bins)

    extent = np.min(xedges), np.max(xedges), np.max(yedges), np.min(yedges)
    
    plt.figure(figsize=(9,10))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        if (int(i/2) == 0):
            what_to_plot = HIST_v.T
        elif (int(i/2) == 1):
            what_to_plot = LAMBDA_v.T
        else:
            what_to_plot = NMF_v.T
            
        if (i%2 == 0):
            plt.imshow(np.ma.masked_where(what_to_plot == 0, what_to_plot), cmap=plt.cm.magma_r, alpha=.85, extent=extent)
        else:
            plt.imshow(what_to_plot, cmap=plt.cm.magma_r, 
                       norm=colors.LogNorm(vmin=1e-4, vmax=1e-1),
                       alpha=.85, extent=extent)
        
        plot_court.draw_court(outer_lines=True, lw=1.5)

        plt.xlim(-300,300)
        plt.ylim(-100,500)
        if (i == 0):
            plt.title('%s: %s'%(player, add_title[int(i/2)]), fontsize=15)
        else:
            plt.title('%s: %s \n emphasizing shots away from basket'%(player, add_title[int(i/2)]), fontsize=15)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('LGCPmodel_%s.png'%(player))
    
#%%

player = 'Jimmy Butler'
player_index = top_players_nameList.index(player)

Xn_v = player_shotHist_train[player]
histN_v = Xn_v/np.sum(Xn_v)


z0_guess = np.log(np.mean(Xn_v))
zn_v_guess = 0. * Xn_v
z_guess = np.append(z0_guess, zn_v_guess)

neg_logLike = lambda *args: -ln_postprob(*args)
result = scipy.optimize.minimize(neg_logLike, z_guess, 
                                 args=(Xn_v, logdet_cov_K, inv_cov_K, nbins))
z_MaxLike = result["x"]
z0_MaxLike = z_MaxLike[0]
zn_MaxLike = z_MaxLike[1:]
lambdaN_v = np.exp(z0_MaxLike + zn_MaxLike)
norm_lambdaN_v = lambdaN_v / np.sum(lambdaN_v)

# We can repeat this for each player, and save the results
LL = np.zeros((num_players,np.prod(bins)))
year = 2013
for i, player in enumerate(top_players_nameList):
    norm_lambdaN_v = np.loadtxt('player%d_lambda_seed%d/norm_lambda_%s.txt'%(year, randSeed, player))
    LL[i,:] = norm_lambdaN_v[:]


n_comp = 10
model = NMF(n_components=n_comp, init='nndsvda', max_iter=8000, tol=1e-7,
            solver='mu', beta_loss='kullback-leibler')
W_10 = model.fit_transform(LL)
H_10 = model.components_
# We can do this for a lower tolerance and more iterations, and save the results

#%%

player, year = ('Jimmy Butler', 2013)
player_index = top_players_nameList.index(player)
Xn_v = player_shotHist_train[player]
histN_v = Xn_v/np.sum(Xn_v)
norm_lambdaN_v =  np.loadtxt('player%d_lambda_seed%d/norm_lambda_%s.txt'%(year, randSeed, player))
npzfile = np.load('NMF_results_lowTol.npz')
num_features = 10
norm_lambdaN_v_NMF = np.matmul(npzfile['W_%d_norm'%num_features] , npzfile['H_%d_norm'%num_features])[player_index,:]

plot_player_x6(player, norm_lambdaN_v, histN_v, norm_lambdaN_v_NMF, 
               add_title=['raw histogram',r'LGCP', 'LGCP + NMF'])

#%%

def plot_featuresH(H, n_comp, xedges, yedges, bins, figsize=(20,5), basis_names = None):
    plt.figure(figsize=figsize)
    if basis_names is None:
        basis_names = ['Basis vector %d'%(i) for i in range(n_comp)]
    for i in range(n_comp):
        plt.subplot(2, n_comp/2, i+1)

        extent = np.max(xedges), np.min(xedges), np.max(yedges), np.min(yedges)

        plt.imshow(H[i,:].reshape(bins[0],bins[1]).T, cmap=plt.cm.magma_r, alpha=.85,
                   extent=extent)
        plot_court.draw_court(outer_lines=True, lw=1.)

        plt.xlim(-300,300)
        plt.ylim(-100,500)
        plt.title(basis_names[i], fontsize=15)
        plt.axis('off')
    plt.savefig('NMF_features_nComp%d.png'%n_comp)

num_features = 10
column_names = ['at basket',
                'straight 3s',
                'corner 3s',
                'left block',
                'baseline mid',
                'high-post',
                'right block',
                'long 2s',
                'inside',
                'wing 3s'
               ]
df_Wnorm = pd.DataFrame(npzfile['W_%d_norm'%num_features], index=top_players_nameList, 
                        columns = column_names)
plot_featuresH(npzfile['H_%d_norm'%num_features], num_features, xedges, yedges, bins, figsize=(13,5),
               basis_names=column_names)

def plot_radar(df_Wnorm, n_comp):
    import radar_plot
    theta = radar_plot.radar_factory(num_features, frame='polygon')
    spoke_labels = df_Wnorm.columns.values.tolist()
    
    fig, axes = plt.subplots(figsize=(14, 14), nrows=2, ncols=3,
                             subplot_kw=dict(projection='radar'))
    
    for i, ax in enumerate(axes.flatten()):
        ax.set_rgrids([0.1, 0.2, 0.3, 0.4])
        ax.set_ylim(0, 0.5)
        player_name = df_Wnorm.index.values[i]
        dat = np.array(df_Wnorm.loc[player_name], dtype='float')
        ax.set_title(player_name, weight='bold', size='large', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        color = 'green'
        ax.plot(theta, dat, color=color)
        ax.fill(theta, dat, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)    
    fig.savefig('NMF_radarPlot_nComp%d.png'%n_comp)
    
plot_radar(df_Wnorm, num_features)


