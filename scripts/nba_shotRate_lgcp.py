import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy

import seaborn as sns

import plot_court
import sklearn.model_selection
import pymc3 as pm
import emcee
from sklearn.decomposition import NMF


# Load data and keep desired columns
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


################################################################
################################################################


# Number of bins and range in each direction used to make the grid for analysis
ii = 1
if (ii == 0):
    bins, binRange = ([20,14], [[-250,250], [-47.5,302.5]])
    sigma2 = 3.5
elif (ii == 1):
    bins, binRange = ([25,18], [[-250,250], [-47.5,312.5]])
    sigma2 = 12.5
    sigma2 = 20
    sigma2 = 40
    sigma2 = 60
elif (ii == 2):
    bins, binRange = ([30,21], [[-250,250], [-47.5,302.5]])
    sigma2 = 69.
elif (ii == 3):
    bins, binRange = ([40,28], [[-250,250], [-47.5,302.5]])
elif (ii == 4):
    bins, binRange = ([50,35], [[-250,250], [-47.5,302.5]])

hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(df.LOC_X, df.LOC_Y, 
                                                                    df.SHOT_MADE_FLAG,
                                                                    statistic='count',
                                                                    bins=bins, 
                                                                    range=binRange)
# Creating the grid we will use for analysis
XX, YY = np.meshgrid(xedges, yedges)
binX_flat = XX.T[:-1,:-1].flatten()
binY_flat = YY.T[:-1,:-1].flatten()
binXY = np.column_stack((binX_flat.T, binY_flat.T))
dist_matrix = scipy.spatial.distance_matrix(binXY, binXY)


def cov_func(dist_matrix, sigma2, phi2):
    return sigma2 * np.exp( -(dist_matrix**2) / (2 * phi2) )

phi2 = 25.**2
# sigma2 = 1./np.sqrt(2 * np.pi * phi2)

cov_K = cov_func(dist_matrix, sigma2, phi2)
det_cov_K = np.linalg.det(cov_K)
inv_cov_K = np.linalg.inv(cov_K)


################################################################
################################################################


train_df = {}
test_df = {}
for i, team_abbrev in enumerate(set(df.TEAM_ABBREV)):  
    temp = df[df.TEAM_ABBREV == team_abbrev]
    train_df[team_abbrev], test_df[team_abbrev] = sklearn.model_selection.train_test_split(temp, test_size = 0.2)

    
teams_shotHist_train = {}
for i, team_abbrev in enumerate(set(df.TEAM_ABBREV)):  
    temp = train_df[team_abbrev]
    hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                        temp.SHOT_MADE_FLAG,
                                                                        statistic='count',
                                                                        bins=bins, 
                                                                        range=binRange)
    teams_shotHist_train[team_abbrev] = hist2d.flatten()
teams_shotHist_train


################################################################
################################################################


def ln_prior(zn_v, det_cov_K, inv_cov_K):
    part1 = -np.log(2 * np.pi * (det_cov_K**0.5))
    part2 = -0.5 * np.dot(zn_v, np.dot(inv_cov_K, zn_v))
    return part1 + part2

def lambdaN_func(z0, zn_v):
    return np.exp(z0 + zn_v)

def ln_lambdaN_func(z0, zn_v):
    return z0 + zn_v

def ln_factorial(n):
    # an improvement of the Sterling Approximation of log(n!)
    # given by Srinivasa Ramanujan (Ramanujan 1988)
    # scipy.misc.factorial stops worknig at large values of n
    sterling = n * np.log(n) - n
    correct = (1./6) * np.log(n * (1 + 4*n*(1 + 2*n))) + np.log(np.pi)/2
    return sterling + correct

def ln_likelihood(z0, zn_v, Xn_v):
    part1 = -lambdaN_func(z0, zn_v)
    part2 = Xn_v * ln_lambdaN_func(z0, zn_v)
    part3 = np.nan_to_num(-ln_factorial(Xn_v))
    #print(np.sum(part1), np.sum(part2), np.sum(part3))
    #print(part3)
    return np.sum(part1 + part2 + part3)

def ln_postprob(z, Xn_v, det_cov_K, inv_cov_K):
    z0 = z[0]
    zn_v = z[1:]
    return ln_prior(zn_v, det_cov_K, inv_cov_K) + ln_likelihood(z0, zn_v, Xn_v)


################################################################
################################################################


LL = np.zeros((30,np.prod(bins)))
for i, team_abbrev in enumerate(teams_shotHist_train):
    try:
        norm_lambdaN_v = np.loadtxt('team_lambda/norm_lambda_%s.txt'%(team_abbrev))
    except:
        Xn_v = teams_shotHist_train[team_abbrev]
        z0_guess = np.log(np.mean(Xn_v))
        zn_v_guess = np.zeros(len(Xn_v))
        z_guess = np.append(z0_guess, zn_v_guess)
    
        neg_logLike = lambda *args: -ln_postprob(*args)
        result = scipy.optimize.minimize(neg_logLike, z_guess, 
                                         args=(Xn_v, det_cov_K, inv_cov_K))
        z_MaxLike = result["x"]
        z0_MaxLike = z_MaxLike[0]
        zn_MaxLike = z_MaxLike[1:]
        lambdaN_v = np.exp(z0_MaxLike + zn_MaxLike)
        norm_lambdaN_v = lambdaN_v / np.sum(lambdaN_v)
    
        np.savetxt('team_lambda/norm_lambda_%s.txt'%(team_abbrev), norm_lambdaN_v)
    print(team_abbrev)
    LL[i,:] = norm_lambdaN_v[:]
    

################################################################
################################################################
    
    
n_comp = 8
model = NMF(n_components=n_comp, init='nndsvda', max_iter=2000, solver='cd', sparseness='components')
W = model.fit_transform(LL)
H = model.components_    


plt.figure(figsize=(20,14))
for i in range(n_comp):
    plt.subplot(1, n_comp, i+1)

    extent = np.max(xedges), np.min(xedges), np.max(yedges), np.min(yedges)

    plt.imshow(H[i,:].reshape(bins[0],bins[1]).T, cmap=plt.cm.gist_heat_r, alpha=.9, vmax=1,
               extent=extent)
    plot_court.draw_court(outer_lines=True, lw=1.)

    plt.xlim(-300,300)
    plt.ylim(-100,500)
    plt.title('Basis vector %d'%(i), fontsize=15)
    plt.axis('off')
plt.show()
