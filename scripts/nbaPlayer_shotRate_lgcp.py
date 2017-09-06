import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy

import seaborn as sns

import plot_court
import sklearn.model_selection
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


num_players = 350
top_players_shotNum = df.PLAYER_NAME.value_counts()[:num_players]
top_players_nameList = top_players_shotNum.index.tolist()


################################################################
################################################################


train_df = {}
test_df = {}
for i, player in enumerate(set(top_players_nameList)):  
    temp = df[df.PLAYER_NAME == player]
    train_df[player], test_df[player] = sklearn.model_selection.train_test_split(temp, test_size = 0.2)

    
player_shotHist_train = {}
for i, player in enumerate(set(top_players_nameList)):  
    temp = train_df[player]
    hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                        temp.SHOT_MADE_FLAG,
                                                                        statistic='count',
                                                                        bins=bins, 
                                                                        range=binRange)
    player_shotHist_train[player] = hist2d.flatten()
    
    
player_shotMadeHist_train = {}
for i, player in enumerate(set(top_players_nameList)):  
    temp = train_df[player][train_df[player].SHOT_MADE_FLAG == 1.0]
    hist2d, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(temp.LOC_X, temp.LOC_Y, 
                                                                        temp.SHOT_MADE_FLAG,
                                                                        statistic='count',
                                                                        bins=bins, 
                                                                        range=binRange)
    player_shotMadeHist_train[player] = hist2d.flatten()


################################################################
# Likelihood-max for shot count distribution (Log Gaussian Cox Process)
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
    
def plot_player_normLambda(player):
    norm_lambdaN_v = np.loadtxt('player_lambda/norm_lambda_%s.txt'%(player))
    LAMBDA_v = np.reshape(norm_lambdaN_v, bins)
    ##########
    extent = np.min(xedges), np.max(xedges), np.max(yedges), np.min(yedges)
    
#    plt.imshow(LAMBDA_v.T, cmap=plt.cm.gist_heat_r, alpha=.9,
#               extent=extent)
    plt.imshow(LAMBDA_v.T, cmap=plt.cm.RdYlBu_r, alpha=.5,
               extent=extent)
    plot_court.draw_court(outer_lines=True, lw=1.5)
    
    plt.xlim(-300,300)
    plt.ylim(-100,500)
    plt.title('%s: LGCP'%(player), fontsize=15)
#    plt.axis('off')
    plt.tight_layout()
    plt.show()


################################################################
################################################################


LL = np.zeros((num_players,np.prod(bins)))
for i, player in enumerate(top_players_nameList):
    try:
        norm_lambdaN_v = np.loadtxt('player_lambda/norm_lambda_%s.txt'%(player))
    except:
        Xn_v = player_shotHist_train[player]
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
    
        np.savetxt('player_lambda/norm_lambda_%s.txt'%(player), norm_lambdaN_v)
    print(player)
    LL[i,:] = norm_lambdaN_v[:]

    

################################################################
################################################################


n_comp = 10
model = NMF(n_components=n_comp, init='nndsvd', max_iter=2000, solver='cd')
W = model.fit_transform(LL)
H = model.components_    


plt.figure(figsize=(20,14))
for i in range(n_comp):
    plt.subplot(2, n_comp/2 + 1, i+1)

    extent = np.max(xedges), np.min(xedges), np.max(yedges), np.min(yedges)

#    plt.imshow(H[i,:].reshape(bins[0],bins[1]).T, cmap=plt.cm.gist_heat_r, alpha=.9,
#               extent=extent)
    plt.imshow(H[i,:].reshape(bins[0],bins[1]).T, cmap=plt.cm.RdYlBu_r, alpha=.5,
               extent=extent)
    plot_court.draw_court(outer_lines=True, lw=1.)

    plt.xlim(-300,300)
    plt.ylim(-100,500)
    plt.title('Basis vector %d'%(i), fontsize=15)
    plt.axis('off')
plt.show()


################################################################
# Likelihood-max for FG% distribution (Inhomogeneous Binomial Process)
################################################################

# When compared to LGCP, still use a spatially vary field variable zn
# local success probability (or field goal %, i.e. FG%) is the logistic function of zn


def ln_prior_binomial(zn_v, det_cov_K, inv_cov_K):
    part1 = -np.log(2 * np.pi * (det_cov_K**0.5))
    part2 = -0.5 * np.dot(zn_v, np.dot(inv_cov_K, zn_v))
    return part1 + part2

def binomialP_func(z0, zn_v):
    # Input: field variables
    # Output: Bernouli success prob (from logistic function)
    return 1./(1. + np.exp(-(z0 + zn_v)))

def ln_factorial(n):
    # an improvement of the Sterling Approximation of log(n!)
    # given by Srinivasa Ramanujan (Ramanujan 1988)
    # scipy.misc.factorial stops worknig at large values of n
    sterling = n * np.log(n) - n
    correct = (1./6) * np.log(n * (1 + 4*n*(1 + 2*n))) + np.log(np.pi)/2
    return sterling + correct

def ln_binomialCoeff(n, k):
    return ln_factorial(n) - ln_factorial(k) - ln_factorial(n-k)

def ln_likelihood_binomial(z0, zn_v, Xn_made_v, Xn_v):
    part1 = ln_binomialCoeff(Xn_v, Xn_made_v)
    part2 = Xn_made_v * np.log(binomialP_func(z0, zn_v))
    part3 = (Xn_v - Xn_made_v) * np.log(1 - binomialP_func(z0, zn_v))
    #print(np.sum(part1), np.sum(part2), np.sum(part3))
    #print(part3)
    return np.sum(part1 + part2 + part3)

def ln_postprob_binomial(z, Xn_made_v, Xn_v, det_cov_K, inv_cov_K):
    z0 = z[0]
    zn_v = z[1:]
    return ln_prior_binomial(zn_v, det_cov_K, inv_cov_K) + ln_likelihood(z0, zn_v, Xn_v)


################################################################
################################################################
    
def plot_player_fgPercent(player):
    fgPercent_v = np.loadtxt('player_FGp/FGpercent_%s.txt'%(player))
    FGper_v = np.reshape(fgPercent_v, bins)
    ##########
    extent = np.min(xedges), np.max(xedges), np.max(yedges), np.min(yedges)
    
    plt.imshow(FGper_v.T, cmap=plt.cm.RdYlBu_r, alpha=.5, vmax=1.,
               extent=extent)
    plot_court.draw_court(outer_lines=True, lw=1.5)
    
    plt.xlim(-300,300)
    plt.ylim(-100,500)
    plt.title('%s: est FG percent'%(player), fontsize=15)
#    plt.axis('off')
    plt.tight_layout()
    plt.show()


################################################################
################################################################

player = 'Kevin Durant'
try:
    fgPercent_v = np.loadtxt('player_FGp/FGpercent_%s.txt'%(player))
except:
    Xn_v = player_shotHist_train[player]
    Xn_made_v = player_shotMadeHist_train[player]
#    z0_guess = -np.log((float(np.sum(Xn_v)) / np.sum(Xn_made_v)) - 1)
    z0_guess = -10.0
    zn_v_guess = np.zeros(len(Xn_v))
    z_guess = np.append(z0_guess, zn_v_guess)
    
    neg_logLike = lambda *args: -ln_postprob_binomial(*args)
    result = scipy.optimize.minimize(neg_logLike, z_guess, 
                                     args=(Xn_made_v, Xn_v, det_cov_K, inv_cov_K))
    z_MaxLike = result["x"]
    z0_MaxLike = z_MaxLike[0]
    zn_MaxLike = z_MaxLike[1:]
    fgPercent_v = binomialP_func(z0_MaxLike, zn_MaxLike)
    np.savetxt('player_FGp/FGpercent_%s.txt'%(player), fgPercent_v)

plot_player_fgPercent(player)
plot_player_normLambda(player)

#FGper = np.zeros((num_players,np.prod(bins)))
#for i, player in enumerate(top_players_nameList):
#    try:
#        fgPercent_v = np.loadtxt('player_FGp/FGpercent_%s.txt'%(player))
#    except:
#        Xn_v = player_shotHist_train[player]
#        Xn_made_v = player_shotMadeHist_train[player]
#        z0_guess = -np.log((float(len(Xn_v)) / len(Xn_made_v)) - 1)
#        zn_v_guess = np.zeros(len(Xn_v))
#        z_guess = np.append(z0_guess, zn_v_guess)
#    
#        neg_logLike = lambda *args: -ln_postprob_binomial(*args)
#        result = scipy.optimize.minimize(neg_logLike, z_guess, 
#                                         args=(Xn_made_v, Xn_v, det_cov_K, inv_cov_K))
#        z_MaxLike = result["x"]
#        z0_MaxLike = z_MaxLike[0]
#        zn_MaxLike = z_MaxLike[1:]
#        fgPercent_v = binomialP_func(z0_MaxLike, zn_MaxLike)
#    
#        np.savetxt('player_FGp/FGpercent_%s.txt'%(player), fgPercent_v)
#    print(player)
#    FGper[i,:] = fgPercent_v[:]
    
    
################################################################
################################################################


#n_comp = 10
#model = NMF(n_components=n_comp, init='nndsvd', max_iter=2000, solver='cd')
#W = model.fit_transform(LL)
#H = model.components_    
#
#
#plt.figure(figsize=(20,14))
#for i in range(n_comp):
#    plt.subplot(2, n_comp/2 + 1, i+1)
#
#    extent = np.max(xedges), np.min(xedges), np.max(yedges), np.min(yedges)
#
#    plt.imshow(H[i,:].reshape(bins[0],bins[1]).T, cmap=plt.cm.gist_heat_r, alpha=.9,
#               extent=extent)
#    plot_court.draw_court(outer_lines=True, lw=1.)
#
#    plt.xlim(-300,300)
#    plt.ylim(-100,500)
#    plt.title('Basis vector %d'%(i), fontsize=15)
#    plt.axis('off')
#plt.show()
