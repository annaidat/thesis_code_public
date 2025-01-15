import time
from argparse import ArgumentParser
import MDAnalysis
import ray
import logging
import warnings
import os
import mdtraj as md
import pandas as pd
import numpy as np
import shutil
import os
import sys
from FRETpredict import FRETpredict
sys.path.append('BLOCKING')
from main import BlockAnalysis
from statsmodels.tsa.stattools import acf
import ast

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--log',dest='log_path',type=str,required=True)
parser.add_argument('--cycle',dest='cycle',type=int,required=True)
parser.add_argument('--num_cpus',dest='num_cpus',type=int)
parser.add_argument('--cutoff',dest='cutoff',type=float)
args = parser.parse_args()

logging.basicConfig(filename=args.log_path+'/log',level=logging.INFO)

dp = 0.005 #step size for alphas
theta = 50 #regularization parameter
eta = 10 #to adjust balance between FRET and Rg
xi_0 = 1 #initial value for xi
rc = 2.0 #cutoff for the LJ potential, is 2.0, has sometimes been trained with 4.0, but this is not relevant for now. 
#gamma = 0 #to adjust Rg data  

os.environ["NUMEXPR_MAX_THREADS"]="1"

df = pd.read_csv('residues_CV3_original.csv').set_index('one')
if 'alphas' not in df.columns:
    df['alphas'] = 0.0
df[f'alphas_{args.cycle:d}'] = df.alphas

# Load Rg data
proteins_rg = pd.read_csv('rg_trainset.csv',index_col=0)
#proteins_rg = proteins_rg.sample(frac=1,random_state=121)
proteins_rg['path'] = proteins_rg.index.map(lambda name: f'{name:s}/{args.cycle:d}/') 
# Adding extra paths
#proteins_rg['rgpath'] = ('/groups/sbinlab/trolle/salt/simulations/november/saxs/')

# Load FRET data
def parse_list_columns(df, column_names):
    for col in column_names:
        df[col] = df[col].apply(ast.literal_eval)
    return df

proteins_fret = pd.read_csv('fret_trainset.csv', index_col=0,delimiter=':')
proteins_fret = parse_list_columns(proteins_fret, ["dyes", "labels"])
proteins_fret['path'] = proteins_fret.index.map(lambda name: f'{name:s}/{args.cycle:d}')
#proteins_fret['aapath'] = '/groups/sbinlab/trolle/salt/cg2all/november/'+proteins_fret.index.map(lambda name: f'{name:s}/joined')
proteins_fret['aapath'] = proteins_fret.index.map(lambda name: f'{name:s}/{args.cycle:d}/joined/')

for name in proteins_fret.index:
    if not os.path.isdir(proteins_fret.loc[name].path):
        os.mkdir(proteins_fret.loc[name].path)
    if not os.path.isdir(proteins_fret.loc[name].path+'/calcFRET'):
        os.mkdir(proteins_fret.loc[name].path+'/calcFRET')

ray.init(num_cpus=args.num_cpus)

def calc_ratio_and_chi(df, name): #to normalise values to the 0.15 M salt concentration
    ratio_reference = df.loc[name, 'ratio_reference']
    df.loc[name, 'sim_ratio'] = df.loc[name, (f'sim_{args.cycle:d}')]/df.loc[ratio_reference,(f'sim_{args.cycle:d}')]
    df.loc[name, 'exp_ratio'] = df.loc[name, ('experimental_value')]/df.loc[ratio_reference,('experimental_value')]
    exp_error = df.loc[name, ('experimental_error')]
    ref_error = df.loc[ratio_reference,('experimental_error')]
    df.loc[name, 'exp_ratio_err'] = df.loc[name, 'exp_ratio'] * np.sqrt(np.power((exp_error/df.loc[name, ('experimental_value')]),2)+np.power((ref_error/df.loc[ratio_reference,('experimental_value')]),2))
    df.loc[name,(f'chi2_{args.cycle:d}')] = ((df.loc[name,'sim_ratio']-df.loc[name,'exp_ratio'])/df.loc[name,'exp_ratio_err'])**2

@ray.remote(num_cpus=1)
def calc_fret(name, prot):
    prefix = prot.path+f'/calcFRET/{name:s}'
    u = MDAnalysis.Universe(prot.aapath+'/allatom.pdb',prot.aapath+'/allatom.dcd')
    FRET = FRETpredict(u, log_file = prot.path+'/fret_log', residues = prot.labels,
            temperature = prot.temp,
            donor=prot.dyes[0], acceptor=prot.dyes[1], electrostatic=False,
            libname_1=prot.dyes[0]+' cutoff10',
            libname_2=prot.dyes[1]+ ' cutoff10',
            fixed_R0=True,
            r0=prot.r0,
            output_prefix=prefix, verbose=False, calc_distr=False)
    FRET.run()
    df = pd.read_pickle(prefix+f'-data-{prot.labels[0]:d}-{prot.labels[1]:d}.pkl')
    eff = df.loc['Edynamic2','Average']
    eff_err = df.loc['Edynamic2','SD']
    #save values for later
    df.loc[name, f'sim_{args.cycle:d}'] = eff
    df.loc[name, f'err_{args.cycle:d}'] = eff_err
    return name, eff, eff_err

def reweight_fret(name, prot, weights):
    prefix = prot.path+f'/calcFRET/{name:s}'
    u = MDAnalysis.Universe(prot.aapath+'/allatom.pdb',prot.aapath+'/allatom.dcd')
    FRET = FRETpredict(u, log_file = prot.path+'/fret_log', residues = prot.labels,
            temperature = prot.temp,
            donor=prot.dyes[0], acceptor=prot.dyes[1], electrostatic=False,
            libname_1=prot.dyes[0]+' cutoff10',
            libname_2=prot.dyes[1]+ ' cutoff10',
            fixed_R0=True,
            r0=prot.r0,
            output_prefix=prefix, verbose=False, calc_distr=False)
    FRET.reweight(user_weights = weights)
    df = pd.read_pickle(prefix+f'-data-{prot.labels[0]:d}-{prot.labels[1]:d}.pkl')
    eff = df.loc['Edynamic2','Average']
    eff_err = df.loc['Edynamic2','SD']
    #chi2_fret = np.power((prot.EC-eff)/prot.EC_err,2)
    return eff, eff_err

def autoblock(cv, multi=1):
    block = BlockAnalysis(cv, multi=multi)
    block.SEM()
    return block.sem, block.bs

def reweight_rg(name,prot,weights):
    rg_array = np.load(prot.path+'/joined/rg_array.npy')
    rg = np.dot(rg_array, weights)
    #chi2_rg = np.power((prot.exp_rg-rg)/prot.exp_rg_err,2)
    return rg

def calc_rg(df,name,prot):
    traj = md.load_dcd((prot.path+'/joined/traj.dcd'),prot.path+'/joined/top.pdb')
    traj = traj.atom_slice(traj.top.select('name CA'))
    masses = df.loc[list(prot.fasta),'MW'].values
    masses[0] += 2 #what is this. Is it for the N-terminal? aha yes it is, adds the hydrogens to the N-terminal
    masses[-1] += 16 #what is this. Is it for the C-terminal? Adds the oxygen (and H?) to the C-terminal
    # calculate the center of mass
    cm = np.sum(traj.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(traj.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rg_array = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    rg_mean = np.mean(rg_array)
    rg_se, rg_blocksize = autoblock(rg_array)
    #chi2_rg = np.power((prot.exp_rg-rg_mean)/prot.exp_rg_err,2)
    # calculate acf
    acf_rg_2 = acf(rg_array,nlags=2,fft=True)[2]
    return rg_array, rg_mean, rg_se, rg_blocksize, acf_rg_2

def calc_ah_energy(df,name,prot):
    term_1 = np.load(prot.path+'/joined/energies/energy_sums_1.npy')
    term_2 = np.load(prot.path+'/joined/energies/energy_sums_2.npy')
    unique_pairs = np.load(prot.path+'/joined/energies/unique_pairs.npy')
    df.lambdas = df.lambdas + df.alphas*(prot.ionic-0.15)
    lambdas = 0.5*(df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)
    return term_1+np.nansum(lambdas*term_2,axis=1)

@ray.remote(num_cpus=1)
def calc_weights(df,name,prot):
    new_ah_energy = calc_ah_energy(df,name,prot) #changed prot to prot.loc[name], prot doesnt have a .path. Deleted and changed all the way down when calc_weights is called.   
    ah_energy = np.load(prot.path+'/joined/AHenergy.npy')
    kT = 8.3145*prot.temp*1e-3
    weights = np.exp((ah_energy-new_ah_energy)/kT)
    weights /= weights.sum()
    eff = np.exp(-np.sum(weights*np.log(weights*weights.size)))
    return name,weights,eff

def reweight(dp,df, proteins_fret,proteins_rg): #df is residues_CV3.csv
    trial_proteins_fret = proteins_fret.copy()
    trial_proteins_rg = proteins_rg.copy()
    trial_df = df.copy()
    res_sel = np.random.choice(trial_df.index, 5, replace=False)
    #res_sel = [FYW]
    trial_df.loc[res_sel,'alphas'] += np.random.normal(0,dp,res_sel.size)

    # calculate LJ energies, weights and fraction of effective frames
    # I think there was an error here. Used to be ray.get([calc_weights.remote(name,prot,trial_df) but as stated above this does not work
    # Changed it to trial_df.loc[name] but that does not make sense since trial_df is a copy of df, which is the residues csv
    # Changed it to match the github version from cutoffs paper now
    weights = ray.get([calc_weights.remote(trial_df,name,prot) for name,prot in pd.concat((trial_proteins_fret,trial_proteins_rg),sort=True).iterrows()])
    for name,w,eff in weights:
        if eff < 0.6:
            return False, df, proteins_fret, proteins_rg

    for name,w,eff in weights:
        if name in trial_proteins_fret.index:
            eff, eff_err = reweight_fret(name,trial_proteins_fret.loc[name],w)
            trial_proteins_fret.at[name,'eff'] = eff
            trial_proteins_fret.at[name, (f'sim_{args.cycle:d}')] = eff
            trial_proteins_fret.at[name,'eff_err'] = eff_err
            calc_ratio_and_chi(trial_proteins_fret,name)
            #chi2_fret = trial_proteins_fret.at[name,(f'chi2_{args.cycle:d}')] not needed, chi2 is already placed in the dataframe with the above func.
        elif name in trial_proteins_rg.index:
            rg = reweight_rg(name,trial_proteins_rg.loc[name],w)
            trial_proteins_rg.at[name,'rg'] = rg
            trial_proteins_rg.at[name, (f'sim_{args.cycle:d}')] = rg
            calc_ratio_and_chi(trial_proteins_rg,name)
            #chi2_rg = trial_proteins_rg.at[name,(f'chi2_{args.cycle:d}')]
    return True, trial_df, trial_proteins_fret, trial_proteins_rg

logging.info(df.alphas)

fret_eff = ray.get([calc_fret.remote(name,proteins_fret.loc[name]) for name in proteins_fret.index])
for name, eff, eff_err in fret_eff:
    ah_ene = calc_ah_energy(df,name,proteins_fret.loc[name])
    np.save(proteins_fret.loc[name].path+'/joined/AHenergy.npy',ah_ene)
    proteins_fret.at[name,'E'] = eff
    proteins_fret.at[name,'E_err'] = eff_err
    proteins_fret.at[name, (f'sim_{args.cycle:d}')] = eff
    calc_ratio_and_chi(proteins_fret,name)
    #proteins_fret.at[name,'chi2_fret'] = chi2_fret
    if os.path.exists(proteins_fret.loc[name].path+'/initFRET'):
        shutil.rmtree(proteins_fret.loc[name].path+'/initFRET')
    shutil.copytree(proteins_fret.loc[name].path+'/calcFRET',proteins_fret.loc[name].path+'/initFRET')
proteins_fret.to_pickle(str(args.cycle)+'_init_proteins_fret.pkl')

for name in proteins_rg.index:
    ah_ene = calc_ah_energy(df,name,proteins_rg.loc[name])
    np.save(proteins_rg.loc[name].path+'/joined/AHenergy.npy',ah_ene)
    rg_array, rg_mean, rg_se, rg_blocksize, acf_rg_2 = calc_rg(df,name,proteins_rg.loc[name])
    np.save(proteins_rg.loc[name].path+'/joined/rg_array.npy',rg_array)
    proteins_rg.at[name,'rg'] = rg_mean
    proteins_rg.at[name, (f'sim_{args.cycle:d}')] = rg_mean
    proteins_rg.at[name,'rg_err'] = rg_se
    proteins_rg.at[name,'rg_bs'] = rg_blocksize
    calc_ratio_and_chi(proteins_rg,name)
    #proteins.at[name,'chi2_rg'] = chi2_rg
    proteins_rg.at[name,'acf_rg_2'] = acf_rg_2
proteins_rg.to_pickle(str(args.cycle)+'_init_proteins_rg.pkl')

chi2_column_name = f'chi2_{args.cycle:d}'
logging.info('Initial Chi2 FRET {:.3f} +/- {:.3f}'.format(proteins_fret[chi2_column_name].mean(),proteins_fret[chi2_column_name].std()))
logging.info('Initial Chi2 Gyration Radius {:.3f} +/- {:.3f}'.format(proteins_rg[chi2_column_name].mean(), proteins_rg[chi2_column_name].std()))

theta_prior = theta * np.sum(df.alphas**2) #changed alpha to alphas
xi = xi_0

logging.info('Initial theta*prior {:.2f}'.format(theta_prior))
logging.info('theta {:.2f}'.format(theta))
logging.info('xi {:g}'.format(xi))

dfchi2 = pd.DataFrame(columns=['chi2_fret','chi2_rg','theta_prior','alphas','xi'])
cycle_key = f'chi2_{args.cycle:d}'  # Construct the column name for the current cycle
dfchi2.loc[0] = [
    proteins_fret[cycle_key].mean(),
    proteins_rg[cycle_key].mean(),    
    theta_prior,                      
    df.alphas,                        
    xi                                
]

#dfchi2.loc[0] = [proteins_fret[chi2_column_name],proteins_rg[chi2_column_name],theta_prior,df.alphas,xi] this was missing the actual name...

time0 = time.time()
micro_cycle = 0

for k in range(2,200000):

    if (xi<1e-8):
        xi = xi_0
        micro_cycle += 1
        if (micro_cycle==4):
            logging.info('xi {:g}'.format(xi))
            break

    xi = xi * .99
    passed, trial_df, trial_fret, trial_rg = reweight(dp,df,proteins_fret,proteins_rg)
    if passed:
        #theta_prior = theta * np.sum(np.sum(df.alphas**2)) #changed alpha to alphas 
        theta_prior = theta * np.sum(np.sum(trial_df.alphas**2))
        loss_2 = eta*trial_fret[chi2_column_name].mean() + trial_rg[chi2_column_name].mean() + theta_prior
        loss_1 = eta*proteins_fret[chi2_column_name].mean() + proteins_rg[chi2_column_name].mean() + dfchi2.iloc[-1]['theta_prior']
        delta = loss_2 - loss_1
        if ( np.exp(-delta/xi) > np.random.rand() ):
            proteins_fret = trial_fret.copy()
            proteins_rg = trial_rg.copy()
            df = trial_df.copy()
            dfchi2.loc[k-1] = [trial_fret[chi2_column_name].mean(),trial_rg[chi2_column_name].mean(),theta_prior,df.alphas,xi]
            logging.info('Acc Iter {:d}, micro cycle {:d}, xi {:g}, Chi2 FRET {:.2f}, Chi2 Rg {:.2f}, theta*prior {:.2f}'.format(k-1,micro_cycle,xi,trial_fret[chi2_column_name].mean(),trial_rg[chi2_column_name].mean(),theta_prior))
            if k % 1000 == 0:
                logging.info(f"Iteration {k}, Elapsed Time: {time.time() - time0:.2f}s")

       
logging.info('Timing Reweighting {:.3f}'.format(time.time()-time0))
logging.info('Theta {:.3f}'.format(theta))
dfchi2['loss'] = eta*dfchi2.chi2_fret + dfchi2.chi2_rg + dfchi2.theta_prior
dfchi2.to_pickle(str(args.cycle)+'_chi2.pkl')
df.alphas = dfchi2.loc[dfchi2.loss.idxmin()].alphas
df[f'alphas_{args.cycle+1:d}'] = df.alphas
logging.info(df.alphas)
proteins_rg.to_pickle('proteins_rg.pkl')
proteins_fret.to_pickle('proteins_fret.pkl')
logging.info('Cost at 0: {:.2f}'.format(dfchi2.loc[0].loss))
logging.info('Min Cost at {:d}: {:.2f}'.format(dfchi2.loss.idxmin(),dfchi2.loss.min()))
