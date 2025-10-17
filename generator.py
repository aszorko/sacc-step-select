# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:17:35 2024

@author: aszor
"""

import argparse
import numpy as np
import os
import pandas as pd

import torch
from abc import abstractmethod
from stepselect import load_data, Conditional_dXdY, Unconditional_XY, Unconditional_ThetaR_Indep, Unconditional_ThetaR_Joint, LogNormalDist, MarkovModel, NNModel, ParametricModel, get_spline
from stepselect_analysis import run_regression, stepwise_regression, linspline_regression

def cast(tensor, device):
    return tensor.float().to(device)

def getmesh(extent,N,device="cpu"):
    x = np.linspace(extent[0],extent[1], N) #edges (N)
    y = np.linspace(extent[2],extent[3], N)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    #s = np.stack([x, y], axis=1)
    s = np.stack([x[:-1]+dx/2, y[:-1]+dy/2], axis=1) #centres (N-1)

    X, Y = np.meshgrid(s[:, 0], s[:, 1])
    S = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    S = torch.tensor(S).to(device)
    S = S.float()

    return S,x,y,dx,dy

class Metropolis2D:
    
    def __init__(self,xlim,ylim,minaccept=2,maxiter=100,sr=10,sig=1.0,extent=None):
       self.samplerate=sr
       self.minaccept=minaccept
       self.maxiter=maxiter
       self.x=0
       self.y=0
       self.sig=sig
       self.xlim = xlim
       self.ylim = ylim
       self.rng = np.random.default_rng()
       self.models = []
          
    def set_seed(self,seed):
        self.rng = np.random.default_rng(seed=seed)


    @abstractmethod
    def evaluate(self,t,x,y):
       pass
   
    def reset(self,rand=False):
       for m in self.models:
            m.reset()
       if rand:
          self.x = self.xlim[0] + self.rng.uniform()*(self.xlim[1]-self.xlim[0])
          self.y = self.ylim[0] + self.rng.uniform()*(self.ylim[1]-self.ylim[0])
       else:
          self.x = self.xlim[0] + 0.5*(self.xlim[1]-self.xlim[0])
          self.y = self.ylim[0] + 0.5*(self.ylim[1]-self.ylim[0])
          
    def iterate(self,t):
       i=0
       j=0
       currP = self.evaluate(t,self.x,self.y)
       while i < self.minaccept or j < self.samplerate:
           if j>self.maxiter:
               print('max iterations, logP =',np.log(currP))
               break
           newx = self.x + self.sig*self.rng.normal()
           newy = self.y + self.sig*self.rng.normal()
           a = self.rng.uniform()
           if newx < self.xlim[0] or newx > self.xlim[1] or newy < self.ylim[0] or newy > self.ylim[1]:
               continue
           newP = self.evaluate(t,newx,newy)
           #print(newP)
           j += 1
           if not np.isfinite(1/currP):
               ratio=1.0
           else:
               ratio = newP/currP
           if  a<ratio:
               i += 1
               #print('accept')
               self.x = newx
               self.y = newy
               currP = newP
       
       print(self.x,self.y)
       print(i/j) #i of total j valid candidate steps accepted
       
       return self.x, self.y, i/j
    
class Metro_NN(Metropolis2D):
    def __init__(self,model,xlim,ylim,t_extra=1.0,dt=0.02,**kwargs):
        super().__init__(xlim,ylim,**kwargs)
        self.nn_model = model
        self.models = [self.nn_model]
        self.max_hist = model.max_hist
        self.t_extra = t_extra
        self.dt = dt
        
    def evaluate(self,t,x,y):
        pred = self.nn_model.get_predictors_alt(t,[[x,y]],logit=False)
        
        return pred.iloc[0,0]

    def sample_t(self):
        #get intensity function
        if i>self.max_hist:
            t0 = self.nn_model.event_times[0,i-self.max_hist]
            dl = self.condition_t(self.nn_model.event_times[:,(i-self.max_hist):]-t0,self.nn_model.spatial_locations[:,(i-self.max_hist):,:])
        else:
            dl = self.condition_t(self.nn_model.event_times,self.nn_model.spatial_locations)
        
        #sample intensity for new time point
        for j in range(len(dl)):
            r = self.rng.uniform()
            if r<dl[j]:
                t = ((j+0.5)*self.dt)
                break
            
        return t
        
    def condition_t(self,event_times,spatial_locations):
        t_steps = event_times.shape[1]
        
        with torch.no_grad():
           intensities,lambdas,hiddens,_ = self.nn_model.model.temporal_model.lambda_traj(event_times[:,:t_steps], spatial_locations[:,:t_steps,:], None, 0, event_times[0,t_steps-1]+self.t_extra, self.dt)
        l = np.array([x.detach().cpu().numpy() for x in lambdas]).T
        dl = np.diff(l[0,:]) #intensity function
    
        return dl



    
class Metro_SemiPar(Metropolis2D):
    def __init__(self,markov_model,param_model,mod_res,timedata,xlim,ylim,dt=0.02,t_max=5.0,**kwargs):
        super().__init__(xlim,ylim,**kwargs)
        self.markov_model = markov_model
        self.param_model = param_model
        self.time_edges = np.arange(0,t_max,dt)
        self.time_dist = self.get_time_dist(timedata)
        if param_model is None:
            self.models = [self.markov_model]
        elif markov_model is None:
            self.models = [self.param_model]
        else:
            self.models = [self.markov_model,self.param_model]
        #self.coeffs = coeffs
        self.mod_res = mod_res
        self.dt = dt
        self.t_max = t_max
        
    def evaluate(self,t,x,y):
        if self.param_model is None:  
           pred = self.markov_model.get_predictors(t,[[x,y]],logit=False)
           return pred.iloc[0,0]
        elif self.markov_model is None:
           pred1 = self.param_model.get_predictors(t,[[x,y]])
           logit = self.mod_res.get_prediction(pred1).linpred[0]
           #logit = sum(self.coeffs[1:]*np.array(pred1.iloc[0,:]))           
           return np.exp(logit) #1/(1+np.exp(-logit))
        else:
           pred1 = self.param_model.get_predictors(t,[[x,y]])
           p_markov = self.markov_model.get_predictors(t,[[x,y]],logit=False)
           p = p_markov.iloc[0,0]
           #pred2 = np.log(p/(1-p))
           #logit = self.coeffs[0] + sum(self.coeffs[1:-1]*np.array(pred1.iloc[0,:])) + self.coeffs[-1]*pred2
           logit = self.mod_res.get_prediction(pred1).linpred[0]
           #logit = sum(self.coeffs[1:]*np.array(pred1.iloc[0,:]))
           return p*np.exp(logit)
            
    
    def sample_t(self):
        a = self.rng.random()
        ind = np.argmax(self.time_dist>a)
        return (self.time_edges[ind] + self.time_edges[ind+1])/2
    
    def get_time_dist(self,timedata):
        h = np.zeros(len(self.time_edges)-1)
        for data in timedata:
            arr = np.diff(data[:,0])
            hnew,_ = np.histogram(arr,bins=self.time_edges)
            h += hnew
        return np.cumsum(h) / sum(h)

class OneStepSemiPar(Metro_SemiPar):
    #markov step model is used to sample nsamples
    #parametric model (from logistic regression) calculates P(x,y)
    #one sample picked based on P(x,y)
    #evaluate() function is not used
    
    def __init__(self,*args,nsamples=40,ncasecontrol=10,**kwargs):
        super().__init__(*args,**kwargs)
        self.nsamples=nsamples
        self.beta_adjust = np.log(ncasecontrol-1) - np.log(nsamples-1)
       
    def iterate(self,t):
        
        steps = self.markov_model.dist.get_candidates(self.x,self.y,self.nsamples)
        pred1 = self.param_model.get_predictors(t,steps)
        pred1['t'] = t
        logit = self.mod_res.get_prediction(pred1).linpred + self.beta_adjust
        p_rel = 1/(1+np.exp(-logit))#np.exp(logit)
        p_cum = np.cumsum(p_rel)/np.sum(p_rel)
        a = self.rng.random()
        ind = np.argmax(p_cum>a)
        winner = steps[ind]
        self.x = winner[0]
        self.y = winner[1]
        
        #print(p_rel[ind],np.mean(p_rel),min(p_rel),max(p_rel))
        
        return self.x, self.y, np.nan

    def evaluate(self,t,x,y):
        pass





    
def generate(mcmc,t_steps,maxtime=-1.0,sample_direct=False,reset_rand=False,amp_spline=None):

    currt = 0.0

    allrate = []
    allxy = []
    allt = []
    
    mcmc.reset(rand=reset_rand)
    
    for m in mcmc.models:
        m.model_update(currt,mcmc.x,mcmc.y)
    
    for i in range(t_steps):
        
        t = mcmc.sample_t()
            
        #add new time point
        currt += t
        
        if currt > maxtime:
            break
        
        if sample_direct:
           #sample from distribution
           out = mcmc.markov_model.get_candidates(mcmc.markov_model.currx,mcmc.markov_model.curry,1,currt)
           x = out[0][0]
           y = out[0][1]
           rate = np.nan
        else:
           #use Metropolis-Hastings with get_predictors for P
           x,y,rate = mcmc.iterate(currt)
           
        
        allxy.append([x,y])
        allt.append(currt)
        allrate.append(rate)
        
        for m in mcmc.models:
            m.model_update(currt,x,y)
          
        print(i,x,y)
        
    return np.array(allt), np.array(allxy), np.mean(allrate)



if __name__ == '__main__':

    train_set = load_data('waldo_2024', split='train')
    stds = train_set.S_std.numpy().flatten()
    means = train_set.S_mean.numpy().flatten()
    #x_edges = [(x-means[0])/stds[0] for x in [0,1000]]
    #y_edges = [(y-means[1])/stds[1] for y in [0,800]]
    data_extent = (0,1920,0,1080)#tuple(x_edges + y_edges)


    parser = argparse.ArgumentParser()
    parser.add_argument("--samplerate", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--nfix", type=int, default=600)
    parser.add_argument("--maxtime", type=float, default=-1.0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--bound", action="store_true")
    parser.add_argument("--dirsample", action="store_true")
    parser.add_argument("--modeltype", type=str, default="nn")
    parser.add_argument("--ampspline", action="store_true")
    parser.add_argument("--modelname", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    gen_args = parser.parse_args()
     
    #MANUAL OVERRIDES
    gen_args.ampspline = True
    gen_args.dirsample = True
    gen_args.bound = True
    #gen_args.sigma=0.3
    #gen_args.samplerate=20
    gen_args.maxtime = 45.0
    gen_args.start=0
    gen_args.iter=70 #70
    gen_args.modeltype='markov_dxdy'
    gen_args.modelname='_new'
    
    save = True
    
    mcmc_args = {}#{"sr": gen_args.samplerate,
                 #"sig": gen_args.sigma}

    data = [z.detach().numpy() for z in train_set]

    for x in data:
        x[:,1:] = x[:,1:]*stds + means
        
    if gen_args.bound:
        boundstr='_bound'
        extent = data_extent
    else:
        boundstr=''
        extent = (-2.5,2.5,-2.5,2.5)

    if gen_args.ampspline:
       if not gen_args.dirsample:
           raise ValueError('Scaling only supported for direct sample method')
       spl = get_spline(data)
       splstr = '_ampspline'
    else:
       spl = None
       splstr = ''

    kwargs = {'seed':444,
              'sigma_dxdy':30,
              'sigma_xy':1500
              }
    
    mmperpx = 0.2825
    screendist = 1000 #mm
    parargs = {}
    parargs["maxtime_cross"]=11.0
    parargs["maxtime_closest"]=38.0
    parargs["maxtime_visited"]=6.0
    parargs["max_r"] = screendist*np.tan(1.5*np.pi/180)/mmperpx
    
    if gen_args.modeltype=='nn':
        x_dim = 2
                
        model_id = 'jump-16-5'
        nn_model = NNModel(model_id,maxhist=20)
        
        mcmc = Metro_NN(nn_model,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)
        
    
    elif gen_args.modeltype=='markov_dxdy':
        
        #markovdist = Conditional_dXdY(data,data_extent,**kwargs)
        markovdist = Conditional_dXdY(data,data_extent,ampspline=spl,**kwargs)
        markov_model = MarkovModel(markovdist,ampspline=spl)
        mcmc = Metro_SemiPar(markov_model,None,None,data,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)

    elif gen_args.modeltype=='markov_ra':
        markovdist = Unconditional_ThetaR_Indep(data,data_extent,**kwargs)
        markov_model = MarkovModel(markovdist,ampspline=spl)
        mcmc = Metro_SemiPar(markov_model,None,None,data,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)
        
    elif gen_args.modeltype=='markov_raj':
        markovdist = Unconditional_ThetaR_Joint(data,data_extent,**kwargs)
        markov_model = MarkovModel(markovdist,ampspline=spl)
        mcmc = Metro_SemiPar(markov_model,None,None,data,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)

    elif gen_args.modeltype=='semipar':
        
        #ss_df = pd.read_csv('stepselect_waldo_posttimeopt_uniform.csv')
        #ss_df = pd.read_csv("stepselect_waldo_posttimeopt_new_imp_train_r0_2b.csv")
        ss_df = pd.read_csv("stepselect_data/stepselect_waldo2024_train2.csv")
        ss_df['logdist_sq'] = ss_df['logdist']**2
        #ss_df['logdist_cb'] = ss_df['logdist']**3
        #ss_df['log_closest'] = np.log(ss_df['closest_point'])

        #no history dependence
        #cols = ['logdist','logdist_sq','dir_change','dir_change2','cardinality','log_amp_change','centre_dist_init','centre_dist_change']
        #with history dependence
        #cols = ['dist','logdist','logdist_sq','dir_change','dir_change2','cardinality','horiz_align','dist_1back','log_amp_change','centre_dist_init','centre_dist_change','have_visited','meanhistdist','meanhistdist_change','crossings']
        cols = ['dist','logdist','logdist_sq','dir_change','dir_change2','cardinality','horiz_align','log_amp_change','dist_1back','centre_dist_init','centre_dist_change','have_visited','whistdist','whistdist_change','crossings']

        res = stepwise_regression(ss_df,cols,extra=['logdist*t'],thresh=0.001)
         
        #mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols) + ' + logdist*t',ss_df)
        #res = mod.fit()        
        
        #print(res.summary())
        
        #spline_col = 'logdist'
        #res,qs = linspline_regression(ss_df,spline_col,'target ~ cardinality + logdist:t + logdist + crossings',quantiles=[0.25,0.5,0.75])
        #param_model = ParametricModel(data_extent,cols=cols,linspline=spline_col,quantiles=qs)
        #markovdist = Conditional_dXdY(data,data_extent,seed=444)
        #mcmc = Metro_SemiPar(markov_model,param_model,res,data,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)
        

        gen_args.dirsample = False
  
        markovdist = LogNormalDist(data,data_extent,seed=444)
        markov_model = MarkovModel(markovdist)
        
        param_model = ParametricModel(data_extent,cols=cols,**parargs)
        mcmc = OneStepSemiPar(markov_model,param_model,res,data,[extent[0],extent[1]],[extent[2],extent[3]],**mcmc_args)

    else:
        raise ValueError("invalid model type")



    t_steps = gen_args.nfix

       
    for i in range(gen_args.start,gen_args.iter):
       
       mcmc.set_seed(111+i)
           
       if gen_args.dirsample:
          filename='_'.join(['generator',gen_args.modeltype+gen_args.modelname+boundstr+splstr+'_dirsample',str(i)]) + '.csv'           
       elif gen_args.modeltype=='semipar':
          filename='_'.join(['generator',gen_args.modeltype+gen_args.modelname+boundstr,str(i)]) + '.csv'                      
       else:
          filename='_'.join(['generator',gen_args.modeltype+gen_args.modelname+boundstr+splstr,'sr'+str(gen_args.samplerate),'sig'+str(gen_args.sigma),str(i)]) + '.csv'

       times,locs,avrate = generate(mcmc,t_steps,maxtime=gen_args.maxtime,sample_direct=gen_args.dirsample)
       #locs[:,0] = locs[:,0] * stds[0] + means[0]
       #locs[:,1] = locs[:,1] * stds[1] + means[1]
    
    
       data=np.array([times,locs[:,0],locs[:,1]]).T
       df = pd.DataFrame(data,columns=['t','x','y'])
       
       if save:
           filepath = os.path.join('generator_data',filename)
           #with open(filepath,'w') as f:
           #    f.write('Average acceptance rate: ' + str(avrate) + '\n')
           df.to_csv(filepath,index=False)
    
    