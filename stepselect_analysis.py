# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:56:06 2024

@author: aszor
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pymer4.models import Lmer
from scipy import stats
import stepselect
from stepselect import load_data
from scipy.interpolate import splrep, splev
from multiprocessing import Pool
from sklearn.decomposition import PCA
from scipy.optimize import direct


def singleproc_mixedreg(inargs):

    (filename,cols,baseaic,pred,trialtest) = inargs
    df = pd.read_csv(filename) #.iloc[9::15,:]
    df['logdist_sq'] = df['logdist']**2
    df['Trial'] = df.Participant.astype(str) + '_S' + df.Sequence.astype(str)
    for col in cols:
        df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])

    llr = None
    icc = None
    y = None

    
    fixedform = 'target ~ ' + ' + '.join(cols)

    if trialtest:
        #lmer:        
        randomform1 = ' + ('+pred+'-1|Trial)'
        
        logit_mod = Lmer(fixedform+randomform1,
                     data=df, family = 'binomial')
    
        logit_res = logit_mod.fit()
        
        if logit_mod.AIC < baseaic:
            print('Trial random effect significant. Fitting participant...')
    
            randomform2 = ' + ('+pred+'-1|Participant) + ('+pred+'-1|Trial)'
            
            logit2_mod = Lmer(fixedform+randomform2,
                         data=df, family = 'binomial')
    
            logit2_res = logit2_mod.fit()#(control='optimizer="Nelder_Mead"')
        
            if logit2_mod.AIC < logit_mod.AIC:
                print('Participant effect significant.')
        
    
                llr = baseaic-logit2_mod.AIC
                #print(llr[-1])
            
                icc = logit2_mod.ranef_var.Var.iloc[1]/(logit2_mod.ranef_var.Var.iloc[1]+logit2_mod.ranef_var.Var.iloc[0])
                print(icc)
        
                y = logit2_mod.fixef[1][pred]
    
    else:
        #lmer:        
        randomform1 = ' + ('+pred+'-1|Participant)'
        
        logit_mod = Lmer(fixedform+randomform1,
                     data=df, family = 'binomial')
    
        logit_res = logit_mod.fit()

        if logit_mod.AIC < baseaic:
           print('Participant effect significant')

           llr = baseaic-logit_mod.AIC
           y = logit_mod.fixef[pred]
        
    
    
    return llr,icc,y

def multiproc_mixedreg(filename,cols,labels,nproc=6,trialtest=True):


    df = pd.read_csv(filename)
    df['logdist_sq'] = df['logdist']**2

    base_res = run_regression(df, cols)

    basell = base_res.llf
    baseaic = 2*(len(cols)+1)-2*basell 

    randcols = cols + []
    randcols.remove('logdist_sq')
    p= Pool(processes=nproc)#len(randcols))
    res = p.map(singleproc_mixedreg, [(filename,cols,baseaic,pred,trialtest) for pred in randcols])    
    p.terminate()
    p.join()
    
    
    sig_preds = [labels[i] for i,_ in enumerate(res) if res[i][0] is not None]
    sig_res = [x for x in res if x[0] is not None]

    llr,icc,y = zip(*sig_res)
    
    return llr,icc,y,sig_preds
    

def run_mixed_regression(base_df, cols, nn_gain=None, winter=False):
    
    for col in cols:
        base_df[col] = (base_df[col] - np.mean(base_df[col])) / np.std(base_df[col])

    df = base_df
    
    #endog = df.target
    #exog = sm.add_constant(df.loc[:,cols])
    #exog = statsmodels.api.add_constant(df.iloc[:,:-1])
    #cols = ['logdist','cardinality','dir_change','closest_point','crossings','markov_dxdy']
    fixedform = 'target ~ ' + ' + '.join(cols)

    base_res = run_regression(df, cols)

    basell = base_res.llf
    baseaic = 2*(len(cols)+1)-2*basell 
    print('Base AIC:', baseaic)

    allrand = pd.DataFrame()
    llr = []
    icc = []

    df['Trial'] = df.Participant.astype(str) + '_S' + df.Sequence.astype(str)

    for i, pred in enumerate(cols):

        print(pred)

        #lmer:        
        randomform1 = ' + ('+pred+'-1|Trial)'
        
        logit_mod = Lmer(fixedform+randomform1,
                     data=df, family = 'binomial')

        logit_res = logit_mod.fit()

        if logit_mod.AIC < baseaic:
            print('Trial random effect significant. Fitting participant...')
            randomform2 = ' + ('+pred+'-1|Participant) + ('+pred+'-1|Trial)'
            
            logit2_mod = Lmer(fixedform+randomform2,
                         data=df, family = 'binomial')
    
            logit2_res = logit2_mod.fit()#(control='optimizer="Nelder_Mead"')
            
            if logit2_mod.AIC < logit_mod.AIC:
                print('Participant effect significant.')
        
    
                llr.append(baseaic-logit2_mod.AIC)
                #print(llr[-1])
            
                icc.append(logit2_mod.ranef_var.Var.iloc[1]/(logit2_mod.ranef_var.Var.iloc[1]+logit2_mod.ranef_var.Var.iloc[0]))
                print(icc[-1])
        
                y = logit2_mod.fixef[1][pred]
        
                if nn_gain is not None:
                    plt.scatter(nn_gain, y, 80, marker='+', linewidth=2)
                    if winter:
                       plt.scatter(nn_gain[24:], y[24:], 80, marker='+', linewidth=2)
                       plt.legend(["summer","winter"])
    
                    plt.xlabel("NN model loglik gain")
                    plt.ylabel(pred + " coefficient")
                    plt.show()
            
                    res2 = stats.spearmanr(nn_gain, y)
            
                    print(res2.statistic, res2.pvalue)
                allrand[pred] = y


    coefs=np.corrcoefs(np.array(allrand).T)
    plt.figure(figsize=(5,4))
    ax = plt.axes()
    plt.pcolor(np.flipud(coefs),vmin=-1,vmax=1,cmap='RdBu')
    ax.set_yticks(-0.5+np.arange(len(allrand.columns),0,-1),labels=allrand.columns)
    ax.set_xticks(0.5+np.arange(len(allrand.columns)),labels=allrand.columns, rotation=45, ha='right')
    plt.colorbar()
    plt.show()

    if nn_gain is not None:
       endog = np.array(nn_gain)
    else:
       endog = None
    exog = sm.add_constant(allrand)

    plt.bar(0.5+np.arange(len(llr)), llr)
    plt.xticks(0.5+np.arange(len(llr)), labels=allrand.columns, rotation=45, ha='right')
    plt.ylabel('AIC gain with random effects')
    plt.show()


    plt.bar(0.5+np.arange(len(icc)), icc)
    plt.xticks(0.5+np.arange(len(icc)), labels=allrand.columns, rotation=45, ha='right')
    plt.ylabel('Intraclass correlation coefficient')
    plt.show()
    #linear_mod = sm.OLS(endog, exog)
    #res = linear_mod.fit()
    # print(res.summary())

    return endog, exog, llr, icc

def mixed_effects_figs(randeff_file,basic_file,icc,corrmap_title,labels = None):

    randeff_df = pd.read_csv(randeff_file,index_col=0)
    basic_df = pd.read_csv(basic_file)
    
    y_arr = np.array(randeff_df)
    if labels is not None:
        sig_preds = labels
    else:
        sig_preds = randeff_df.columns
    

    total_fix = []
    basic_logdist = []
    par = np.unique(basic_df.Participant)
    for p in par:
       total_fix.append(np.mean(basic_df['Median fixation duration'][basic_df.Participant==p]))
       basic_logdist.append(np.mean(np.log(basic_df['Median saccade distance'][basic_df.Participant==p])))
    
    
    nfix_corr = []
    nfix_p = []
    sig = []
    for i in range(y_arr.shape[1]):
       corr = stats.pearsonr(y_arr[:,i],total_fix)
       #corr = stats.spearmanr(y_arr[:,i],total_fix)
       nfix_corr.append(corr.statistic)
       nfix_p.append(corr.pvalue)
       if corr.pvalue<0.05:
           sig.append(1)
       else:
           sig.append(0)
    print('Fixation time correlations',list(zip(nfix_corr,nfix_p)))

    if randeff_df.columns[0] == 'logdist':
        fig4,axs = plt.subplots(1,2,figsize=(8,3.5))        
        axs[0].scatter(basic_logdist,total_fix)
        axs[0].set_ylabel('Fixation time')
        axs[0].set_xlabel('Log-length (avg)')
        axs[1].scatter(y_arr[:,0],total_fix)
        axs[1].set_yticks([])
        axs[1].set_xlabel('Log-length coefficient')
        corr = stats.pearsonr(basic_logdist,total_fix)
        print('Basic logdist correlation:',corr)

        plt.show()
    else:
        fig4 = None

    pca = PCA()
    pca.fit(y_arr)
    comp1 = pca.components_[0]
    comp2 = pca.components_[1]
    comp3 = pca.components_[2]
    print('PC1 variance',pca.explained_variance_ratio_[0])
    

    
    pc1 = pca.fit_transform(y_arr)[:,0]
    pc2 = pca.fit_transform(y_arr)[:,1]
    nfix_pc1_corr = stats.pearsonr(pc1,total_fix)#np.corrcoef(pc1,total_fix)[0,1]
    print('Fixation time PC1 corr',nfix_pc1_corr.statistic,'p =',nfix_pc1_corr.pvalue)

    if nfix_pc1_corr.statistic<0:
        comp1 = -comp1
        print('Inverting PC1')

    print('PC2 variance',pca.explained_variance_ratio_[1])
    nfix_pc2_corr = stats.pearsonr(pc2,total_fix)#np.corrcoef(pc2,total_fix)[0,1]
    print('Fixation time PC2 corr',nfix_pc2_corr.statistic,'p =',nfix_pc2_corr.pvalue)

    plt.rcParams['hatch.color'] = 'white'  
    plt.rcParams['hatch.linewidth'] = 2
    plt.rcParams['font.size']=14
    pmcols = np.array(['darkred','darkblue','lightgrey','lightgrey'])
    hatches = np.array(['//',''])
    
    w=0.7
    if len(icc)>0:
       fig,axs = plt.subplots(4,1,figsize=(6,7.5))
       skip=0
       axs[0].bar(0.5+np.arange(len(icc)), icc,color='gray',width=w)
       axs[0].set_xticks([])
       axs[0].set_ylim([0,1])
       axs[0].set_ylabel('ICC')
    else:
       fig,axs = plt.subplots(3,1,figsize=(6,6))
       skip=1
       
    axs[1-skip].bar(0.5+np.arange(len(sig_preds)), np.std(y_arr,axis=0),color='gray',width=w)
    axs[1-skip].set_xticks([])
    axs[1-skip].set_ylabel('Std dev.')
    
    inds = (np.array(nfix_corr)>0).astype(int)

    axs[2-skip].bar(0.5+np.arange(len(sig_preds)), nfix_corr,color=pmcols[inds],hatch=hatches[sig],width=w)#,color=pmcols[inds+2*(1-np.array(sig))])
    axs[2-skip].set_xticks([])
    axs[2-skip].set_ylim([-1,1])
    axs[2-skip].grid(axis='y')
    axs[2-skip].set_ylabel(r'$r\,(y,t_\mathrm{fix})$')

    axs[3-skip].bar(0.5+np.arange(len(sig_preds)), comp1, color=pmcols[(np.array(comp1)>0).astype(int)],width=w)
    axs[3-skip].set_ylabel('PC1 loading')
    axs[3-skip].set_ylim([-1,1])
    axs[3-skip].grid(axis='y')
    axs[3-skip].set_xticks(0.5+np.arange(len(sig_preds)), labels=sig_preds, rotation=45, ha='right')

    if len(icc)>0:
       fig.text(-0.02,0.865,'a',fontsize='x-large')
       fig.text(-0.02,0.67,'b',fontsize='x-large')
       fig.text(-0.02,0.475,'c',fontsize='x-large')
       fig.text(-0.02,0.28,'d',fontsize='x-large')

    plt.show()
    
    corrmat = np.corrcoef(y_arr.T)
    fig2 = plt.figure()
    plt.imshow(np.tril(corrmat,k=-1),vmin=-1,vmax=1,cmap='RdBu')
    plt.colorbar()
    plt.xticks(np.arange(len(sig_preds)), labels=sig_preds, rotation=45, ha='right')
    plt.yticks(np.arange(len(sig_preds)), labels=sig_preds, rotation=45, ha='right')
    plt.title(corrmap_title)
    plt.show()
    
    fig3 = plt.figure(figsize=(14,2))
    ax = plt.gca()
    im = ax.pcolor(np.array([comp3,comp2,comp1]), cmap='RdBu', vmin=-
                      1, vmax=1, edgecolors='white', linewidths=5)
    ax.set_yticks([2.5,1.5,0.5],labels=[f'PC1: {str(round(100*pca.explained_variance_ratio_[0]))}%',f'PC2: {str(round(100*pca.explained_variance_ratio_[1]))}%',f'PC3: {str(round(100*pca.explained_variance_ratio_[2]))}%'],fontsize='xx-large')
    ax.set_xticks(0.5+np.arange(len(comp1)),
                     labels=sig_preds, rotation=45, ha='right',fontsize='xx-large')
    ax.set_title(corrmap_title,fontdict={'fontsize':'xx-large'})
    
    for i,x in enumerate(comp1):
        ax.text(i+0.5,2.5,f"{comp1[i]:.2f}",ha='center',va='center',size='x-large',fontweight='bold',color='#555555')
        ax.text(i+0.5,1.5,f"{comp2[i]:.2f}",ha='center',va='center',size='x-large',fontweight='bold',color='#555555')
        ax.text(i+0.5,0.5,f"{comp3[i]:.2f}",ha='center',va='center',size='x-large',fontweight='bold',color='#555555')

    
    return fig, fig2, fig3, fig4, total_fix

def run_regression(df, cols):

    endog = df.target
    exog = sm.add_constant(df.loc[:, cols])
    #exog = statsmodels.api.add_constant(df.iloc[:,:-1])

    logit_mod = sm.Logit(endog, exog)
    logit_res = logit_mod.fit()

    print(logit_res.summary())

    return logit_res



def markov_sigma(pred_df,sig1arr=np.arange(1.0,5.01,1.0),sig2arr=np.arange(0.05,0.251,0.05)):

    cols = [x for x in pred_df.columns if "markov" in x]
    sig = sig1arr
    ds=(sig[1]-sig[0])/2
    sig2 = sig2arr
    ds2=(sig2[1]-sig2[0])/2
    tvals = []
    lls = []
    for col in cols:
        res = run_regression(pred_df,[col])
        tvals.append(res.tvalues[1])
        lls.append(res.llf)
    #plt.plot(sig,lls)
    fig = plt.figure()
    plt.imshow(np.array(lls).reshape([len(sig),len(sig2)]),extent=(sig2[0]-ds2,sig2[-1]+ds2,sig[0]-ds,sig[-1]+ds),origin='lower',aspect='auto')
    f = {'size': 16}
    plt.colorbar(label='log likelihood')
    plt.ylabel('$\sigma_{X,Y}$',fontdict=f)
    plt.xlabel('$\sigma_{dX,dY}$',fontdict=f)
    plt.show()
    #plt.plot(sig,tvals)
    fig2 = plt.figure()        
    plt.imshow(np.array(tvals).reshape([len(sig),len(sig2)]),extent=(sig2[0]-ds2,sig2[-1]+ds2,sig[0]-ds,sig[-1]+ds),origin='lower',aspect='auto')
    plt.colorbar()
    plt.show()
    print(lls)
    
    return fig

    

def time_constants(base_df,title):
    
    
    base_df["logdist_sq"] = base_df.logdist**2
    base_df["dist"] = np.exp(base_df.logdist)
    base_cols = ['dist','logdist','logdist_sq','cardinality','horiz_align','centre_dist_init','centre_dist_change','dist_1back','dir_change','dir_change2','amp_change']#,'delaunay_mean','delaunay_std']


    cols = get_all_interactions(base_cols,extra=['logdist*t'])
    mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
    res = mod.fit()
    print(res.summary())
    base_aic = res.aic


    w = 5
    
    cross_effect = []
    cross_ll = []
    cross_aic = []
    cross_t = list(range(1,46))
    for t in cross_t:
        cols = get_all_interactions(base_cols,extra=['logdist*t','crossings_' + str(t)])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print('Linear AIC gain',base_aic - res.aic)
        cross_aic.append(base_aic - res.aic)

        #cols = get_all_interactions(base_cols,extra=['crossings_' + str(t),'logdist*t'])
        cols = get_all_interactions(base_cols+['crossings_' + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        #cross_effect.append(res.tvalues['crossings_'+ str(t)])
        cross_effect.append(res.tvalues['crossings_'+ str(t)])
        cross_ll.append(res.llr/res.nobs)
    cross_ll_sm = np.convolve(cross_ll, np.ones(w), 'same') / w
    
    
    visited_effect = []
    visited_ll = []
    visited_aic = []
    visited_t = list(range(1,46))
    for t in visited_t:
        cols = get_all_interactions(base_cols,extra=['logdist*t','have_visited_' + str(t)])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print('Linear AIC gain',base_aic - res.aic)
        visited_aic.append(base_aic - res.aic)

        cols = get_all_interactions(base_cols+['have_visited_' + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        visited_effect.append(res.tvalues['have_visited_'+ str(t)])
        visited_ll.append(res.llr/res.nobs)
    visited_ll_sm = np.convolve(visited_ll, np.ones(w), 'same') / w

 

    closest_effect = []
    closest_ll = []
    closest_aic = []
    closest_t = list(range(1,46))
    for t in closest_t:
        cols = get_all_interactions(base_cols,extra=['logdist*t','whistdist_change_' + str(t)])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print('Linear AIC gain',base_aic - res.aic)
        closest_aic.append(base_aic - res.aic)

        cols = get_all_interactions(base_cols+['whistdist_' + str(t),'whistdist_change_' + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        closest_effect.append(res.tvalues['whistdist_change_'+ str(t)])
        closest_ll.append(res.llr/res.nobs)
    closest_ll_sm = np.convolve(closest_ll, np.ones(w), 'same') / w
    
    
    
    plt.plot(visited_t, visited_aic)
    plt.plot(closest_t, closest_aic)
    plt.plot(cross_t, cross_aic)
    plt.title(title+' linear model AIC')
    plt.ylabel('AIC gain')
    plt.xlabel('time constant (s)')
    plt.show()    
    
    plt.plot(visited_t, visited_effect)
    plt.plot(closest_t, closest_effect)
    plt.plot(cross_t, cross_effect)
    plt.title(title+' Effect sizes')
    plt.ylabel('z')
    plt.xlabel('time constant (s)')
    plt.show()


    fig1 = plt.figure()

    ww=int(np.floor(w/2))
    plt.scatter(visited_t, visited_ll,marker='x')
    plt.scatter(closest_t, closest_ll,marker='x')
    plt.scatter(cross_t, cross_ll,marker='x')
    plt.legend(['Visited','Sparsity (inc. change)','Crossings'])
    plt.plot(visited_t[ww:-ww], visited_ll_sm[ww:-ww])
    plt.plot(closest_t[ww:-ww], closest_ll_sm[ww:-ww])
    plt.plot(cross_t[ww:-ww], cross_ll_sm[ww:-ww])
    plt.title(title)
    plt.ylabel('log likelihood ratio')
    plt.xlabel('time constant (s)')
    plt.show()
    
    max_cross = "crossings_"+str(cross_t[np.argmax(cross_ll_sm)])
    max_closest = "whistdist_"+str(closest_t[np.argmax(closest_ll_sm)])+"*whistdist_change_"+str(closest_t[np.argmax(closest_ll_sm)])
    max_visited = "have_visited_"+str(visited_t[np.argmax(visited_ll_sm)])


    fig2 = None
    
    return fig1,fig2
    
    cross_effect2 = []
    cross_ll2 = []
    cross_t = list(range(1,46))
    for t in cross_t:
        cols = get_all_interactions(base_cols + [max_closest,max_visited,"crossings_" + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        cross_effect2.append(res.tvalues["crossings_" + str(t)])
        cross_ll2.append(res.llr/res.nobs)
    
    closest_effect2 = []
    closest_ll2 = []
    closest_t = list(range(1,46))
    for t in closest_t:
        cols = get_all_interactions(base_cols + [max_cross,max_visited,"whistdist_" + str(t),"whistdist_change_" + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        closest_effect2.append(res.tvalues["whistdist_change_" + str(t)])
        closest_ll2.append(res.llr/res.nobs)
    
    
    visited_effect2 = []
    visited_ll2 = []
    visited_t = list(range(1,46))
    for t in visited_t:
        cols = get_all_interactions(base_cols + [max_cross,max_closest,"have_visited_" + str(t)],extra=['logdist*t'])
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
        res = mod.fit()
        print(res.summary())
        visited_effect2.append(res.tvalues["have_visited_" + str(t)])
        visited_ll2.append(res.llr/res.nobs)

    fig2 = plt.figure()
    plt.plot(visited_t, visited_ll2)
    plt.plot(closest_t, closest_ll2)
    plt.plot(cross_t, cross_ll2)
    plt.title('including other maxima')
    plt.ylabel('log likelihood ratio')
    plt.xlabel('time constant (s)')
    plt.legend(['Visited','Sparsity (inc. change)','Crossings'])
    plt.show()
    
    
    
    print(max_cross,max_closest,max_visited)
    
    return fig1, fig2

def heatmaps(dataset,filename,index,datalen,data_extent,**kwargs):

    traindata = load_data(dataset,'train')
    stds = traindata.S_std.numpy().flatten()
    means = traindata.S_mean.numpy().flatten()

    print('using model arguments:',kwargs)
    
    traindata = [data.detach().numpy() for data in traindata]
    for x in traindata:
        x[:,1:] = x[:,1:]*stds + means

    spl = stepselect.get_spline(traindata)

    markovdist = stepselect.Conditional_dXdY(traindata,data_extent,seed=444,sigma_xy=1500,sigma_dxdy=30)
    lognormaldist = stepselect.LogNormalDist(traindata,data_extent,seed=444)

    cols = ['dist','logdist','logdist_sq','cardinality','horiz_align','centre_dist_change','dir_change','dir_change2','log_amp_change','dist_1back','centre_dist_init','have_visited','whistdist','whistdist_change','crossings']
    par_cols = ['logdist','cardinality','horiz_align','centre_dist_change','dir_change','dir_change2','log_amp_change','dist_1back','have_visited','whistdist','whistdist_change','crossings']
    ss_df = pd.read_csv(filename)
    ss_df['logdist_sq'] = ss_df['logdist']**2
    res = stepwise_regression(ss_df,cols,extra=['logdist*t'])#,thresh=0.001)

    models = []

    models.append(stepselect.ParametricModel(data_extent,cols=cols,**kwargs))
    models.append(stepselect.MarkovModel(markovdist,predname="markov_dxdy",ampspline=spl))
    models.append(stepselect.MarkovModel(lognormaldist,predname="markov_par"))

    alldata = load_data(dataset)
    data = alldata[index].detach().numpy()
    data[:,1:] = data[:,1:]*stds + means
    
    for i in range(datalen):        
        [t0,x0,y0] = data[i,:]
        t1 = data[i+1,0]

        for m in models:
            m.model_update(t0,x0,y0)

    extent = data_extent
    N=40
    x = np.linspace(extent[0],extent[1], N) #edges (N)
    y = np.linspace(extent[2],extent[3], N)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    s = np.stack([x[:-1]+dx/2, y[:-1]+dy/2], axis=1) #centres (N-1)

    X, Y = np.meshgrid(s[:, 0], s[:, 1])
    steps = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    newdf = pd.DataFrame(np.array(steps),columns=['x','y'])
    newdf['t'] = t1
    for m in models:
        preds_df = m.get_predictors(t1,steps,logit=False)            
        newdf = pd.concat([newdf,preds_df],axis=1)
        print("model complete")
        
    ncasecontrol = 10
    nsamples = N**2
    beta_adjust = np.log(ncasecontrol-1) - np.log(nsamples-1)
    logit = res.get_prediction(newdf).linpred + beta_adjust
    p_par = newdf['markov_par']/(1+np.exp(-logit))#np.exp(logit)

    newdf['markov_dxdy'] = np.log(newdf['markov_dxdy']/sum(newdf['markov_dxdy']) + 1e-4)
    p_par = np.log(p_par / sum(p_par) + 1e-4)
    
    fig1,axgrid = plt.subplots(3,4,figsize=(7,4))
    axs = axgrid.flatten()
    labels = ['Log length','Cardinality','Horizontal bias','Centre dist change','Forward bias','Parallel bias','Length change','1-back distance','Visited','Sparsity','Sparsity change','Crossings']

    skip=0
    for i in range(len(axs)):
        if i>=(len(newdf.columns)-3+skip) or i<skip:
            axs[i].set_axis_off()
            continue
        else:
            col=par_cols[i-skip]
            if col in ['have_visited','crossings']:
                axs[i].imshow(np.array(newdf[col]).reshape(N-1, N-1),origin='lower',extent=extent,interpolation='none',cmap='inferno')                
            else:
                axs[i].imshow(np.array(newdf[col]).reshape(N-1, N-1),origin='lower',extent=extent,cmap='inferno')
            axs[i].set_title(labels[i-skip],fontsize=9)
        axs[i].plot(data[:datalen,1],data[:datalen,2],color='white')
        axs[i].scatter(data[datalen-1,1],data[datalen-1,2],color='white',marker='x')
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.show()
    
    fig2,axgrid = plt.subplots(3,1,figsize=(4,6))
    axs = axgrid.flatten()      
    axs[0].imshow(np.array(newdf['markov_dxdy']).reshape(N-1, N-1),origin='lower',extent=extent)
    axs[0].annotate('ln$(P_\mathrm{nonpar})$\n\nNonparametric',xy=(1.3, .5), xycoords='axes fraction',rotation=270,horizontalalignment='right',verticalalignment='center',multialignment='center',size=14)
    axs[1].imshow(np.array(p_par).reshape(N-1, N-1),origin='lower',extent=extent)
    axs[1].annotate('ln$(P_\mathrm{par})$\n\nParametric',xy=(1.3, .5), xycoords='axes fraction',rotation=270,horizontalalignment='right',verticalalignment='center',multialignment='center',size=14)
    im=axs[2].imshow(np.array(p_par-newdf['markov_dxdy']).reshape(N-1, N-1),origin='lower',extent=extent,vmin=-3,vmax=3,cmap='RdBu')
    cb = plt.colorbar(im,pad=0.0,fraction=0.2)#,ax=axs[2])
    #cb.set_label('ln $(P_\mathrm{par}/P_\mathrm{nonpar})$',rotation=270,verticalalignment='bottom',size=14)
    axs[2].annotate('ln $(P_\mathrm{par}/P_\mathrm{nonpar})$',xy=(1.3, .5), xycoords='axes fraction',rotation=270,horizontalalignment='right',verticalalignment='center',multialignment='center',size=14)
    for i in range(3):
        axs[i].plot(data[:datalen,1],data[:datalen,2],color='white')
        axs[i].scatter(data[datalen-1,1],data[datalen-1,2],color='white',marker='x')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_anchor('W')
    #axs[3].set_axis_off()       
    plt.show()

    return fig1,fig2,newdf,p_par,data


def nn_hist(total_df):
    lens = [1,5,10,15,20]
    cols = ['nn_diff_test_nohist'] + ['nn_diff_test'+str(k) for k in lens[1:]]
    
    lls = []
    for col in cols:
        res = run_regression(total_df,[col])
        lls.append(res.llr/res.nobs/2/np.log(2))
    
    plt.plot(lens,lls)
    plt.xlabel('history length')
    plt.ylabel('log likelihood gain')
    plt.show()


def compare_synthetic(data_file,synthetic_file,measure,gentitle='generator',extent=None):
    
    pred_data = pd.read_csv(data_file)

    models = []
    models.append(stepselect.ParametricModel(extent,cols=[measure]))

    gendata_df = pd.read_csv(synthetic_file,header=1)
    gendata = np.array(gendata_df)
    pred_gendata = stepselect.step_select_run(gendata,None,models,0) 
    
    inds = pred_data.target==1
    plt.hist(pred_data.crossings[inds],bins=range(0,10),density=True,alpha=0.7)
    inds = pred_gendata.target==1
    plt.hist(pred_gendata.crossings[inds],bins=range(0,10),density=True,alpha=0.7)
    plt.legend(['data',gentitle])
    plt.xlabel(measure)
    plt.ylabel('frequency')
    plt.show()

def time_dependence():
    filename = "stepselect_waldo_posttimeopt_imp2.csv"
    base_df = pd.read_csv(filename)
    base_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    base_df = base_df.dropna()
    total_df = base_df[base_df.target==1]

    sortinds = np.argsort(np.array(total_df.t))
    t = np.array(total_df.t)[sortinds]
    logdist = np.array(total_df.logdist)[sortinds]
    crossings = np.array(total_df.crossings)[sortinds]
    
    x = np.linspace(0,30,50)
    sfactor = 1.0
    
    spl1 = splrep(t,logdist,s=len(t)*np.var(logdist)*sfactor)
    sply1 = splev(x,spl1)
    
    plt.scatter(t,logdist)
    plt.plot(x,sply1,color='black')
    plt.show()

    sply11 = splev(t,spl1)
    spl12 = splrep(t,(logdist-sply11)**2,s=len(t)*np.var((logdist-sply11)**2)*sfactor)
    sply12 = splev(x,spl12)
    
    plt.scatter(t,(logdist-sply11)**2)
    plt.plot(x,sply12,color='black')
    plt.show()


    spl2 = splrep(t,crossings,s=len(t)*np.var(crossings)*sfactor)
    sply2 = splev(x,spl2)
    
    plt.scatter(t,crossings)
    plt.plot(x,sply2,color='black')
    plt.show()

    
    return ([spl1,spl12,spl2], [sply1,sply12,sply2])






def linspline_regression(df, spline_col, formula, quantiles=[0.25,0.5,0.75]):
    #for controlling for step sampling function, from Forrester et al. Ecology 2009
    newcols = []
    qs = np.quantile(df[spline_col],quantiles)
    for i in range(len(quantiles)):
       newcol = spline_col + '_q' + str(i+1)
       newcols.append(newcol)
       df[newcol] = df[spline_col] - qs[i]
       inds = df[newcol]<0
       df[newcol][inds] = 0
    mod = sm.Logit.from_formula(formula + ' + ' + ' + '.join(newcols),sm.add_constant(df))
    res = mod.fit()
    print(res.summary())
    
    return res, qs

def get_all_interactions(cols,extra=[]):
    
    allpreds = cols + []
    for i,col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            if not (('logdist' in col1) and ('logdist' in col2)):
                allpreds.append(col1 + ':' + col2)
    
    allpreds += extra

    return allpreds


def stepwise_regression(df,cols,extra = [],thresh=-1): #0.05):
    #logdist and logdist_sq must come first and second!
    #Set thresh<0 to use AIC, thresh>0 uses p values
    
    ss_df = sm.add_constant(df)
    ss_df = df
    
    allpreds = get_all_interactions(cols,extra=extra)
    print('Starting with',len(allpreds),'terms')
    
    done=False
    oldaic = np.inf
    while True:
        mod = sm.Logit.from_formula('target ~ ' + ' + '.join(allpreds),ss_df)
        #mod = ConditionalLogit.from_formula('target ~ ' + ' + '.join(allpreds),ss_df,groups=ss_df['choice_id'],hasconst=False)
        newres = mod.fit()
        regrows = newres.params.keys()
        inds = np.argsort(list(newres.pvalues))
        
        sig = newres.pvalues<thresh
        print(newres.llf)
        
        if thresh<0 and newres.aic>oldaic:
            #keep previous result
            break
        res = newres
        oldaic = res.aic

        for i in inds[::-1]:
            if ':' not in regrows[i]:
                continue
            if 'logdist:' in regrows[i]:
                #wait until quadratic term is gone
                other = regrows[i].split(':')[1]
                if ('logdist_sq'+':'+other) in regrows:
                    #if sig['logdist_sq'+':'+other]:
                    continue
            
            if thresh>0 and sig.iloc[i]:
                done=True
                break
            else:
                print('Removing ' + regrows[i])
                allpreds.remove(regrows[i])
                break
        if done:
            break
    
    print(res.summary())
    print('Finishing with',len(allpreds),'terms')
    
    return res

def var_importance(filename,title,legend=False):
    ss_df = pd.read_csv(filename)
    ss_df['logdist_sq'] = ss_df['logdist']**2




    cols = ['dist','logdist','logdist_sq','centre_dist_init','cardinality','horiz_align','centre_dist_change','dir_change','dir_change2','log_amp_change','dist_1back','have_visited','whistdist','whistdist_change','crossings']
    labels = ['Centre dist change','Forward bias','Parallel bias','Length change','1-back distance','Visited','Sparsity','Sparsity change','Crossings']
    extras = ['logdist*t']

    #centre variables
    for col in cols:
        ss_df[col] = ss_df[col] - np.nanmean(ss_df[col])
    ss_df.t = ss_df.t - np.nanmean(ss_df.t)
    
    thresh=1.0
    
    res = stepwise_regression(ss_df,cols,extra=extras,thresh=thresh)
    basell = res.aic#res.llf

    #base_res = run_regression(ss_df,cols)
    base_effects = [] #[base_res.tvalues[col] for col in cols[5:]]

    start_ind = len(cols) - len(labels)
    
    llr = []
    llr_int = []
    for col in cols[start_ind:]:
       newcols = cols + []
       newcols.remove(col)
       res = stepwise_regression(ss_df,newcols,extra=extras,thresh=thresh)
       llr.append(res.aic-basell)#(basell - res.llf)
       res = stepwise_regression(ss_df,newcols,extra=extras+[col],thresh=thresh)
       llr_int.append(res.aic-basell)#(basell - res.llf)

       if llr[-1] > llr_int[-1]:
          base_effects.append(res.tvalues[col])
       else:
          base_effects.append(0.0)


       
    #labels = ['Centre dist change','Forward bias','Parallel bias','Length change','Visited','Sparsity','Sparsity change','Crossings']

    fig,axs = plt.subplots(2,1)
    pmcols = np.array(['darkred','darkblue'])
    axs[0].bar(0.5+np.arange(len(llr)), base_effects,color=pmcols[(np.array(base_effects)>0).astype(int)])
    axs[0].set_ylabel('Z score')
    axs[0].set_xticks([])
    axs[0].set_ylim([-24,24])
    axs[0].axhline(y=0, color='grey')
    axs[0].set_title(title)
    axs[1].bar(0.5+np.arange(len(llr)), llr, color='darkgrey')
    axs[1].bar(0.5+np.arange(len(llr)), np.array(llr)-np.array(llr_int), color = 'dimgrey')
    axs[1].set_xticks(0.5+np.arange(len(llr)), labels=labels, rotation=45, ha='right')
    axs[1].set_ylabel('AIC gain')
    if legend:
       axs[1].legend(['Interactions','Linear term'])
    plt.show()
    
    

    return llr,llr_int,base_effects,fig

def degtopx(deg,screendist,mmperpx):
    return np.tan(deg*np.pi/180)*screendist/mmperpx

def pxtodeg(px,screendist,mmperpx):
    return np.arctan(px*mmperpx/screendist)*180/np.pi

def spatial_dist(filenames,titles,mmperpx=-1,screendist=-1):
    #default settings plots in pixels. otherwise converts to degrees
    
    #fig,axs = plt.subplots(1,len(filenames),figsize=(10,4))
    fig,axs = plt.subplots(2,len(filenames))
        
    #cols = ['dist','logdist','logdist_sq','centre_dist_init','cardinality','horiz_align','centre_dist_change','dir_change','dir_change2','log_amp_change','dist_1back','have_visited','whistdist','whistdist_change','crossings']
    cols = ['centre_dist_init','centre_dist_change','whistdist','whistdist_change']
    
    extras = ['logdist*t']
    thresh = 1.0
    
    if mmperpx>0:
       degbins = np.linspace(0,18,31)
       centbins =  degtopx(degbins,screendist,mmperpx)
    else:
       centbins = np.linspace(0,1100,23)
       degbins = centbins
    sparbins = np.linspace(0.2,0.8,31)
    tbins = np.linspace(0,40,41)
    tlen = 5.0
    
    assert len(degbins)%2==1
    assert len(sparbins)%2==1
    
    dc = centbins[1]-centbins[0]
    dd = degbins[1]-degbins[0]
    ds = sparbins[1]-sparbins[0]

    scaling = 1.0
    c, s = np.meshgrid(degbins[1::2], sparbins[1::2])

    if mmperpx>0:
        dd_dc = mmperpx/screendist*180/np.pi / (1+np.tan(c*np.pi/180)**2)
    else:
        dd_dc = 1.0

    for j in range(len(filenames)):
        ss_df = pd.read_csv(filenames[j])
        ss_df['logdist_sq'] = ss_df['logdist']**2
    

        h = np.histogram2d(ss_df.centre_dist_init[ss_df.target==1],ss_df.whistdist[ss_df.target==1]-ss_df.whistdist_change[ss_df.target==1],bins=[centbins,sparbins],density=True)
        h1,_,_ = np.histogram2d(ss_df.centre_dist_init[ss_df.target==1],ss_df.whistdist[ss_df.target==1]-ss_df.whistdist_change[ss_df.target==1],bins=[centbins[::2],sparbins[::2]],density=True)
        axs[0,j].imshow(h[0].T,extent =(degbins[0],degbins[-1],sparbins[0],sparbins[-1]),origin='lower',aspect='auto',interpolation='gaussian',filterrad=1)
        #plt.colorbar()
        centmeans = []
        sparmeans = []
        for i in range(len(tbins)-1):
            inds = (ss_df.target==1)&(ss_df.t>tbins[i])&(ss_df.t<tbins[i]+tlen)
            if mmperpx>0:
               centmeans.append(pxtodeg(np.median(ss_df.centre_dist_init[inds]),screendist,mmperpx))
            else:   
               centmeans.append(np.median(ss_df.centre_dist_init[inds]))
            sparmeans.append(np.median(ss_df.whistdist[inds]-ss_df.whistdist_change[inds]))
        axs[0,j].plot(centmeans,sparmeans,color='white')
        axs[0,j].scatter(centmeans[:-1:5],sparmeans[:-1:5],marker='o',s=10,color='white',zorder=1)
        axs[0,j].scatter(centmeans[-1],sparmeans[-1],marker='x',s=30,color='red',zorder=2)
        if mmperpx>0:
            axs[1,j].set_xlabel('Distance from center (degrees)')
        else:
            axs[1,j].set_xlabel('Distance from center (pixels)')
        axs[0,j].set_xticks([])
        if j==0:
            axs[0,j].set_ylabel('Sparsity')
            axs[1,j].set_ylabel('Sparsity')
        else:
            axs[0,j].set_yticks([])
            axs[1,j].set_yticks([])
        axs[0,j].set_title(titles[j])

        res = stepwise_regression(ss_df,cols,extra=extras,thresh=thresh)
        
        shade = 0.9*(1 - h1.T.flatten()/max(h1.flatten()))
        #grad_c = scaling * dc**2 * (res.params['centre_dist_change']+c*res.params['centre_dist_init:centre_dist_change']+s*res.params['centre_dist_change:whistdist'])
        #grad_s = scaling * ds**2 * (res.params['whistdist'] + res.params['whistdist_change'] + c*res.params['centre_dist_init:whistdist'] + c*res.params['centre_dist_init:whistdist_change'] + s*res.params['whistdist:whistdist_change'])
        grad_c = scaling * dc * dd * (res.params['centre_dist_change']+c/dd_dc*res.params['centre_dist_init:centre_dist_change']+s*res.params['centre_dist_change:whistdist'])
        grad_s = scaling * ds**2 * (res.params['whistdist'] + res.params['whistdist_change'] + c/dd_dc*(res.params['centre_dist_init:whistdist'] + res.params['centre_dist_init:whistdist_change']) + s*res.params['whistdist:whistdist_change'])
        axs[1,j].quiver(c,s,grad_c,grad_s,angles='xy',color=[(x,x,x) for x in shade])
        
        fig.text(0.02,0.91,'a',fontsize='large')
        fig.text(0.02,0.44,'b',fontsize='large')

    plt.show()    
    
    return fig
    
def compare_var_importance(base_effects1,base_effects2):  


    labels = ['Centre dist change','Forward bias','Parallel bias','Length change','1-back distance','Visited','Sparsity','Sparsity change','Crossings']
    all_effects = np.array([base_effects1,base_effects2])

    fig4 = plt.figure(figsize=(14,2))
    ax = plt.gca()
    im = ax.pcolor(np.flipud(all_effects), cmap='RdBu', vmin=-
                      30, vmax=30, edgecolors='white', linewidths=5)
    ax.set_yticks([0.5,1.5],labels=['Random pixels','Waldo'],fontsize='xx-large')
    ax.set_xticks(0.5+np.arange(len(base_effects1)),
                     labels=labels, rotation=45, ha='right',fontsize='xx-large')
    
    for i,x in np.ndenumerate(all_effects):
        ax.text(i[1]+0.5,1.5-i[0],f"{x:.1f}",ha='center',va='center',size='xx-large',fontweight='bold',color='#333333')
        
        
    #cbar = plt.colorbar(im, ax=ax,fraction=0.2,aspect=10,pad=0.02)
    #cbar.ax.set_ylabel('Effect size z\n',fontsize='x-large',rotation=270,labelpad=15.0)
    text=fig4.text(0.01,0.85,'c',fontsize=32)
    
    return fig4

def timeopt_func(x,*args):
    base_cols = ['dist','logdist','logdist_sq','cardinality','horiz_align','centre_dist_init','centre_dist_change','dist_1back','dir_change','dir_change2','amp_change']#,'delaunay_mean','delaunay_std']

    base_df = args[0]
    #if np.any(np.round(x)<1) or np.any(np.round(x)>45):
    #    print(x,0.0)
    #    return 0.0
    t1 = x[0]
    t2 = x[1]
    t3 = x[2]
    cols = get_all_interactions(base_cols+['crossings_' + str(round(t1)),'whistdist_' + str(round(t2)),'whistdist_change_' + str(round(t2)),'have_visited_' + str(round(t3))],extra=['logdist*t'])
    mod = sm.Logit.from_formula('target ~ ' + ' + '.join(cols),sm.add_constant(base_df))
    res = mod.fit()

    print(x,-res.llr/res.nobs)
    return -res.llr/res.nobs
        

def timeopt_auto(base_df):
    base_df["logdist_sq"] = base_df.logdist**2
    base_df["dist"] = np.exp(base_df.logdist)
    bounds = [(0.51,45.5),(0.51,45.5),(0.51,45.5)]
    #args = {'args':base_df}
    #ans = basinhopping(timeopt_func,[23,23,23],stepsize=4.0,minimizer_kwargs = args)
    ans = direct(timeopt_func,bounds,args=(base_df,),vol_tol=1e-5)#,eps=1.0)
    
    print(ans.x,ans.fun)
    #base_cols = ['dir_change','dir_change2','amp_change']


def compare_par_nonpar(filename,test_filename):
    
    cols = ['dist','logdist','logdist_sq','cardinality','horiz_align','centre_dist_change','dir_change','dir_change2','log_amp_change','dist_1back','centre_dist_init','have_visited','whistdist','whistdist_change','crossings']
    ss_df = pd.read_csv(filename)
    ss_df['logdist_sq'] = ss_df['logdist']**2
    res = stepwise_regression(ss_df,cols,extra=['logdist*t'])

    mod = sm.Logit.from_formula('target ~ markov_dxdy',sm.add_constant(ss_df))
    res2 = mod.fit()
    print(res2.summary())
    
    test_df = pd.read_csv(test_filename)
    test_df['logdist_sq'] = test_df['logdist']**2

    pred = res.get_prediction(test_df).predicted
    pred2 = res2.get_prediction(test_df).predicted

    ll = sum(-np.log(pred)*test_df.target-np.log(1-pred)*(1-test_df.target)) / len(pred)
    ll2 = sum(-np.log(pred2)*test_df.target-np.log(1-pred2)*(1-test_df.target)) / len(pred2)


if __name__ == "__main__":
    pass