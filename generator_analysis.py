# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:17:35 2024

@author: aszor
"""

import numpy as np
import os

import pandas as pd
from cycler import cycler

from scipy.stats import bootstrap, norm
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import seaborn as sns
from stepselect import load_data, ParametricModel

    
    

def spatial_stats(x,y,t,gridsize=50,tstep=2):
    
    gridx = np.arange(0,1001,gridsize)
    gridy = np.arange(0,801,gridsize)
    gridt = np.arange(0,30.1,tstep)
    visitsites = np.zeros([len(gridx)-1,len(gridy)-1])
    
    visitsum = np.zeros(len(gridt))
    for i in range(len(x)):
        if np.isfinite(x[i]):
            xloc = np.argmax(gridx>x[i])
            yloc = np.argmax(gridy>y[i])
            ind = np.argmax(gridt>t[i])
            if xloc>0 and yloc>0 and ind>0:
                if visitsites[xloc-1,yloc-1]==0:
                   visitsites[xloc-1,yloc-1] = 1
                   visitsum[ind:] += 1
                
    
    nsites = (len(gridx)-1)*(len(gridy)-1)
    empsites = np.array(visitsum)/nsites
    
    return empsites


def crossing_comparison(datadir,cats,dataset,leg=None,normalized=True,maxt=45.0,extent=None,mmperpx=-1,screendist=-1,**mod_args):
    
    colors = list(sns.color_palette("Dark2").as_hex())
    #colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17']
    cyc = cycler(color=colors[5:1:-1])
    
    train_set = load_data(dataset, split='train')
    if normalized:
       stds = train_set.S_std.numpy().flatten()
       means = train_set.S_mean.numpy().flatten()
    else:
       stds = np.array([1.0,1.0])
       means = np.array([0.0,0.0])
    #alldata = train_set + load_data('waldo_short', split='test') + load_data('waldo_short', split='val')
    testdata = load_data(dataset, split='test')
    valdata = load_data(dataset, split='val')
    alldata = testdata + valdata
    allIDs = testdata.IDs + valdata.IDs
    
    print('model parameters:',mod_args)
    if mmperpx > 0:
        mod = ParametricModel(extent,cols=['logdist','crossings','crossing_r','dist_deg'],mmperpx=mmperpx,screendist=screendist,**mod_args)
        plot_degrees = True
        print("Using degree units")
    else:
        mod = ParametricModel(extent,cols=['logdist','crossings','crossing_r'],**mod_args)
        plot_degrees = False
        print("Using pixel units")
        
    mint = mod.maxtime_cross
    
    files = os.listdir(datadir)
    all_crossings = []
    all_logdist = []
    all_logdist_px = []
    all_dist = []
    all_t = []
    all_spatial = []
    all_spatial_std = []
    all_r1 = []
    all_r2 = []
    cross_per_trial = []
    t_per_trial = []
    cat_per_trial = []
    n_trials = []
    cat_spatial = []
    trial_spatial = []
    trial_cross = []
    
    n_trials.append(0)
    all_crossings.append([])
    all_logdist.append([])
    all_logdist_px.append([])
    all_dist.append([])
    all_t.append([])
    all_r1.append([])
    all_r2.append([])

    
    i=0
    for ii,data in enumerate(alldata):
        print(allIDs[ii])
        data = data.cpu().numpy()
        df = pd.DataFrame(data,columns=['t','x','y'])
        df.x = means[0]+stds[0]*df.x
        df.y = means[1]+stds[1]*df.y
        n_trials[i] += 1
        cross_per_trial.append([])
        t_per_trial.append([])
        cat_per_trial.append(0)
        mod.reset()


        cat_spatial.append(spatial_stats(means[0]+stds[0]*data[:,1],means[1]+stds[1]*data[:,2],data[:,0]))
        trial_spatial.append(cat_spatial[-1][-1])
        
        for j in range(1,len(df.t)):
            mod.model_update(df.t[j-1],df.x[j-1],df.y[j-1])
            if j<2:
                continue
            if df.t[j]>maxt:
                break
            pred = mod.get_predictors(df.t[j],[[df.x[j],df.y[j]]])
            all_crossings[i].append(pred.crossings[0])
            all_logdist_px[i].append(pred.logdist[0])
            if plot_degrees:
               all_logdist[i].append(np.log(pred.dist_deg[0]))
               all_dist[i].append(pred.dist_deg[0])                
            else:
               all_logdist[i].append(pred.logdist[0])
               all_dist[i].append(np.exp(pred.logdist[0]))
            r1 = pred.crossing_r1[0]
            r2 = pred.crossing_r2[0]
            all_r1[i] += r1 #([np.exp(pred.logdist[0]) for x in r2])
            all_r2[i] += r2
            all_t[i].append(df.t[j])
            cross_per_trial[-1].append(pred.crossings[0])
            t_per_trial[-1].append(df.t[j])
        
        trial_cross.append(np.mean(cross_per_trial[-1]))
    
    
    
    for ii,matchstr in enumerate(cats):
        i = ii+1
        n_trials.append(0)
        all_crossings.append([])
        all_logdist.append([])
        all_logdist_px.append([])
        all_dist.append([])
        all_t.append([])
        all_r1.append([])
        all_r2.append([])
        cat_spatial = []
        for file in [f for f in files if matchstr in f]:
            n_trials[i] += 1
            cross_per_trial.append([])
            t_per_trial.append([])
            cat_per_trial.append(i)
            print(file)
            mod.reset()
            df = pd.read_csv(os.path.join(datadir,file))#,header=1)
            cat_spatial.append(spatial_stats(df.x,df.y,df.t))
            for j in range(1,len(df.t)):
                mod.model_update(df.t[j-1],df.x[j-1],df.y[j-1])
                if j<2:
                    continue
                if df.t[j]>maxt:
                    break
                pred = mod.get_predictors(df.t[j],[[df.x[j],df.y[j]]])
                all_crossings[i].append(pred.crossings[0])
                all_logdist_px[i].append(pred.logdist[0])
                if plot_degrees:
                   all_logdist[i].append(np.log(pred.dist_deg[0]))
                   all_dist[i].append(pred.dist_deg[0])                
                else:
                   all_logdist[i].append(pred.logdist[0])
                   all_dist[i].append(np.exp(pred.logdist[0]))
                r1 = pred.crossing_r1[0]
                r2 = pred.crossing_r2[0]
                all_r1[i] += r1 #([np.exp(pred.logdist[0]) for x in r2])
                all_r2[i] += r2
                all_t[i].append(df.t[j])
                cross_per_trial[-1].append(pred.crossings[0])
                t_per_trial[-1].append(df.t[j])
        all_spatial.append(np.mean(np.array(cat_spatial),axis=0))    
        all_spatial_std.append(np.std(np.array(cat_spatial),axis=0))    


    


    
    print(n_trials)
    all_spatial = [np.mean(np.array(cat_spatial),axis=0)] + all_spatial
    all_spatial_std = [np.std(np.array(cat_spatial),axis=0)] + all_spatial_std
    
    
    
    print('Total saccades:',[len(x) for x in all_crossings])
    print('Total crossings:',[len(x) for x in all_r1])
    print('Mean crossings per saccade:',[np.mean(x) for x in all_crossings])
    print('Median saccade length:',[np.median(np.exp(x)) for x in all_logdist])
    print('Lower IQR saccade length:',[np.quantile(np.exp(x),0.25) for x in all_logdist])
    print('Upper IQR saccade length:',[np.quantile(np.exp(x),0.75) for x in all_logdist])
    print('Median crossing saccade length:',[np.median(x) for x in all_r1])
    print('Median crossed saccade length:',[np.median(x) for x in all_r2])
    print('Ratio crossed/all:',[np.median(all_r2[i])/np.median(np.exp(all_logdist[i])) for i in range(len(all_logdist))])
    
    if leg is None:
       leg = ['data'] + cats
    
    for i,y in enumerate(all_spatial):
       plt.errorbar(np.arange(0,30.1,2),y,yerr=all_spatial_std[i],capsize=4)
    plt.xlabel('time (s)')
    plt.ylabel('mean spatial coverage')  
    plt.legend(leg)
    plt.show()
    
    
    plt.hist(all_crossings,bins=np.arange(-0.5,9.0,1.0),histtype='bar',density=True)
    plt.legend(leg)
    plt.xlabel('# crossings')
    plt.ylabel('density')
    plt.show()
    
    
    plt.scatter(trial_cross,trial_spatial)
    plt.xlabel('Mean crossings')
    plt.ylabel('Spatial coverage')
    #print(np.corrcoef(trial_cross,trial_spatial))
    plt.show()
    
    
    allh = []
    mind = 1
    maxd = 8
    maxcross = 6 #10
    dist_bins = 14 #28
    
    for i,label in enumerate(leg):
        inds = np.array(all_t[i])>mint
        h,_,_ = np.histogram2d(np.array(all_crossings[i])[inds],np.array(all_logdist[i])[inds],bins=[np.arange(-0.5,maxcross,1.0),np.linspace(mind,maxd,dist_bins+1)])
        h_norm = h/h.sum()#axis=1,keepdims=True)
        allh.append(h_norm)
        #plt.imshow(h_norm,origin='lower',extent=(mind,maxd,0,maxcross),aspect=0.75)
        plt.imshow(h,origin='lower',extent=(mind,maxd,0,maxcross),aspect=0.75)
        plt.colorbar()
        plt.xlabel('log (saccade length)')
        plt.ylabel('# crossings')
        plt.title(label)
        plt.show()
    
    if plot_degrees:
       r_bins = np.linspace(0,18,25)
    else:
       r_bins = np.linspace(0,1100,23)
    distx = r_bins[:-1] + (r_bins[1]-r_bins[0])/2
    
    all_disth = []
    all_r1h = []
    all_r2h = []
    all_rh = []
    for i,label in enumerate(leg):
        
        dist_h,_ = np.histogram(all_dist[i],bins=r_bins)
        r2h,_ = np.histogram(all_r2[i],bins=r_bins)
        all_r2h.append(r2h)
        r1h,_ = np.histogram(all_r1[i],bins=r_bins)
        all_r1h.append(r1h)
        rh,_ = np.histogram(all_r1[i] + all_r2[i],bins=r_bins)
        all_rh.append(rh)
        dist_h = dist_h.astype(float) + 1e-5
        all_disth.append(dist_h)
        h,_,_ = np.histogram2d(all_r2[i],all_r1[i],bins=[r_bins,r_bins])
        h = ((h/dist_h).T/dist_h).T
        h = gaussian_filter(h,1.5)
        h = h / sum(sum(h))
        plt.imshow(h,origin='lower',extent=(r_bins[0],r_bins[-1],r_bins[0],r_bins[-1]),aspect=1.0,vmin=0,vmax=0.02)
        plt.colorbar()
        plt.xlabel('current saccade')
        plt.ylabel('past saccade')
        plt.title(label)
        plt.show()
        
        plt.plot(r_bins[1:],np.log(dist_h))
        plt.show()
        
    r_bins_fine = np.linspace(0,r_bins[-1],50)
    distx_fine = r_bins_fine[:-1] + (r_bins_fine[1]-r_bins_fine[0])/2
    disth_fine,_ = np.histogram(all_dist[0],bins=r_bins_fine)

    rh_fine,_ = np.histogram(all_r1[0],bins=r_bins_fine)


    if plot_degrees:
        hist_div = 1.5 
    else:
        hist_div = 50.0
        
    
    fig77 = plt.figure(figsize=(4.5,3))
    fig77.gca().set_prop_cycle(cyc)
    for i,label in enumerate(leg):
       if i>1:
           marker=':'
       else:
           marker='-'
       plt.plot(distx,all_r1h[i]/all_disth[i]/distx,marker)#/len(all_crossings[i]),marker)
    fig77.gca().set_prop_cycle(cyc)
    #for i,label in enumerate(leg):
        #quartiles = np.quantile(np.exp(all_logdist[i]),[0.5,0.25,0.75])
        #plt.errorbar(quartiles[0],0.0008-0.0002*i,xerr=np.array([abs(quartiles[1:]-quartiles[0])]).T,fmt='P',alpha=0.7)
    plt.fill_between([0] + list(distx_fine),[0] + list(disth_fine/sum(disth_fine)/hist_div),color='lightgrey')
    plt.plot([0] + list(distx_fine),[0] + list(rh_fine/sum(rh_fine)/hist_div),color='darkgrey',linewidth=1)    
    if plot_degrees:
        plt.xlabel('Crossing saccade length (degrees)')
        plt.ylabel('# crossings per degree')
    else:
        plt.xlabel('Crossing saccade length (pixels)')
        plt.ylabel('# crossings per pixel')
        #plt.ylim([0,0.008])
        #plt.yticks([0,0.002,0.004,0.006,0.008])
    plt.xlim([0,distx[-1]])
    plt.legend(leg)
    plt.show()


    rh_fine,_ = np.histogram(all_r2[0],bins=r_bins_fine)

    fig99 = plt.figure(figsize=(4.5,3))
    fig99.gca().set_prop_cycle(cyc)
    for i,label in enumerate(leg):
       if i>1:
           marker=':'
       else:
           marker='-'
       plt.plot(distx,all_r2h[i]/all_disth[i]/distx,marker)#/len(all_crossings[i]),marker)
    fig99.gca().set_prop_cycle(cyc)
    #for i,label in enumerate(leg):
        #quartiles = np.quantile(np.exp(all_logdist[i]),[0.5,0.25,0.75])
        #plt.errorbar(quartiles[0],0.0008-0.0002*i,xerr=np.array([abs(quartiles[1:]-quartiles[0])]).T,fmt='P',alpha=0.7)
    plt.fill_between([0] + list(distx_fine),[0] + list(disth_fine/sum(disth_fine)/hist_div),color='lightgrey')
    plt.plot([0] + list(distx_fine),[0] + list(rh_fine/sum(rh_fine)/hist_div),color='darkgrey',linewidth=1)    
    if plot_degrees:
        plt.xlabel('Crossed saccade length (degrees)')
        plt.ylabel('# crossings per degree')
    else:
        plt.xlabel('Crossed saccade length (pixels)')
        plt.ylabel('# crossings per pixel')
        #plt.ylim([0,0.008])
        #plt.yticks([0,0.002,0.004,0.006,0.008])
    plt.xlim([0,distx[-1]])
    #plt.legend(leg)    
    plt.show()

    rh_fine,_ = np.histogram(all_r1[0] + all_r2[0],bins=r_bins_fine)
    

    fig88 = plt.figure(figsize=(4.5,3))
    fig88.gca().set_prop_cycle(cyc)
    for i,label in enumerate(leg):
       if i>1:
           marker=':'
       else:
           marker='-'
       plt.plot(distx,all_rh[i]/all_disth[i]/distx,marker)#/len(all_crossings[i]),marker)
    fig88.gca().set_prop_cycle(cyc)
    #quartiles = np.quantile(np.exp(all_logdist[0]),[0.5,0.25,0.75])
    #plt.errorbar(quartiles[0],0.0002,xerr=np.array([abs(quartiles[1:]-quartiles[0])]).T,fmt='P',alpha=0.7,color='darkgrey')
    plt.fill_between([0] + list(distx_fine),[0] + list(disth_fine/sum(disth_fine)/hist_div),color='lightgrey')
    plt.plot([0] + list(distx_fine),[0] + list(rh_fine/sum(rh_fine)/hist_div),color='darkgrey',linewidth=1)    
    if plot_degrees:
        plt.xlabel('Saccade length (degrees)')
        plt.ylabel('# crossings per degree')
    else:
        plt.xlabel('Saccade length (pixels)')
        plt.ylabel('# crossings per pixel')
        #plt.ylim([0,0.015])
        #plt.yticks([0,0.005,0.01])
    plt.xlim([0,distx[-1]])
    plt.legend(leg,frameon=False)
    fig88.text(0.01,0.9,'a',fontsize='x-large')
    plt.show()
    

        
    fig = plt.figure(figsize=(4,3))
    ax1 = fig.add_axes((0.15,0.15,0.1,0.6))
    ax1.set_prop_cycle(cyc)
    for x in all_crossings[:2]:
       ax1.hist(x,bins=np.arange(-0.5,maxcross,1.0),alpha=0.5,orientation='horizontal',density=True)
    ax1.set_ylabel('# crossings')
    ax1.set_xticks([])
    ax1.set_ylim([-0.5,maxcross-0.5])
    ax1.xaxis.set_inverted(True)
    ax2 = fig.add_axes((0.3,0.15,0.5,0.6))
    ax2.set_prop_cycle(cyc)
    #hand = ax2.imshow(allh[0]-allh[1],vmin=-0.4,vmax=0.4,cmap='RdBu',origin='lower',extent=(mind,maxd,-0.5,maxcross-0.5),aspect='auto')
    hand = ax2.imshow(allh[0]-allh[1],cmap='RdBu',origin='lower',extent=(mind,maxd,-0.5,maxcross-0.5),aspect='auto')
    ax2.set_xlabel('log (saccade dist)')
    ax2.set_ylim([-0.5,maxcross-0.5])
    ax2.set_yticks([])
    ax3 = fig.add_axes((0.85,0.15,0.04,0.6))
    ax3.set_prop_cycle(cyc)
    fig.colorbar(hand,cax=ax3)
    ax4 = fig.add_axes((0.3,0.8,0.5,0.1))
    ax4.set_prop_cycle(cyc)
    for x in all_logdist[:2]:
       ax4.hist(x,bins=np.linspace(mind,maxd,dist_bins+1),alpha=0.5,density=True)
    ax4.set_xlim([mind,maxd])
    ax4.set_xticks([])
    ax4.set_yticks([])

    fig.text(0.01,0.9,'a',fontsize='x-large')
    #plt.title('data-markov model difference')
    plt.show()
    
    fig9=plt.figure()
    fig9.gca().set_prop_cycle(cyc)
    x = np.linspace(mind,maxd,dist_bins+1)
    dx = x[1]-x[0]
    x = x[:-1] + dx/2

    inds = np.array(all_t[0])>mint
    mu_data = np.mean(np.array(all_logdist[0])[inds])
    std_data = np.std(np.array(all_logdist[0])[inds])
    k_data = np.mean(np.array(all_crossings[0])[inds])*np.exp(-mu_data-std_data**2/2)
    
    inds = np.array(all_t[1])>mint
    mu_markov = np.mean(np.array(all_logdist[1])[inds])
    std_markov = np.std(np.array(all_logdist[1])[inds])
    k_markov = np.mean(np.array(all_crossings[1])[inds])*np.exp(-mu_markov-std_markov**2/2)

    plt.plot(x,allh[0][0,:],':')
    plt.plot(x,allh[1][0,:],':')
    fig9.gca().set_prop_cycle(cyc)
    cross = np.arange(1,maxcross).reshape(-1,1)
    plt.plot(x,np.sum(allh[0][1:,:]*cross,axis=0),'+')
    plt.plot(x,np.sum(allh[1][1:,:]*cross,axis=0),'+')
    fig9.gca().set_prop_cycle(cyc)
    plt.plot(x,dx*k_data*np.exp(x)*norm.pdf(x, loc=mu_data, scale=std_data))
    plt.plot(x,dx*k_markov*np.exp(x)*norm.pdf(x, loc=mu_markov, scale=std_markov))
    plt.xlabel('log length')
    plt.ylabel('Probability')
    plt.legend(['data no crossing','markov no crossing','data crossing','markov crossing','length proportional fit','length proportional fit'])
    plt.show()


    fig8=plt.figure()
    fig8.gca().set_prop_cycle(cyc)
    cross = np.arange(1,maxcross).reshape(-1,1)
    plt.plot(x,np.sum(allh[0][1:,:]*cross,axis=0)/np.sum(allh[0],axis=0)/np.exp(x))
    plt.plot(x,np.sum(allh[1][1:,:]*cross,axis=0)/np.sum(allh[1],axis=0)/np.exp(x))
    plt.xlabel('log length')
    plt.ylabel('Probability per pixel')
    plt.show()
       
    fig2 = plt.figure()
    fig2.gca().set_prop_cycle(cyc)
    x = np.linspace(0,30,50)
    #sfactor = 1.0

    t_bins = np.arange(0,maxt+1,3)
    t_cent = t_bins[:-1] + (t_bins[1]-t_bins[0])/2
    
    #sortinds = []
    #all_t_orig = []
    #for i,t in enumerate(all_t):
    #    sortinds.append(np.argsort(t))
    #    all_t_orig.append(all_t[i])
    #    all_t[i] = np.array(all_t[i])[sortinds[-1]]

    hands = []
    for i,y in enumerate(all_logdist_px):
        logdist = np.array(y)#[sortinds[i]]
        #spl = splrep(all_t[i],logdist,s=len(y)*np.var(logdist)*sfactor)
        #sply = splev(x,spl)
        
        allq = []
        for j in range(len(t_bins)-1):
            inds = np.bitwise_and((all_t[i]>=t_bins[j]),(all_t[i]<t_bins[j+1]))
            q = np.quantile(logdist[inds],[0.5,0.25,0.75])
            allq.append(list(q))
            
        allq = np.array(allq).T
        ymed = allq[0,:]
        ylims=allq[1:,:]
        if i==0:
           plt.fill_between(t_cent,ylims[0,:],ylims[1,:],alpha=0.5) 
           hand, = plt.plot(t_cent,ymed)
           hands.append(hand)
        else:
           hand = plt.errorbar(t_cent,ymed,yerr=abs(ylims-ymed),fmt='P',capsize=4)
           hands.append(hand[0])
    plt.xlabel('time (s)')
    plt.ylabel('log (saccade length)')
    plt.legend(hands,leg)
    plt.show()
    
    allcum = []
    allcum_std = []
    
    for i,y in enumerate(all_crossings):
        crossings = np.array(y)#[sortinds[i]]
        #spl = splrep(all_t[i],crossings,s=len(y)*np.var(crossings)*sfactor)
        #sply = splev(x,spl)
        allmedian = []
        alllow = []
        allhigh = []
        allcum.append([])
        allcum_std.append([])
        for j in range(len(t_bins)-1):
            #per saccade
            inds = np.bitwise_and((all_t[i]>=t_bins[j]),(all_t[i]<t_bins[j+1]))
            res = bootstrap((crossings[inds],),np.mean)
            allmedian.append(np.median(res.bootstrap_distribution))
            alllow.append(res.confidence_interval[0])
            allhigh.append(res.confidence_interval[1])
            #cumulative per trial
            #inds = all_t[i]<t_bins[j+1]
            #allcum[i].append(sum(crossings[inds])/n_trials[i])
            curr_cross = []
            for k in range(len(cross_per_trial)):
               if cat_per_trial[k] == i:
                  inds = np.array(t_per_trial[k])<t_bins[j+1]
                  curr_cross.append(sum(np.array(cross_per_trial[k])[inds]))
            allcum[i].append(np.mean(curr_cross))
            allcum_std[i].append(np.std(curr_cross))#/np.sqrt(n_trials[i]))

            
        conf = np.array([alllow,allhigh])    
        if i>1:
            marker='P'
        else:
            marker='-P'
        plt.errorbar(t_cent,allmedian,yerr=abs(conf-np.array(allmedian)),fmt=marker,capsize=4)
    plt.xlabel('time (s)')
    plt.ylabel('mean # crossings')
    plt.legend(leg)
    plt.show()
    
    fig3 = plt.figure(figsize=(4.5,3))
    fig3.gca().set_prop_cycle(cyc)
    for i in range(len(allcum)):
        if i>1:
            marker=':P'
        else:
            marker='-P'
        plt.errorbar(t_bins[1:],allcum[i],yerr=allcum_std[i],fmt=marker,capsize=4)
    #plt.legend(leg,frameon=False,fontsize='medium')
    plt.xlabel('time (s)')
    plt.ylabel('mean cumulative crossings')
    fig3.text(0.01,0.9,'b',fontsize='x-large')
    plt.show()

    return fig88, fig77, fig99, fig2, fig3, all_logdist, all_t


if __name__ == '__main__':
    
    kwargs = {}
    kwargs['extent'] = (0,1920,0,1080)
    kwargs['mmperpx'] = 0.2825
    kwargs['screendist'] = 1000
    kwargs['maxtime_cross'] = 7.0
    
    fig_perpix,fig_crossing_perpix,fig_crossed_perpix,fig_ampspline,fig_cumcross,all_logdist,all_t = crossing_comparison('generator_data',['markov_dxdy_dirsample','semipar_nohist','semipar_hist'],'waldo_2024',normalized=True,leg=['Data','Nonparametric','Parametric (no memory)','Parametric (memory)'],**kwargs)

    
