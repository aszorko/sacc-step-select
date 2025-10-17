# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:17:35 2024

@author: aszor
"""

from abc import abstractmethod
import numpy as np
import sys
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import datasets

from scipy.ndimage import gaussian_filter
from scipy.interpolate import splrep,splev
from scipy.spatial import Delaunay


def load_data(data, split="train"):

    if data == "waldo_2024":
        return datasets.EyeTracking2024(split=split)
    elif data == "randpix":
        return datasets.EyeTrackingRandpix(split=split)
    else:
        raise ValueError(f"Unknown data option {data}")


class MarkovModel:
    def __init__(self,cond_dist,seed=222,predname="markov",ampspline=None):
        self.dist = cond_dist
        self.rng = np.random.default_rng(seed)
        self.predname=predname
        self.currx=None
        self.curry=None
        self.currt=None
        self.ampspline=ampspline

    def reset(self):
        self.currx=None
        self.curry=None
        self.currt=None

    def model_update(self,t,x,y):
        self.currx=x
        self.curry=y
        self.currt=t

    def get_candidates(self,currx,curry,n,t):
        #pass on to distribution object
        
        if self.ampspline is not None:
            scale = np.exp(splev(t,self.ampspline))
            return self.dist.get_candidates(currx,curry,n,ampscale=scale)
        else:
            return self.dist.get_candidates(currx,curry,n)
            
        
        
    def get_predictors(self,t,steps,logit=True,normalize=False):
        #pass on to distribution object

        if self.ampspline is not None:
            scale = np.exp(splev(t,self.ampspline))
            predarr = self.dist.get_predictors(self.currx,self.curry,t,steps,ampscale=scale)
        else:
            predarr = self.dist.get_predictors(self.currx,self.curry,t,steps)

        #pred = np.log(predarr)
        if logit:
           if normalize:
              P = predarr / sum(predarr)
           else:
              P = predarr
           pred = np.log(P/(1-P))
        else:
           pred = predarr
           
        df = pd.DataFrame()
        df[self.predname] = pred
        return df
    

class ParametricModel:
    #until 6/1/2025: maxtime_cross=17.0,maxtime_closest=16.0,maxtime_visited=24.0
    #6-17/1/2025: maxtime_cross=4.0,maxtime_closest=30.0,maxtime_visited=3.0
    #17/1 - 18/4/2025: maxtime_cross=12.0,maxtime_closest=30.0,maxtime_visited=5.0, max_r=0.2
    def __init__(self,extent,
                 cols=['dist','dist_1back','logdist','cardinality','horiz_align','vert_align','dir_change','dir_change2','amp_change','log_amp_change','wall_dist_init','wall_dist_change','centreang','centre_dist_init','centre_dist_change','have_visited','crossings','closest_point','whistdist','whistdist_change'],
                 seed=333,
                 maxtime_cross=11.0,
                 maxtime_closest=38.0,
                 maxtime_visited=6.0,
                 maxtime_dir=2.5,
                 max_r=0.2,
                 mmperpx=-1,
                 screendist=-1,
                 col_t=False,
                 linspline=None,
                 quantiles=None):
        self.allsteps = []
        self.allangs = []
        self.extent = extent #only used for walldist,dist_deg and crossing_r
        self.rng = np.random.default_rng(seed)
        self.currwd = None
        self.currcd = None
        self.currang = None
        self.curramp = None
        self.maxtime_cross = maxtime_cross
        self.maxtime_closest = maxtime_closest
        self.maxtime_visited = maxtime_visited
        self.maxtime_dir = maxtime_dir
        self.max_r = max_r #radius for have_visited function
        self.cols = cols
        self.col_t = col_t
        self.mmperpx = mmperpx #for converting to degrees
        self.screendist = screendist #for converting to degrees
        self.linspline = linspline
        self.quantiles = quantiles
        if extent is not None:
           self.cent_x = (extent[0]+extent[1])/2
           self.cent_y = (extent[2]+extent[3])/2
        
    def reset(self):
        self.allsteps = []
        self.allangs = []
        self.currwd = None
        self.currwhd = None
        self.currang = None
        self.curramp = None
        self.currmd = None
            
    def model_update(self,t,x,y):
        self.allsteps.append([t,x,y])
        if self.extent is not None:
           self.currwd = self.walldist(x,y)
           self.currcd = self.centredist(x,y)
        if len(self.allsteps)>1:
            dx = self.allsteps[-1][1]-self.allsteps[-2][1]
            dy = self.allsteps[-1][2]-self.allsteps[-2][2]
            self.currang = np.arctan2(dy,dx)
            self.allangs.append(self.currang)
            self.curramp = np.sqrt(dx**2+dy**2)
            self.currmd = self.meanhistdist(x,y,t)
            self.currwhd = self.whistdist(x,y,t)
            self.curr_centroid_dist = self.centroid_dist(x,y,t)
        
    def walldist(self,x,y):
        wd = min([abs(x-self.extent[0]),abs(x-self.extent[1]),abs(y-self.extent[2]),abs(y-self.extent[3])])
        return wd / min(abs(self.extent[1]-self.extent[0])/2,abs(self.extent[3]-self.extent[2])/2)
    
    def centredist(self,x,y):
        return np.sqrt((x-(self.extent[0]+self.extent[1])/2)**2 + (y-(self.extent[2]+self.extent[3])/2)**2)
    
    def centreang(self,x,y):
        x1 = self.allsteps[-1][1]
        y1 = self.allsteps[-1][2]
        y0 = (self.extent[2]+self.extent[3])/2
        x0 = (self.extent[0]+self.extent[1])/2
        return np.cos(np.arctan2(y0-y,x0-x)-np.arctan2(y0-y1,x0-x1))


    def closest_point(self,x0,y0,t):
        n = len(self.allsteps)
        dists = []
        if n==1: #use current point. only used for first point in generator mode
            x1 = self.allsteps[0][1]
            y1 = self.allsteps[0][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
            
        for i in range(1,n): #i=1 -> fixation before current
            if len(dists)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_closest:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))

        return (min(dists),np.argmin(dists))

    def meanhistdist(self,x0,y0,t):
        n = len(self.allsteps)
        dists = []
        #if n==1: #use current point. only used for first point in generator mode
        #    x1 = self.allsteps[0][1]
        #    y1 = self.allsteps[0][2]
        #    dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
            
        for i in range(n): #i=1 -> fixation before current
            if len(dists)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_closest:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))

        return np.mean(dists)

    def whistdist(self,x0,y0,t):
        n = len(self.allsteps)
        dists = []
        #if n==1: #use current point. only used for first point in generator mode
        #    x1 = self.allsteps[0][1]
        #    y1 = self.allsteps[0][2]
        #    dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
        
        for i in range(n): #i=1 -> fixation before current
            #enforce minimum two distances
            if len(dists)>1 and t-self.allsteps[n-i-1][0]>self.maxtime_closest:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))

        return np.mean(dists)/np.max(dists)

    def whistdist_old(self,x0,y0,t):
        n = len(self.allsteps)
        dists = []
        w = []
        #if n==1: #use current point. only used for first point in generator mode
        #    x1 = self.allsteps[0][1]
        #    y1 = self.allsteps[0][2]
        #    dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
            
        for i in range(n): #i=1 -> fixation before current
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
            w.append(np.exp(-(t-self.allsteps[n-i-1][0])/self.maxtime_closest))

        return sum(np.array(dists)*np.array(w))/sum(w)


    def centroid_dist(self,x0,y0,t):
        n = len(self.allsteps)
        xs = []
        ys = []
        #if n==1: #use current point. only used for first point in generator mode
        #    x1 = self.allsteps[0][1]
        #    y1 = self.allsteps[0][2]
        #    dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
            
        for i in range(n): #i=1 -> fixation before current
            if len(xs)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_closest:
                break
            xs.append(self.allsteps[n-i-1][1])
            ys.append(self.allsteps[n-i-1][2])
        
        dist = np.sqrt((np.mean(xs)-x0)**2+(np.mean(ys)-y0)**2)

        return dist
    
    def have_visited_alt(self,x0,y0,t):
        n = len(self.allsteps)
        
        nvisits = 0
        
        for i in range(1,n):
            if t-self.allsteps[n-i-1][0]>self.maxtime_visited:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            d = np.sqrt((x1-x0)**2+(y1-y0)**2)
            if d < self.max_r: #* np.exp(-(t-self.allsteps[n-i-1][0])/self.maxtime_visited):
                nvisits += 1 #np.exp(-(t-self.allsteps[n-i-1][0])/self.maxtime_visited)

        return nvisits
    
    def have_visited(self,x0,y0,t):
        n = len(self.allsteps)
            
        for i in range(1,n):
            if t-self.allsteps[n-i-1][0]>self.maxtime_visited:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            d = np.sqrt((x1-x0)**2+(y1-y0)**2)
            if d < self.max_r: #* np.exp(-(t-self.allsteps[n-i-1][0])/self.maxtime_visited):
                return 1.0 #np.exp(-(t-self.allsteps[n-i-1][0])/self.maxtime_visited)

        return 0.0

    def crossings(self,x0,y0,t):
        x1 = self.allsteps[-1][1]
        y1 = self.allsteps[-1][2]
        #context = get_context()
        #Point, Segment = context.point_cls, context.segment_cls
        r1 = []
        r2 = []
        crossings = 0
        if self.mmperpx>0:
           x_cent = self.cent_x
           y_cent = self.cent_y
        n = len(self.allsteps)
        for i in range(2,n):
            if t-self.allsteps[n-i][0]>self.maxtime_cross:
                break
            x2 = self.allsteps[n-i][1]
            x3 = self.allsteps[n-i-1][1]
            y2 = self.allsteps[n-i][2]
            y3 = self.allsteps[n-i-1][2]
            
            #check if the two segments intersect
            denom = (x3-x2)*(y1-y0)-(y3-y2)*(x1-x0) 
            if denom != 0: #unequal gradients -> can intersect
               ss = ((x3-x2)*(y2-y0)-(y3-y2)*(x2-x0))/denom
               tt = ((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))/denom
               if ss > 0 and ss < 1 and tt > 0 and tt < 1:
                   crossings += 1
                   if self.mmperpx<0:
                      r1.append(np.sqrt((x1-x0)**2+(y1-y0)**2))
                      r2.append(np.sqrt((x3-x2)**2+(y3-y2)**2))
                   else:
                      r1.append(dva(x0-x_cent,x1-x_cent,y0-y_cent,y1-y_cent,self.mmperpx,self.screendist))
                      r2.append(dva(x2-x_cent,x3-x_cent,y2-y_cent,y3-y_cent,self.mmperpx,self.screendist))
                   
            #unit_segments = [Segment(Point(x0, y0), Point(x1, y1)), Segment(Point(x2, y2), Point(x3, y3))]
            #try:
            #    if segments_intersect(unit_segments):
            #       crossings += 1
            #except: #e.g. degenerate points
            #    continue
                
        return crossings,r1,r2
    
    def dir_moment(self,x0,y0,t):
        n = len(self.allsteps)

        if n<2:
            return 0.0

        xs = []
        ys = []
        dx = x0-self.allsteps[-1][1]
        dy = y0-self.allsteps[-1][2]

            
        for i in range(1,n):
            if len(xs)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_dir:
                break
            xs.append(np.cos(self.allangs[n-i-1]))
            ys.append(np.sin(self.allangs[n-i-1]))
            
        totx = sum(xs)
        toty = sum(ys)
        
        return np.sqrt(totx**2 + toty**2)/len(xs)*np.cos(np.arctan2(dy,dx)-np.arctan2(toty,totx))
    
    def dir_moment2(self,x0,y0,t):
        n = len(self.allsteps)

        if n<2:
            return 0.0

        xs = []
        ys = []
        dx = x0-self.allsteps[-1][1]
        dy = y0-self.allsteps[-1][2]

            
        for i in range(1,n):
            if len(xs)>0 and t-self.allsteps[n-i-1][0]>3*self.maxtime_dir:
                break
            t_fact = np.exp(-self.maxtime_dir*(t-self.allsteps[n-i-1][0]))
            xs.append((self.allsteps[n-i][1]-self.allsteps[n-i-1][1])*t_fact)
            ys.append((self.allsteps[n-i][2]-self.allsteps[n-i-1][2])*t_fact)
            
        totx = np.mean(xs)
        toty = np.mean(ys)
        
        return np.sqrt(totx**2 + toty**2)*np.cos(np.arctan2(dy,dx)-np.arctan2(toty,totx))

    def delaunay(self,x0,y0,t):

        points = np.concatenate([np.array(self.allsteps)[:,1:],[[x0,y0]]])
        tri = Delaunay(points)
        simplices = points[tri.simplices]
        others = []
        
        point = np.array([x0,y0])
        vertex_edges = []
        adjacency_mask = np.isin(simplices, point).all(axis=2).any(axis=1)
        for simplex in simplices[adjacency_mask]:
            self_mask = np.isin(simplex, point).all(axis=1)
            for other in simplex[~self_mask]:
                if not np.isin(others,other).any():
                    dist = np.linalg.norm(point - other)
                    vertex_edges.append(dist)
                    others.append(other)
        
        return np.mean(vertex_edges),np.std(vertex_edges)
        

    def get_predictors(self,t,steps,logit=False):
        predarr = []
        
        for [x,y] in steps:
           preds = {}
           
           dx = x-self.allsteps[-1][1]
           dy = y-self.allsteps[-1][2]
           dist = np.sqrt(dx**2+dy**2)
           #preds["t"] = t
           if "dist" in self.cols:
              preds["dist"] = dist
           if "dist_1back" in self.cols: #1-back distance
              if len(self.allsteps)<2:
                 preds["dist_1back"] = dist
              else:
                 preds["dist_1back"] = np.sqrt((x-self.allsteps[-2][1])**2 + (y-self.allsteps[-2][2])**2)
           if "logdist" in self.cols:
              preds["logdist"] = np.log(dist)
           if "logdist_sq" in self.cols:
              preds["logdist_sq"] = np.log(dist)**2
           if "logdist_cb" in self.cols:
              preds["logdist_cb"] = np.log(dist)**3
           if "dist_deg" in self.cols:
              preds["dist_deg"] = dva(x-self.cent_x,self.allsteps[-1][1]-self.cent_x,y-self.cent_y,self.allsteps[-1][2]-self.cent_y,self.mmperpx,self.screendist)
           if "cardinality" in self.cols:
              preds["cardinality"] = np.cos(4*np.arctan2(dy,dx))
           if "horiz_align" in self.cols:
              preds["horiz_align"] = np.cos(2*np.arctan2(dy,dx))
           if "vert_align" in self.cols:
              preds["vert_align"] = np.cos(np.pi+2*np.arctan2(dy,dx))
           if "dir_change" in self.cols:
              if self.currang is None:
                  preds["dir_change"] = 0.0
              else:
                  preds["dir_change"] = np.cos(np.arctan2(dy,dx)-self.currang)
           if "dir_change2" in self.cols:
              if self.currang is None:
                  preds["dir_change2"] = 0.0
              else:
                  preds["dir_change2"] = np.cos(2*(np.arctan2(dy,dx)-self.currang))           
           if "amp_change" in self.cols:
              if self.curramp is None:
                  preds["amp_change"] = 0.0
              else:
                  preds["amp_change"] = abs(dist-self.curramp)
           if "log_amp_change" in self.cols:
              if self.curramp is None:
                  preds["log_amp_change"] = 0.0
              else:
                  preds["log_amp_change"] = abs(np.log(dist)-np.log(self.curramp))
           if "centreang" in self.cols:
              preds["centreang"] = self.centreang(x,y)
           if "centre_dist_init" in self.cols:
              preds["centre_dist_init"] = self.currcd
           if "centre_dist_change" in self.cols:
              preds["centre_dist_change"] = self.centredist(x,y) - self.currcd
           if "wall_dist_init" in self.cols:
              preds["wall_dist_init"] = self.currwd
           if "wall_dist" in self.cols or "wall_dist_change" in self.cols:
              wd = self.walldist(x,y)
           if "wall_dist" in self.cols:
              #preds["wall_dist"] = 1 / (wd + 1e-3)
              preds["wall_dist"] = wd
           if "wall_dist_change" in self.cols:
              #preds["wall_dist_change"] = 1 / (wd + 1e-3) - 1 / (self.currwd + 1e-3)
              preds["wall_dist_change"] = wd - self.currwd
           if "have_visited" in self.cols:
              name = "have_visited"
              if self.col_t:
                 name += "_"+str(self.maxtime_visited)    
              preds[name] = self.have_visited(x,y,t)
           if "crossings" in self.cols or "log_crossings" in self.cols or "crossing_r" in self.cols:
              cross,r1,r2 = self.crossings(x,y,t)
           if "crossings" in self.cols:
              name="crossings"
              if self.col_t:
                 name += "_"+str(self.maxtime_cross)    
              preds[name] = cross
           if "log_crossings" in self.cols:
              name="log_crossings"
              if self.col_t:
                 name += "_"+str(self.maxtime_cross)    
              preds[name] = np.log(1+cross)
           if "crossing_r" in self.cols:
              preds['crossing_r1'] = r1
              preds['crossing_r2'] = r2
           if "closest_point" in self.cols or "log_closest" in self.cols:
              (r,t_closest) = self.closest_point(x,y,t)
           if "closest_point" in self.cols:
              name="closest_point"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = r
              #preds[name+"_t"] = t_closest
           if "log_closest" in self.cols:
              name="log_closest"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = np.log(r)
           if "meanhistdist" in self.cols or "meanhistdist_change" in self.cols:
              md = self.meanhistdist(x,y,t)
           if "meanhistdist" in self.cols:
              name = "meanhistdist"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = md
           if "meanhistdist_change" in self.cols:
              name = "meanhistdist_change"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              if self.currmd is not None:
                 preds[name] = md - self.currmd
              else:
                 preds[name] = 0.0
           if "whistdist" in self.cols or "whistdist_change" in self.cols:
              whd = self.whistdist(x,y,t)
           if "whistdist" in self.cols:
              name = "whistdist"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = whd
           if "whistdist_change" in self.cols:
              name = "whistdist_change"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              if self.currwhd is not None:
                 preds[name] = whd - self.currwhd
              else:
                 preds[name] = 0.0
           if "centroid_dist" in self.cols or "centroid_dist_change" in self.cols:
              md = self.centroid_dist(x,y,t)
           if "centroid_dist" in self.cols:
              name = "centroid_dist"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = md
           if "centroid_dist_change" in self.cols:
              name = "centroid_dist_change"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              if self.curr_centroid_dist is not None:
                 preds[name] = md - self.curr_centroid_dist
              else:
                 preds[name] = 0.0
           if "dir_moment" in self.cols:
              name="dir_moment"
              if self.col_t:
                 name += "_"+'{:.1f}'.format(self.maxtime_dir).replace('.','_')    
              preds[name] = self.dir_moment(x,y,t)
           if "delaunay_mean" in self.cols or "delaunay_std" in self.cols:
              dl_mean,dl_std = self.delaunay(x,y,t)
              if "delaunay_mean" in self.cols:   
                 preds["delaunay_mean"] = dl_mean
              if "delaunay_std" in self.cols:   
                 preds["delaunay_std"] = dl_std           

           predarr.append(preds)
           
           if self.linspline is not None:
               for i,q in enumerate(self.quantiles):
                   if preds[self.linspline] > q:
                      preds[self.linspline + '_q' + str(i+1)] = preds[self.linspline] - q
                   else:
                      preds[self.linspline + '_q' + str(i+1)] = 0
           
        pred_df = pd.DataFrame(predarr)
        return pred_df
        

#Super class for all histogram-based data representations.
#Sub-classes must have a get_distribution() method which takes any extra init() arguments
#Optional methods include get_candidates(x,y,n) for sampling of n possible steps at time t
#And get_predictors(x,y,t,steps) for Markovian step selection
class Distribution:
    def __init__(self,dataset,extent,seed=111,nhist_x=50,nhist_y=50,nhist_dx=100,nhist_dy=100,nhist_dist=200,nhist_ang=40,**kwargs):
        self.rng = np.random.default_rng(seed)
        self.extent = extent
        self.cellsize_x = (extent[1]-extent[0])/nhist_x
        self.cellsize_y = (extent[3]-extent[2])/nhist_y
        self.cellsize_dx = (extent[1]-extent[0])/nhist_dx
        self.cellsize_dy = (extent[3]-extent[2])/nhist_dy
        self.x_edges = np.linspace(extent[0],extent[1],nhist_x+1)
        self.y_edges = np.linspace(extent[2],extent[3],nhist_y+1)
        self.dx_edges = np.linspace(-(extent[1]-extent[0]),(extent[1]-extent[0]),2*nhist_dx+1)
        self.dy_edges = np.linspace(-(extent[3]-extent[2]),(extent[3]-extent[2]),2*nhist_dy+1)
        meshx,meshy = np.meshgrid(self.x_edges[1:],self.y_edges[1:])
        self.mx = meshx.flatten()
        self.my = meshy.flatten()
        meshdx,meshdy = np.meshgrid(self.dx_edges[1:],self.dy_edges[1:])
        self.mdx = meshdx.flatten()
        self.mdy = meshdy.flatten()
        self.maxdist = np.sqrt((self.extent[1]-self.extent[0])**2+(self.extent[3]-self.extent[2])**2)
        self.dist_edges = np.linspace(0,self.maxdist,nhist_dist+1)
        self.ang_edges = np.linspace(-np.pi,np.pi,nhist_ang+1)
        meshr,meshang = np.meshgrid(self.dist_edges[1:],self.ang_edges[1:])
        self.mr = meshr.flatten()
        self.mang = meshang.flatten()
        self.distribs = self.get_distribution(dataset,**kwargs)
        self.cumdistribs = [np.cumsum(x.T.flatten()) for x in self.distribs]

    @abstractmethod    
    def get_distribution(dataset,**kwargs):
        #returns a list of distributions
        pass

class Conditional_XY(Distribution):
    #P(x(t+1),y(t+1)|x(t),y(t))
    def __init__(self,dataset,extent,seed=111,**kwargs):
        super().__init(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma=0.3):
        
        h = np.zeros([len(self.x_edges)-1,len(self.y_edges)-1,len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               cx = np.argmax(self.x_edges>data[t,1])
               cy = np.argmax(self.y_edges>data[t,2])
               if px>0 and py>0 and cx>0 and cy>0:
                  h[cx-1,cy-1,px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,[sigma/self.cellsize_x,sigma/self.cellsize_y,sigma/self.cellsize_x,sigma/self.cellsize_y])
        
        #normalize
        for i in range(len(self.x_edges)-1):
            for j in range(len(self.y_edges)-1):
                h_smooth[:,:,i,j] = h_smooth[:,:,i,j] / sum(h_smooth[:,:,i,j].flatten())
                
        return [h_smooth]
    
class Conditional_dXdY(Distribution):
    #P(dx(t),dy(t)|x(t),y(t))
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma_xy=4,sigma_dxdy=0.1,ampspline=None):        
        h = np.zeros([len(self.dx_edges)-1,len(self.dy_edges)-1,len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               if ampspline is not None:
                   #control for time evolution
                   ampscale = np.exp(splev(data[t,0],ampspline))
               else:
                   ampscale = 1.0
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               dx = np.argmax(self.dx_edges>(data[t,1]-data[t-1,1])/ampscale)
               dy = np.argmax(self.dy_edges>(data[t,2]-data[t-1,2])/ampscale)
               if px>0 and py>0 and dx>0 and dy>0:
                  h[dx-1,dy-1,px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,[sigma_dxdy/self.cellsize_dx,sigma_dxdy/self.cellsize_dy,sigma_xy/self.cellsize_x,sigma_xy/self.cellsize_y])
        
        #normalize
        for i in range(len(self.x_edges)-1):
            for j in range(len(self.y_edges)-1):
                h_smooth[:,:,i,j] = h_smooth[:,:,i,j] / sum(h_smooth[:,:,i,j].flatten())
                
        return [h_smooth]
    
    def get_predictors(self,currx,curry,t,steps,minp=1e-10,ampscale=1.0):
        predarr = []
        for [x,y] in steps:
           #preds = {}
           dx = (x-currx)/ampscale
           dy = (y-curry)/ampscale
           px = np.argmax(self.x_edges>currx)
           py = np.argmax(self.y_edges>curry)
           ix = np.argmax(self.dx_edges>dx)
           iy = np.argmax(self.dy_edges>dy)
           
           if px>0 and py>0 and ix>0 and iy>0:
              p = self.distribs[0][ix-1,iy-1,px-1,py-1] + minp
           else:
              p = minp
           
           predarr.append(p)
           
        return np.array(predarr)

    def get_candidates(self,currx,curry,n,ampscale=1.0):
        steps = []
        for i in range(n):
           
           px = np.argmax(self.x_edges>currx)
           py = np.argmax(self.y_edges>curry)
           cumdistrib = np.cumsum(self.distribs[0][:,:,px-1,py-1].squeeze().T.flatten())
           
           while True:
               r1 = self.rng.random()
               ind = np.argmax(cumdistrib>r1)
               dxbin = self.mdx[ind]
               dybin = self.mdy[ind]
               dxind = np.argmax(self.dx_edges==dxbin)
               dyind = np.argmax(self.dy_edges==dybin)
    
               r21 = self.rng.random()
               r22 = self.rng.random()
               
               dx = r21*self.dx_edges[dxind-1] + (1-r21)*self.dx_edges[dxind]
               dy = r22*self.dy_edges[dyind-1] + (1-r22)*self.dy_edges[dyind]
               x = currx + dx*ampscale
               y = curry + dy*ampscale
                              
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps

              
           
class Unconditional_XY(Distribution):
    #P(x(t),y(t))
    def __init__(self,dataset,extent,seed=333,**kwargs):
        super().__init__(dataset,extent,**kwargs)
                
    def get_distribution(self,dataset,sigma=0.5):        
        h = np.zeros([len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               if px>0 and py>0:
                  h[px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,[sigma/self.cellsize_x,sigma/self.cellsize_y])
        
        #normalize                
        return [h_smooth/sum(h_smooth.flatten())]

    def get_candidates(self,currx,curry,n):
        steps = []
        for i in range(n):
           
            r1 = self.rng.random()
            ind = np.argmax(self.cumdistribs[0]>r1)
            xbin = self.mx[ind]
            ybin = self.my[ind]
            xind = np.argmax(self.x_edges==xbin)
            yind = np.argmax(self.y_edges==ybin)
 
            r21 = self.rng.random()
            r22 = self.rng.random()
            x = r21*self.x_edges[xind-1] + (1-r21)*self.x_edges[xind]
            y = r22*self.y_edges[yind-1] + (1-r22)*self.y_edges[yind]
                                         
            steps.append([x,y])
        return steps

        
class Unconditional_dXdY(Distribution):
    #P(dx(t),dy(t))
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma=0.1):
        
        h = np.zeros([len(self.dx_edges)-1,len(self.dy_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               dx = np.argmax(self.dx_edges>(data[t,1]-data[t-1,1]))
               dy = np.argmax(self.dy_edges>(data[t,2]-data[t-1,2]))
               if dx>0 and dy>0:
                  h[dx-1,dy-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,[sigma/self.cellsize_dx,sigma/self.cellsize_dy])
        
        #normalize
        h_smooth = h_smooth / sum(h_smooth.flatten())
                
        return [h_smooth]
    
    def get_candidates(self,currx,curry,n,ampscale=1.0):
        steps = []
        for i in range(n):
           
           while True:
               r1 = self.rng.random()
               ind = np.argmax(self.cumdistribs[0]>r1)
               dxbin = self.mdx[ind]
               dybin = self.mdy[ind]
               dxind = np.argmax(self.dx_edges==dxbin)
               dyind = np.argmax(self.dy_edges==dybin)
               #if dxind==0:
               #    print(r1,ind,dxbin,dybin)
    
               r21 = self.rng.random()
               r22 = self.rng.random()
               x = currx + ampscale*(r21*self.dx_edges[dxind-1] + (1-r21)*self.dx_edges[dxind])
               y = curry + ampscale*(r22*self.dy_edges[dyind-1] + (1-r22)*self.dy_edges[dyind])
                              
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps

        
class Unconditional_ThetaR_Indep(Distribution):
    #P(theta)*P(r)
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset):
        
        h_dists = np.zeros(len(self.dist_edges)-1)
        h_angs = np.zeros(len(self.ang_edges)-1)
        
        for data in dataset:
            spatial_locations = data[:,1:3]
            diff_data = np.diff(spatial_locations,axis=0)
            dists_data = np.sqrt(diff_data[:,0]**2+diff_data[:,1]**2)
            angs_data = np.arctan2(diff_data[:,1],diff_data[:,0])
            
            h,_ = np.histogram(dists_data,bins=self.dist_edges)
            h_dists += h
            h,_ = np.histogram(angs_data,bins=self.ang_edges)
            h_angs += h
            #hack to enforce symmetry
            #h,_ = np.histogram(-angs_data,bins=self.ang_edges)
            #h_angs += h
            
        return [h_dists/np.sum(h_dists),h_angs/np.sum(h_angs)]
    
    def get_candidates(self,currx,curry,n,ampscale=1.0):
        steps = []
        for i in range(n):
           
           while True:
               r1 = self.rng.random()
               r11 = self.rng.random()
               ind1 = np.argmax(self.cumdistribs[0]>r1)
               dist = r11*self.dist_edges[ind1] + (1-r11)*self.dist_edges[ind1+1]
    
               r2 = self.rng.random()
               r22 = self.rng.random()
               ind2 = np.argmax(self.cumdistribs[1]>r2)
               ang = r22*self.ang_edges[ind2] + (1-r22)*self.ang_edges[ind2+1]
               
               x = currx + dist*np.cos(ang)*ampscale
               y = curry + dist*np.sin(ang)*ampscale
               
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps

    def get_predictors(self,currx,curry,t,steps,minp=1e-10):
        predarr = []
        for [x,y] in steps:
           #preds = {}
           dx = x-currx
           dy = y-curry
           r = np.sqrt(dx**2+dy**2)
           ang = np.arctan2(dy,dx)
           i_r = np.argmax(self.dist_edges>r)
           i_ang = np.argmax(self.ang_edges>ang)
           
           if i_r>0 and i_ang>0:
              p = self.distribs[0][i_r-1]*self.distribs[1][i_ang-1] + minp
           else:
              p = minp
           
           predarr.append(p)
           
        return np.array(predarr)

class Unconditional_ThetaR_Joint(Distribution):
    #P(theta)*P(r)
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma_r=0.02,sigma_ang=0.1):
        
        h_tot = np.zeros([len(self.dist_edges)-1,len(self.ang_edges)-1])
        
        for data in dataset:
            spatial_locations = data[:,1:3]
            diff_data = np.diff(spatial_locations,axis=0)
            dists_data = np.sqrt(diff_data[:,0]**2+diff_data[:,1]**2)
            angs_data = np.arctan2(diff_data[:,1],diff_data[:,0])
            
            h,_,_ = np.histogram2d(dists_data,angs_data,bins=(self.dist_edges,self.ang_edges))
            h_tot += h
        
        cellsize_r = self.dist_edges[1]-self.dist_edges[0]
        cellsize_ang = self.ang_edges[1]-self.ang_edges[0]
        h_tot = gaussian_filter(h_tot,[sigma_r/cellsize_r,sigma_ang/cellsize_ang])
            
        return [h_tot/np.sum(h_tot.flatten())]
    
    def get_candidates(self,currx,curry,n,ampscale=1.0):
        steps = []
        for i in range(n):
           
           while True:
               r1 = self.rng.random()
               ind = np.argmax(self.cumdistribs[0].flatten()>r1)
               rbin = self.mr[ind]
               angbin = self.mang[ind]
               rind = np.argmax(self.dist_edges==rbin)
               angind = np.argmax(self.ang_edges==angbin)
               #if dxind==0:
               #    print(r1,ind,dxbin,dybin)
    
               r21 = self.rng.random()
               r22 = self.rng.random()
               
               r = r21*self.dist_edges[rind-1] + (1-r21)*self.dist_edges[rind]
               ang = r22*self.ang_edges[angind-1] + (1-r22)*self.ang_edges[angind]
               x = currx + r*np.cos(ang)*ampscale
               y = curry + r*np.sin(ang)*ampscale
                              
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps


    def get_predictors(self,currx,curry,t,steps,minp=1e-10):
        predarr = []
        for [x,y] in steps:
           #preds = {}
           dx = x-currx
           dy = y-curry
           r = np.sqrt(dx**2+dy**2)
           ang = np.arctan2(dy,dx)
           i_r = np.argmax(self.dist_edges>r)
           i_ang = np.argmax(self.ang_edges>ang)
           
           if i_r>0 and i_ang>0:
              p = self.distribs[0][i_r-1,i_ang-1] + minp
           else:
              p = minp
           
           predarr.append(p)
           
        return np.array(predarr)
    

class LogNormalDist:    
    def __init__(self,dataset,extent,seed=None,man_params=None,**kwargs):
        self.rng = np.random.default_rng(seed)
        self.extent = extent
        if dataset is not None:
           self.params = self.get_distribution(dataset)
        else:
           self.params = man_params

    def get_distribution(self,dataset,**kwargs):
        all_logdist = np.array([])
        for data in dataset:
            dx = np.diff(data[:,1])
            dy = np.diff(data[:,2])
            logdist = np.log(np.sqrt(dx**2+dy**2))
            all_logdist = np.concatenate([all_logdist,logdist])
        mean = np.mean(all_logdist)
        std = np.std(all_logdist)
        
        return [mean,std]

    def get_candidates(self,currx,curry,n,ampscale=1.0):
        steps = []
        for i in range(n):
            while True:
                d = ampscale*np.exp(self.rng.normal()*self.params[1] + self.params[0])
                ang = 2*np.pi*self.rng.random()
                x = currx + d*np.cos(ang)
                y = curry + d*np.sin(ang)
                if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                    break
            steps.append([x,y])
        return steps
        
    def get_predictors(self,currx,curry,t,steps):
        p = []
        for [x,y] in steps:
            r = np.sqrt((x-currx)**2 + (y-curry)**2)
            if r==0:
                p.append(0.0)
            else:
                #lognormal pdf
                p.append(np.exp(-(np.log(r)-self.params[0])**2/2/self.params[1]**2)/r/self.params[1]/np.sqrt(2*np.pi))

        return np.array(p)
       
   
def step_select_run(data,distobj,models,n,pred_start,prevdata=None):
    
    assert pred_start>0


    df = pd.DataFrame()
    
    ii = 0
    skips = 0
    
    datalen = data.shape[0]-1
    for i in range(datalen):
        sys.stdout.write("\r%d%%" % round(100*i/datalen))
        sys.stdout.flush()
        
        [t1,x1,y1] = data[i+1,:]
        [t0,x0,y0] = data[i,:]
        
        if x1==x0 and y1==y0: #invalid zero-length saccade
            skips += 1
            continue

        for m in models:
            m.model_update(t0,x0,y0)
        if (i-skips)<pred_start:
            continue

        if n>0 and prevdata is None:
           steps = distobj.get_candidates(x0,y0,n)
        else:
           steps = []
           for j in range(n):
              steps.append([prevdata['x'][(n+1)*ii+j],prevdata['y'][(n+1)*ii+j]])
        steps.append([x1,y1])
        target = [0 for j in range(n)]
        target.append(1)

        newdf = pd.DataFrame(np.array(steps),columns=['x','y'])
        newdf['t']=t1
        newdf['dx']=newdf.x - x0
        newdf['dy']=newdf.y - y0
        for m in models:
            preds_df = m.get_predictors(t1,steps)            
            newdf = pd.concat([newdf,preds_df],axis=1)
        newdf['target'] = target
        df = pd.concat([df,newdf])
        ii += 1 #number of rows
    
    return df

    
def markov_test(distobj,n=100):
    
    out = [[0,0]]
    traj = out
    for i in range(n):
        out = distobj.get_candidates(out[0][0],out[0][1],1)
        traj.append(out[0])

    tarr = np.array(traj)
    plt.plot(tarr[:,0],tarr[:,1])       
    plt.show()
     
    return tarr
 

def import_preds(new_df,old_df):
    if len(new_df)==len(old_df) and sum(abs(new_df["x"]-old_df["x"]))<1e-6*len(new_df):
       try:
           for col in new_df.columns:
              old_df[col] = new_df[col]
           print("\nSuccessfully combined with existing file.")
           return old_df
       except:
           print("\nError combining columns. Saving new columns only")
           return new_df           
    else:
        print("\nData frame importing failed! Candidate steps do not agree.")
        print(len(new_df),len(old_df))
        return new_df    

def add_ID_cols(pred_df,dataset,testID):
    
    if dataset == "waldo":
       pred_df["Season"] = testID[:6]
       nums = testID[6:].split('-S')
       pred_df["Participant"] = int(nums[0])
       if len(nums)>1:
          pred_df["Sequence"] = int(nums[1])
    elif dataset == "waldo_2024":
       pred_df["Season"] = testID[:3]
       nums = testID[3:].split('-S')
       pred_df["Participant"] = int(nums[0])
       if len(nums)>1:
          pred_df["Sequence"] = int(nums[1])
    #other datasets can be added below
    #elif dataset == "xyz":
    else:
       raise ValueError("Unknown dataset in ID parser")
          
    
    return pred_df


def main(testdata,basedist,models,n_candidates,pred_start=1,save_file=None,existing_file=None,data_ids=None):
    #iterates through data and collects predictors and sequence IDs (if applicable)
    #data_ids should be a tuple containing dataset identifier and a list of strings with participant info for each
    
    if existing_file is not None:
       total_df = pd.read_csv(existing_file)
       curr_row = 0
       
    warnings.filterwarnings('ignore')

    if data_ids is not None:
        assert len(data_ids[1])==len(testdata)
        
    #run new/changed models if needed
    if len(models)>0:
        print(f"Running {len(models)} new or modified models.")
        new_df = pd.DataFrame()

        for i,data in enumerate(testdata):
           print('\nSequence',i+1,'of',len(testdata))
           if existing_file is not None:
                n_rows = (n_candidates+1)*(len(data)-pred_start-1)
                prevdata = total_df.iloc[curr_row:(curr_row+n_rows),:]
                prevdata = prevdata.reset_index(drop=True)
                curr_row += n_rows
           else:
                prevdata = None   
           pred_df = step_select_run(data,basedist,models,n_candidates,pred_start,prevdata=prevdata) 
           #pass to ID parser
           if data_ids is not None:
              pred_df = add_ID_cols(pred_df,data_ids[0],data_ids[1][i])
           new_df = pd.concat([new_df,pred_df],axis=0)
           
           for m in models:
              m.reset()
           
        
        new_df = new_df.reset_index(drop=True)
        if existing_file is not None:
            total_df = import_preds(new_df,total_df)
        else:
            total_df = new_df

        total_df['choice_id'] = np.arange(len(total_df))//(n_candidates+1)
            
        if save_file is not None and existing_file is not save_file:
            total_df.to_csv(save_file,index=False)
            print("\nSuccessfully saved.")
        else:
            print("\nNot saving: new filename needed!")
            
    return total_df

def time_constant_opt(markovdist,extent,**kwargs):
    models = []
    models.append(ParametricModel(extent,cols=['dist','logdist','logdist_sq','horiz_align','cardinality','dir_change','dir_change2','dist_1back','amp_change','centre_dist_init','centre_dist_change'],**kwargs))
    
    
    for t in range(1,46):
       kwargs["maxtime_cross"]=t
       models.append(ParametricModel(extent,cols=["crossings"],col_t=True,**kwargs))
    #for t in range(1,46):
    #   kwargs["maxtime_closest"]=t
    #   models.append(ParametricModel(extent,cols=["meanhistdist","meanhistdist_change"],col_t=True,**kwargs))
    for t in range(1,46):
       kwargs["maxtime_visited"]=t
       models.append(ParametricModel(extent,cols=["have_visited"],col_t=True,**kwargs))
    
    for t in range(1,46):
       kwargs["maxtime_closest"]=t
       models.append(ParametricModel(extent,cols=["whistdist","whistdist_change"],col_t=True,**kwargs))
    
    #for t in np.arange(0,11,1):
    #   kwargs["maxtime_dir"]=t
    #   models.append(ParametricModel(extent,cols=["dir_moment"],col_t=True,**kwargs))
    #models.append(MarkovModel(markovdist,predname="markov_dxdy"))

    return models

def sigma_opt(traindata,extent,sig1arr=np.arange(1.0,5.01,1.0),sig2arr=np.arange(0.05,0.251,0.05)):
    models = []
    for sig in sig1arr:
       for sig2 in sig2arr:
          markovdist = Conditional_dXdY(traindata,extent,sigma_xy=sig,sigma_dxdy=sig2)
          models.append(MarkovModel(markovdist,predname="markov_xy_{:.1f}_dxdy_{:.1f}".format(sig,sig2).replace('.','_')))
          print(sig,sig2)
          
    return models

def dva(x1,x2,y1,y2,mmperpx,screendist):
    #coordinates are relative to centre of screen
    theta1 = np.arctan(x1*mmperpx/screendist)
    theta2 = np.arctan(x2*mmperpx/screendist)
    theta3 = np.arctan(y1*mmperpx/screendist)
    theta4 = np.arctan(y2*mmperpx/screendist)
    
    return np.sqrt((theta1-theta2)**2 + (theta3-theta4)**2)*180/np.pi 

def get_spline(dataarr,plot=False):
    #cubic spline to control for temporal drift in median amplitude
    
    all_t = []
    all_logdist = []
    
    for data in dataarr:
        dx = np.diff(data[:,1])
        dy = np.diff(data[:,2])
        curr_t = data[1:,0]
        curr_logdist = np.log(np.sqrt(dx**2+dy**2))
        inds = np.isfinite(curr_logdist)
        all_t.append(curr_t[inds])
        all_logdist.append(curr_logdist[inds])

    t = np.concatenate(all_t)
    logdist = np.concatenate(all_logdist)
    
    sortinds = np.argsort(t)
    t = t[sortinds]
    logdist = logdist[sortinds]
    
    sfactor = 1.0
    
    spl = splrep(t,logdist-np.mean(logdist),s=len(t)*np.var(logdist)*sfactor)

    if plot:
        fig = plt.figure()
        plt.scatter(t,logdist,s=2,c='black')
        plt.plot(t,np.mean(logdist)+splev(t,spl),zorder=3)
        plt.plot([t[0],t[-1]],[np.mean(logdist),np.mean(logdist)],':',zorder=2)
        plt.xlabel('time (s)')
        plt.ylabel('log (saccade length)')
        plt.show()
        
        return spl, fig

    return spl

if __name__ == '__main__':
    
    
    """
    At minimum, the following need to be specified:
    - traindata and testdata: list of numpy arrays with dim (N,3)
      columns: (t,x,y)
      If the data comes straight from basic_measures.py, use:
          data_npz = np.load(npz_path)
          data = [data_npz[x] for x in data_npz.files]
      Train and test can be the same if overfitting is not an issue.
    - extent: tuple (xmin,xmax,ymin,ymax)
      These are the screen boundaries in data units.
    - n_candidates: set to zero if you only want predictors for real data points
    - data_ids: set to None if participant info not needed in dataset
      otherwise, this is a tuple of (dataset name string, list of ID strings)
      if using .npz file: data_ids = ('my_data',data_npz.files)
      A parser should then be added to the function add_ID_cols.
      minimal example:
          if dataset=="my_data":
              pred_df["SeqID"] = testID
    - models: a list of instantiated predictive models
    """
    
    ### SELECT DATASET BELOW    
    dataset = 'waldo_2024'
           
    data_ids = None 
    
    kwargs = {}
    

    if dataset=="waldo_2024":
        traindata = load_data('waldo_2024','train')
        testdata1 = load_data('waldo_2024','test')
        testdata2 = load_data('waldo_2024','val')
        testdata = testdata1 + testdata2
        testIDs = testdata1.IDs + testdata2.IDs
        trainIDs = traindata.IDs
        extent = (0,1920,0,1080)
        mmperpx = 0.2825
        screendist = 1000 #mm
        normalized=False
        kwargs["maxtime_cross"]=21.0
        kwargs["maxtime_closest"]=39.0
        kwargs["maxtime_visited"]=6.0
        data_ids = ('waldo_2024',testIDs)

    elif dataset=="randpix":
        #train distributions on NN training data, step selection on NN test+val data
        traindata = load_data('randpix','train')
        testdata1 = load_data('randpix','test')
        testdata2 = load_data('randpix','val')
        testdata = testdata1 + testdata2
        testIDs = testdata1.IDs + testdata2.IDs
        trainIDs = traindata.IDs
        extent = (0,1920,0,1080)
        mmperpx = 0.2825
        screendist = 1000 #mm
        normalized=False
        kwargs["maxtime_cross"]=10.0
        kwargs["maxtime_closest"]=5.0 #38.0
        kwargs["maxtime_visited"]=14.0
        data_ids = ('waldo_2024',testIDs) #use same parsing as waldo2024

    else:
        raise ValueError("invalid data type")
    

    stds = traindata.S_std.numpy().flatten()
    means = traindata.S_mean.numpy().flatten()

    traindata = [data.detach().numpy() for data in traindata]
    testdata = [data.detach().numpy() for data in testdata]

    if normalized: #x,y units in data standard deviations
        extent = ((extent[0]-means[0])/stds[0],(extent[1]-means[0])/stds[0],(extent[2]-means[1])/stds[1],(extent[3]-means[1])/stds[1])        
    else:    #x,y units in pixels
        for x in traindata:
           x[:,1:] = x[:,1:]*stds + means
        for x in testdata:
           x[:,1:] = x[:,1:]*stds + means
        #use foveal angle for max_r
        kwargs["max_r"] = screendist*np.tan(1.5*np.pi/180)/mmperpx

 

    #basedist = Unconditional_XY(traindata,extent) #for sigma_opt
    #basedist = Unconditional_dXdY(traindata,extent)
    basedist = LogNormalDist(traindata,extent)
    
    spl = get_spline(traindata)
    markovdist = Conditional_dXdY(traindata,extent,seed=444,sigma_dxdy=30,sigma_xy=1500,ampspline=spl)
    #markovdist = Unconditional_ThetaR_Joint(traindata,extent,seed=444)
    
 

    #Import previously run steps and predictors
    #Models in list will be added if they dont exist or overwrite older columns if they do
    #Set to None to start from scratch
    existing_file = None #'stepselect_data/stepselect_waldo2024_train.csv' #'stepselect_waldo_posttimeopt_imp.csv' #'stepselect_waldo_nn_diff_varhist5.csv' #'stepselect_predictors_nn60_new.csv'
    
    save_file = 'stepselect_data/stepselect_waldo2024_train.csv' #'stepselect_output.csv'
    
    models = []    

    ### LIST OF PREDICTIVE MODELS SPECIFIED BELOW
    
    #models = time_constant_opt(None,extent,**kwargs)
    #models = sigma_opt(traindata,extent,sig1arr=np.arange(250,1251,250),sig2arr=np.arange(10,51,10))
    
    models.append(ParametricModel(extent,**kwargs))
    models.append(MarkovModel(markovdist,predname="markov_dxdy",ampspline=spl))
 
    n_candidates = 9 #num false steps per real step

    
    #pred_df = main(testdata,basedist,models,n_candidates,save_file=save_file,existing_file=existing_file,data_ids=data_ids) 
    pred_df = main(traindata,basedist,models,n_candidates,save_file=save_file,existing_file=existing_file,data_ids=(data_ids[0],trainIDs))        
    #pred_df = main(traindata+testdata,basedist,models,n_candidates,save_file=save_file,existing_file=existing_file,data_ids=(data_ids[0],trainIDs+testIDs))        
        


