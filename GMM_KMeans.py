#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 00:21:27 2018

@author: nishimehta
"""

import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import csv
from matplotlib.patches import Ellipse


#%%
colors=['red','green','blue']

#%%

#%%
def Kmeansconverge(img,k):
    # flatten image to a list of points
    pts = img.reshape(img.shape[0]*img.shape[1],img.shape[2])  
    #initialize random cluster centers
    Mu = pts[np.random.choice(pts.shape[0], k, replace=False), :]
    while(True):
        # compute distance
        dist = np.linalg.norm(Mu - pts[:,None], axis=-1)
        # classify points closest to the centroid
        v = np.argmin(dist,axis=1)
        # compute next Mu values
        Mu_next = np.zeros((k,len(pts[0])))
        for i in range(k):
            Mu_next[i] = np.mean(pts[v.ravel()==i],axis=0)
        # convergence step if updated mu is equal to previous return
        if(np.array_equal(Mu,Mu_next)):
            return v,Mu
        else:
            Mu = Mu_next

#%%
def GMM(X,_Mu,_sig,_p,Iter):
    p = np.copy(_p)
    Mu = np.copy(_Mu)
    sig = np.copy(_sig)
    prob = np.zeros((len(Mu),len(X)))
    # calculate initial probabilities 
    for i in range(len(Mu)):
        prob[i] =  p[i]*multivariate_normal.pdf(X, mean=Mu[i], cov=sig[i])
    prob = prob/np.sum(prob,axis=0)
    # classify points with maximum probability to that cluster
    v = np.argmax(prob,axis=0)
    
    for i in range(Iter):
        # recompute Mu and valriance
        Mu = np.zeros(Mu.shape)
        sig = np.zeros(sig.shape)
        for i in range(len(Mu)):
            p[i] = np.mean(prob[i])
            Mu[i] = np.sum(prob[i][:,None] * X,axis=0)/np.sum(prob[i])
            X_MU = (X-Mu[i])
            for j in range(len(X)):
                X_MU_T = np.transpose(X_MU[j])
                X_MU_r = X_MU_T.reshape(len(X_MU[j]),1)
                X_MU_sq = np.multiply(X_MU_r,X_MU[j])
                sig[i]+=np.dot(prob[i][j],X_MU_sq)
            sig[i] = np.dot(1/np.sum(prob[i]),sig[i]) 
        # recompute probabilities
        for i in range(len(Mu)):
            prob[i] =  p[i]*multivariate_normal.pdf(X, mean=Mu[i], cov=sig[i])
        prob = prob/np.sum(prob,axis=0)
        v = np.argmax(prob,axis=0)
    return Mu,sig,v
    
#%%
# a function for graph settings
def getGraphParams():
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return fig,ax
#%%
# function for plotting ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip








#%%

img = cv2.imread('baboon.jpg') 
k = [3,5,10,20]
for i in k:
    vec,mu_final = Kmeansconverge(img,i)
    imgpoints = mu_final[vec]
    im = imgpoints.reshape(img.shape[0],img.shape[1],img.shape[2])
    cv2.imwrite('task3_baboon_'+str(i)+'.jpg',im)
    print('Kmeans clusters=',i)


#%%

sig1 = sig2 =sig3 = [[0.5,0],[0,0.5]]
sig = np.array([sig1,sig2,sig3])
p1=p2=p3=1/3
p=np.array([p1,p2,p3])
prob = np.zeros((len(Mu),len(X)))

Mu_Iter1_GMM,sig,v = GMM(X,Mu,sig,p,1)
print(Mu_Iter1_GMM)
print(v)

#%%
print('Task 3.5b')
old_f=[]
with open('old_faithful.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        datarow=[]
        for column in row:
            datarow.append(float(column))
        old_f.append(datarow)
old_f = np.array(old_f)

Mu_oldF = [[4.0, 81], [2.0, 57], [4.0, 71]]
p1_oldF = 1/3
p_oldF = np.array([p1_oldF,p1_oldF,p1_oldF]) 
sig1_oldF = [[1.30,13.98 ],[13.98,184.82]]
sig_oldF = np.array([sig1_oldF,sig1_oldF,sig1_oldF])
iterations = 5

for k in range(1,iterations+1):
    Mu_Iter_GMM_oldF,sig_iter,v_oldF = GMM(old_f,Mu_oldF,sig_oldF,p_oldF,k)
    fig,ax = getGraphParams()
    for i in range(len(Mu_Iter_GMM_oldF)):
        x,y = old_f[v_oldF.ravel()==i][:, :-1].flatten(), old_f[v_oldF.ravel()==i][:, -1].flatten()
        kwargs = dict(color=colors[i])
        plot_cov_ellipse(sig_iter[i],Mu_Iter_GMM_oldF[i],**kwargs)
        plt.scatter(x,y, marker= '.' ,c=colors[i])
        plt.scatter(Mu_Iter_GMM_oldF[i][0],Mu_Iter_GMM_oldF[i][1],marker='o',c=colors[i],s=100,edgecolors='black')
    
    fig.savefig('task3_gmm_iter'+str(k)+'.jpg')


    
