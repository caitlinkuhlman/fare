import numpy as np
import pandas as pd
import random as random
import pickle

def formatRank_german(df):
    tmp = pd.DataFrame()
    tmp['y']=df.sort_values('y_pred',ascending=False).index
    tmp['y_pred']=tmp.index
    tmp['g']=df.sort_values('y_pred',ascending=False).reset_index()['g']
    return tmp
    
def formatRank_compas(df):
    tmp = pd.DataFrame()
    tmp['y']=df.sort_values('y_pred').index
    tmp['y_pred']=tmp.index
    tmp['g']=df.sort_values('y_pred')['g']
    return tmp
    
def readFA_IRData(inpath, filename, funct):
    return funct(pd.read_pickle(inpath+filename))
    
def getAllFA_IRData(inpath, funct):
    d ={}

    d['cb'] = readFA_IRData(inpath, 'ColorblindRanking.pickle', funct)
    d['base'] = d['cb'].copy()
    d['base']['y_pred']=d['base']['y']
    d['feld'] = readFA_IRData(inpath, 'FeldmanRanking.pickle', funct)
    d['feld']['y'] = d['cb']['y']
    d['fair1'] = readFA_IRData(inpath, 'FairRanking01PercentProtected.pickle', funct)
    d['fair2'] = readFA_IRData(inpath, 'FairRanking02PercentProtected.pickle', funct)
    d['fair3'] = readFA_IRData(inpath, 'FairRanking03PercentProtected.pickle', funct)
    d['fair4'] = readFA_IRData(inpath, 'FairRanking04PercentProtected.pickle', funct)
    d['fair5'] = readFA_IRData(inpath, 'FairRanking05PercentProtected.pickle', funct)
    d['fair6'] = readFA_IRData(inpath, 'FairRanking06PercentProtected.pickle', funct)
    d['fair7'] = readFA_IRData(inpath, 'FairRanking07PercentProtected.pickle', funct)
    d['fair8'] = readFA_IRData(inpath, 'FairRanking08PercentProtected.pickle', funct)
    d['fair9'] = readFA_IRData(inpath, 'FairRanking09PercentProtected.pickle', funct)
    return d

def plainFA_IRData(inpath):
    d ={}

    d['cb'] = pd.read_pickle(inpath+'ColorblindRanking.pickle')
    d['base'] = d['cb'].copy()
    d['base']['y_pred']=d['base']['y']
    d['feld'] = pd.read_pickle(inpath+'FeldmanRanking.pickle')
    d['feld']['y'] = d['cb']['y']
    d['fair1'] = pd.read_pickle(inpath+'FairRanking01PercentProtected.pickle')
    d['fair2'] = pd.read_pickle(inpath+'FairRanking02PercentProtected.pickle')
    d['fair3'] = pd.read_pickle(inpath+'FairRanking03PercentProtected.pickle')
    d['fair4'] = pd.read_pickle(inpath+'FairRanking04PercentProtected.pickle')
    d['fair5'] = pd.read_pickle(inpath+'FairRanking05PercentProtected.pickle')
    d['fair6'] = pd.read_pickle(inpath+'FairRanking06PercentProtected.pickle')
    d['fair7'] = pd.read_pickle(inpath+'FairRanking07PercentProtected.pickle')
    d['fair8'] = pd.read_pickle(inpath+'FairRanking08PercentProtected.pickle')
    d['fair9'] = pd.read_pickle(inpath+'FairRanking09PercentProtected.pickle')
    return d