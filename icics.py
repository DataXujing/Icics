# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from  sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold

import random


#compute euclid dist

def EuclidDis1(A,B):
    '''
    ==============================================================
    Description: A method to compute two matrix's euclid distance.

    Usage：EuclidDis1(A,B)
    ==============================================================
    Arguments:

        A: A matrix
        B: A matrix
    the method return a np.matrix object.
    '''
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA())**0.5
    return np.matrix(ED)

#model1

def recomm(dfold,df,bh,ywy,topn,ncluster,epsilon=0.001,init='k-means++',random_state=123,max_iter=1000,algorithm="auto"):
    '''
    =============================================================================================
    Description: The method which can assign cases to a salesman.

    Usage：recomm(dfold,df,bh,ywy,topn,ncluster,epsilon=0.001)
    =============================================================================================
    Arguments:

        dfold:     A DataFrame, the history case feature which this ywy has option.
        df:        A DataFrame, the new case feature which ready to assign.
        bh:        A list, the new case's id number which is one-to-one correspondence with df.
        ywy:       A string, the salesman's id number.
        topn:      An int number, how many cases should assign to the salesman.
        ncluster:  An int number, how many class should cluster.
        epsilon:   A little real number which avoid algorithmic errors
        ......
        -----------------------------------------------------------------------------------------
        more arguments find in the function of KMeans.
    Require:
        import numpy as np
        import pandas as pd
        from  sklearn.cluster import KMeans

    '''
    clf = KMeans(n_clusters=ncluster,init = init,random_state=random_state,max_iter=max_iter,algorithm=algorithm).fit(dfold)

    A = np.matrix(clf.cluster_centers_)
    B = np.matrix(df)
    dist = EuclidDis1(A, B)
    dist = np.array(dist)
    

    dist1 = dist.copy()
    dist_sort = pd.DataFrame({'bh':bh,'cluster':list(np.argmax(dist1,axis=0)),'distance':list(np.max(dist1,axis=0))})
    N = len(list(dist_sort.bh))
    Ni = pd.DataFrame(dist_sort.cluster.value_counts())
    Ni.reset_index(inplace=True)
    Ni.rename(columns={"index":0,"cluster":1},inplace=True)
    #Ni =pd.DataFrame(np.array(Ni0))
    Ni.rename(columns={0:'class',1:'counts'},inplace=True)

    dist_sort["group_sort"] = dist_sort['distance'].groupby(dist_sort['cluster']).rank(ascending=1,method='first')
    
    dist_sort = dist_sort.merge(Ni,left_on='cluster',right_on='class',how="left")
          
    dist_sort1 = dist_sort.groupby('cluster').apply(lambda x: x[x.group_sort <= topn*x.counts/(float(N)+epsilon)])    #可能会出bug
 
    recommend = pd.DataFrame({'bh':np.array(dist_sort1['bh']),'ywy':list(np.unique(ywy))*len(np.array(dist_sort1['bh']))})
    
    return recommend



#model2

def icic(dfold0,df0,topn0,bhywy0,ncluster,shuffle=0,epsilon=0.001,init='k-means++',random_state=123,max_iter=1000,algorithm="auto",path=0):
    '''
    ==========================================================================
    Description: The method which can assign cases to more than one salesmans.

    Usage：icic(dfold0,df0,shuffle=0,topn0,bhywy0,ncluster,...)
    ==========================================================================

    Arguments:

            dfold0:      A DataFrame, the history case feature which ywys has option.
                         dfold0 should incloud 'ywy0' and 'ajbh'and other features.

            df0:         A DataFrame, the new case feature which ready to assign.df0 should
                         incloud 'ajbh' and other features that the same as dfold0.

            shuffle:     0 or 1, if 1,then random disorder of the order of df0 and bhywy0.

            topn0:       A DataFrame,inclouding two cols like: 'ywy0' and 'topn_ywy',topn_ywy means 
                         how many cases should assign to this ywy0.

            bhywy0:      A DataFrame,inclouding two or one cols like: 'ywy0' and 'ajbh', and you can 
                         not provide the 'ajbh' some times.

            path:        0 or a string like '~/desktop/files.xlsx',it is a Excel file you want to save,
                         if path=0,means you do not need to save the assign result to you CP or disk.
  
            ......
            ------------------------------------------------------------------------------------------------
            more arguments find in the function of recomm. You can use help(recomm) or ?(recomm) to get
            support.
    Require:
        import numpy as np
        import pandas as pd
        from  sklearn.cluster import KMeans
        import random
    '''

    recommend_last  =pd.DataFrame({"bh":[],"ywy":[]})
    if shuffle==1:
        nrowdf0 = range(df0.shape[0])
        random.shuffle(nrowdf0)
        df0 = df0.iloc[nrowdf0]

        nrowbhywy0 = range(bhywy0.shape[0])
        random.shuffle(nrowbhywy0)
        bhywy0 = bhywy0.iloc[nrowbhywy0]
    else:
        pass


    dflst = df0.copy()
    ywylist = list(np.unique(list(bhywy0.ywy0)))

    for ywy in ywylist:
        if len(list(df0.ajbh)) >=1:
            dfold = dfold0[dfold0.ywy0==ywy]
            dfold1 = dfold.drop("ywy0",axis=1)
            dfold2 = dfold1.drop("ajbh",axis=1)
            bh = list(df0.ajbh)
            df2 =df0.drop('ajbh',axis=1)
            topn = int(np.unique(list(topn0[topn0.ywy0== ywy].topn_ywy)))
            recommend = recomm(dfold2,df2,bh,ywy,topn,ncluster,epsilon=0.001,init='k-means++',random_state=123,max_iter=1000,algorithm="auto")
            recommend_last = recommend_last.append(recommend)
    
            df0 = df0[np.invert(df0['ajbh'].isin(list(recommend.bh)))]
            

    recommend_last.reset_index(drop=True,inplace=True)
    set1 = set(recommend_last.bh)
    set2 = set(dflst.ajbh)
    if len(set2-set1) == 0:
        recommend_last1 = recommend_last
    else:
        dfno=pd.DataFrame({'bh':list(set2-set1),'ywy':list(np.random.choice ( ywylist,size=len(set2-set1)))})
        frames = [recommend_last,dfno]
        recommend_last1 = pd.concat(frames)

    if path == 0:
        pass
    else:
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        recommend_last1.to_excel(writer, sheet_name='Sheet1')
        writer.save()

    return recommend_last1


#Training

def acc_mean(dfold0,bhywy0,topn0,ncluster,n_folds=5,shuffle=True,random_state=33,shuffle_icic=0,epsilon=0.001,init='k-means++',max_iter=1000,algorithm="auto",path=0):
 
    '''
    =============================================================================================
    Description: The method which can training the icic model 

    Usage：acc_mean(labely,dfold0,bhywy0,ncluster,n_folds=5,shuffle=True,......)
    =============================================================================================
    Arguments:

        labely:   the training dataset label.
        n_folds:  the CV params.
        shuffle:  the CV params.
        ......
        -----------------------------------------------------------------------------------------
        more arguments find in the function of icic. You can use help(icic) or ?(icic) to get
        support.
    Require:
        from sklearn.cross_validation import StratifiedKFold

    '''

    mean_acc=[]
    labely = list(dfold0.ywy0) 
    skf = StratifiedKFold(labely,n_folds=n_folds,shuffle =shuffle,random_state=random_state)
    for train_index,test_index in skf:
        
        dfold1 = dfold0.iloc[train_index]
        bhywy_test_real =  dfold0.iloc[test_index][['ywy0','ajbh']]
        df1 = dfold0.iloc[test_index].drop('ywy0',axis=1)
        bhywy1 = bhywy0.iloc[train_index]

        topn1 = topn0[topn0['ywy0'].isin (list(bhywy1.ywy0))]
        
        validdata = icic(dfold1,df1,topn1,bhywy1,ncluster,shuffle=shuffle_icic,epsilon=epsilon,init=init,random_state=random_state,max_iter=max_iter,algorithm=algorithm,path=path)
    
        predata = pd.merge(validdata,bhywy_test_real,left_on='bh',right_on='ajbh',how='left')
        postive1 = 0
        for i in np.arange(0,np.shape(predata)[0]):
           
            if predata.ywy[i]==predata.ywy0[i]:
                postive1+=1       
        acc = float(postive1)/len(test_index)
    mean_acc.append(acc)
    mean_acc1 = np.mean(mean_acc)
    
    return mean_acc1







