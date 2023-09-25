import numpy as np

def sample_from_bins(xrange, dx, bu, ninbin):
    ''' Sample from bins
        xrange: numpy ndarray
            the range of parameters in each dimension 
        dx: numpy array
            the width of the bins in each dimension
        bu: numpy ndarray
            the index of the bins to be sampled
        ninbin: integer
            the number of samples in each bin
            
        Returns
        --------
        x: numpy ndarray
            The multi-dimensional samples
    '''
    ndim = xrange.shape[0]
    bincount = bu.shape[0]
    xlo, xhi = xrange.T[0] + dx * bu, xrange.T[0] + dx * (bu+1)
    x = np.vstack(np.random.uniform(xlo, xhi, size = (ninbin, bincount, ndim)))
    return x

def rnd(x, p=1):
    
    return np.round(x, p)

def get_likelihood_threshold(log_lkl, log_injd, nsel, discard_prob=0.1):
    ''' Estimate the likelihood value that (1) encloses nsel largest likelihoods
        or (2) the value that encloses a fractional probability. Whichever is lower
        log_lkl: numpy array
            the log likelihood values that encompass the likelihood threshold
        log_injd: numpy array
            the injection density of the samples
        nsel: integer
            the number of effective samples enclosed by the new threshold
        discard_prob: float
            the fractional probability new threshold encloses
            
        Returns
        --------
        frac: float
            fraction of effective samples new threshold encloses
        lkl_thr: float
            the new likelihood threshold
        injneff: float
            the effective sample size inside the new likelihood threshold
            (close to nsel if method (1) is applicable
        pinside: float
            the fractional probability inside the new likelihood threshold
            (close to (1-discard_prob) if methid (2) is applicable)
    '''
    
    srtidx = np.flip(np.argsort(log_lkl))
    log_injw = -log_injd[srtidx]
    injw = np.exp(log_injw - log_injw.max())
    log_lkl = log_lkl[srtidx]
    cum_injw = np.cumsum(injw)
    injneff = (injw.sum()) ** 2 / (injw ** 2).sum()
    
    arr = injneff * cum_injw / cum_injw[-1]
    idx = np.where(arr >= min(arr[-1], nsel))[0][0]
    lkl_stop_thr = log_lkl[idx]
    
    lw = log_lkl + log_injw
    w = np.exp(lw - lw.max())
    srtidx = np.argsort(log_lkl)
    cum_w = np.cumsum(w[srtidx])
    ecdf = cum_w/cum_w[-1]
    idx = np.where(ecdf >= discard_prob)[0][0]
    prob_stop_thr = log_lkl[srtidx][idx]
    
    lkl_thr = min(lkl_stop_thr, prob_stop_thr)
    idx = np.where(log_lkl >= lkl_thr)
    frac = injw[idx].sum()/injw.sum()
    injneff *= frac
    pinside = np.sum(w[idx]) / cum_w[-1]
    
    return frac, lkl_thr, injneff, pinside

def get_likelihood_threshold_old(log_lkl, log_injd, nsel):
    ''' Estimate the likelihood value that enclose nsel largest likelihoods'''
    
    log_injw = -log_injd
    srtidx = np.flip(np.argsort(log_lkl))
    injw = np.exp(log_injw - log_injw.max())
    log_lkl = log_lkl[srtidx]
    injw = injw[srtidx]
    cum_injw = np.cumsum(injw)
    injneff = (injw.sum()) ** 2 / (injw ** 2).sum()
    
    arr = injneff * cum_injw / cum_injw[-1]
    idx = np.where(arr >= min(arr[-1], nsel))[0][0]
    lkl_stop_thr = log_lkl[idx]
    frac = injw[:idx].sum() / injw.sum()
    injneff = arr[idx]
    
    #ff = []
    #arr = np.arange(len(injw))
    #for ii in range(100):
    #    idx = np.random.choice(arr, size = len(injw)//16, replace=False)
    #    ff = np.append(ff, injw[idx][log_lkl[idx] > lkl_stop_thr].sum() / injw[idx].sum())
        
    #injneff = 16 * (np.mean(ff)/ff.std()) ** 2
    #print ('---', frac, np.mean(ff), np.std(ff)/4)
    
    return frac, lkl_stop_thr, injneff

def update_injection_density(allx, x, allx_log_injd, xrange, dx, binunique, ninbin, cycle):
    ''' Update injection density of samples in current cycles using contribution 
        from bins in previous cycles. Update injection density of samples from
        previous cycles using contribution from bins in current cycle
        allx: (multid-array) All samples from previous cycles
        x: (multid-array) All samples from current cycle
        allx_log_injd: (array) injection density samples from previous cycles
        binunique: (dictionary) bins from previous cycles
        ninbin: (dictionary) number of samples in each bin for each cycle
        nbins: (dictionary) number of bins in each cycle in each dimension
        cycle: (float) the current cycle number
    '''
    
    ndim = x.shape[1]
    log_injd_x = np.log(ninbin[cycle]) - np.log(dx[cycle]).sum()
    InsLiveVol = np.array([True] * x.shape[0])
    if cycle > 1:    
        binid_allx = ((allx - xrange.T[0]) / dx[cycle]).astype(int)
        label_allx = (binid_allx * 10 ** np.arange(ndim)).sum(axis=1)
        boolean = np.array([False] * binid_allx.shape[0])
        for bu in binunique[cycle]:
            label_bu = (bu * 10 ** np.arange(ndim)).sum()
            idx = np.where(label_allx == label_bu)
            boolean[idx] += (np.abs(bu - binid_allx[idx]).max(axis=1) == 0)
        allx_log_injd[boolean] = np.logaddexp(allx_log_injd[boolean], log_injd_x)
        InsLiveVol = np.append(boolean, InsLiveVol)
        
        log_injd_x *= np.ones(x.shape[0])
        for cyc in range(1, cycle):
            log_injd_cyc = np.log(ninbin[cyc]) - np.log(dx[cyc]).sum()
            binid_x = ((x - xrange.T[0]) / dx[cyc]).astype(int)
            label_x = (binid_x * 10 ** np.arange(ndim)).sum(axis=1)
            
            boolean = np.array([False] * binid_x.shape[0])
            for bu in binunique[cyc]:
                label_bu = (bu * 10 ** np.arange(ndim)).sum()
                idx = np.where(label_x == label_bu)
                boolean[idx] += (np.abs(bu - binid_x[idx]).max(axis=1) == 0)
            log_injd_x[boolean] = np.logaddexp(log_injd_x[boolean], log_injd_cyc)
            
    return allx_log_injd, np.ones(x.shape[0]) * log_injd_x, InsLiveVol

def sample_uniformly_on_cdf(x, log_lkl, log_injd):
    ''' Sample uniformly on cumulative density function
        (organised according to likelihood value)
        x = (multi-d array) parameter samples
        log_injd = (array) log of injection density for the samples
        nsamp = (int) number of samples to be drawn(not all will be unique)
    '''
    srtidx = np.flip(np.argsort(log_lkl))
    log_injw = -log_injd
    injw = np.exp(log_injw - log_injw.max())
    nsamp = int(injw.sum() ** 2 / (injw**2).sum() + 0.5)
    arr = np.linspace(1, nsamp, nsamp)
    cum_injw = np.cumsum(injw[srtidx])
    vidx = np.interp(arr, nsamp * cum_injw/cum_injw[-1], np.arange(len(injw))).astype(int)
    selx = x[srtidx][vidx]
    
    return selx