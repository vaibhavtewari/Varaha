import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

from multiprocessing import Pool

def rnd(x, p=1):
    
    return np.round(x, p)

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
    if np.isscalar(ninbin):
        x = np.vstack(np.random.uniform(xlo, xhi, size = (ninbin, bincount, ndim)))
    else:
        x = np.vstack([np.random.uniform(xlo[kk], xhi[kk], size = (npb, ndim)) for kk, npb in enumerate(ninbin)])
    return x

def get_likelihood_threshold(log_lkl, injw, shrink_frac):
    ''' Estimate the likelihood value that shrinks the volume
        log_lkl: numpy array
            the log likelihood values
        injw: numpy array
            the injection weights of the samples
        sorted: bool
            injection weights sorted according to log likelihood
        shrink_frac: float
            The fractional shrink in fraction
            
        Returns
        --------
        lkl_thr: float
            the new likelihood threshold
    '''

    idx = np.flip(np.argsort(log_lkl))
    cumw = np.cumsum(injw[idx])
    cumw /= cumw[-1]
    lkl_thr = log_lkl[idx][cumw >= shrink_frac][0]
    
    return lkl_thr

def volume_at_threshold(injw, log_lkl, lkl_thr, log_Vcurrent, log_errVcurrent, weighted=False):
    ''' Estimate volume at a likelihood threshold
        injw: numpy array
            the injection weights of the samples
        log_lkl: numpy array
            the log likelihood values
        lkl_thr: float
            the log likelihood threshold
        log_Vcurrent: float
            the logarithm of the volume enclosed by the current likelihood threshold
        log_errVcurrent: float
            the logarithm of the error on the volume enclosed by the current likelihood threshold
        weighted: bool
            based on un-weighted samples(i.e. affter performing rejection sampling)
            
        Returns
        --------
        log_Vnext: float
            The logarithm of estimated volume
        log_errVnext: float
            The logarithm of error on the estimated volume
    '''

    recw = np.array(injw)
    idx = np.where(log_lkl<=lkl_thr)
    recw[idx] = 0

    A = recw.sum()
    B = injw.sum()
    frac = A/B
    log_Vnext = log_Vcurrent + np.log(frac)

    if weighted:
        varA = np.sum((recw - recw.mean())**2)
        varB = np.sum((injw - injw.mean())**2)
        varAB = np.sum((recw - recw.mean())*(injw - injw.mean()))
        frac_err = frac * np.sqrt(varA / A ** 2 + varB / B ** 2 - 2*varAB/A/B)
        log_errVnext = log_Vnext + 0.5 * np.log((frac_err/frac) ** 2 + np.exp(log_errVcurrent - log_Vcurrent) ** 2)
    else:
        frac_err = np.sqrt(1/A - 1/B)
        log_errVnext = log_Vnext + np.log(frac_err)
    
    return log_Vnext, log_errVnext, frac

def split_dict(d, n):
    '''
    Split a dictionary in n parts
    '''
    keys = list(d.keys())
    idx = np.linspace(0, len(keys), n+1).astype(int)
    for ii, i in enumerate(idx[:-1]):
        yield {k: d[k] for k in keys[i: idx[ii+1]]}

def update_injection_density(allx, x, allx_log_injd, xrange, dx, binunique, ninbin, cycle, n=1):
    ''' Update injection density of samples in current cycles using contribution
        from bins in previous cycles. Update injection density of samples from
        previous cycles using contribution from bins in current cycle
        allx: (multid-array) All samples from previous cycles
        x: (multid-array) All samples from current cycle
        allx_log_injd: (array) injection density samples from previous cycles
        xrange: (numpy ndarray) range of parameters
        dx: (numpy array) the bin widths
        binunique: (dictionary) bins from previous cycles
        ninbin: (dictionary) number of samples in each bin for each cycle
        nbins: (dictionary) number of bins in each cycle in each dimension
        cycle: (float) the current cycle number
    '''
    
    if cycle > 1:
        log_injd_bins = np.log(ninbin[cycle]) - np.log(dx[cycle]).sum()
    
        if n==1:
            allx_log_injd = part1([allx, allx_log_injd, xrange, dx[cycle], binunique[cycle], log_injd_bins])
        else:
            pool = Pool(processes = n)
            split_allx = np.array_split(allx,n)
            split_injd = np.array_split(allx_log_injd,n)
            data = zip(split_allx, split_injd, n*[xrange], n*[dx[cycle]], n*[binunique[cycle]], n*[log_injd_bins])
            res = pool.map(part1, data)
            allx_log_injd = np.hstack(res)

    if n==1:
        log_injd_x = part2([x, binunique, ninbin, dx, xrange])
    else:
        pool = Pool(processes = n)
        n = min(cycle, pool._processes)
        data = zip(n*[x], split_dict(binunique, n), split_dict(ninbin, n), split_dict(dx, n), n*[xrange])
        res = pool.map(part2, data)
        log_injd_x = logsumexp(res, axis=0)

    return allx_log_injd, log_injd_x

def part1(data):
    allx, allx_log_injd, xrange, dx, bunq, log_injd_bins = data
    ndim = allx.shape[-1]
    nbins = bunq.shape[0]
    binid_allx = ((allx - xrange.T[0]) / dx).astype(int)
    label_allx = (binid_allx * 10 ** np.arange(ndim)).sum(axis=1)
    for ii in range(nbins):
        label_bu = (bunq[ii] * 10 ** np.arange(ndim)).sum()
        idx = np.where(label_allx == label_bu)[0]
        boolean = (np.abs(bunq[ii] - binid_allx[idx]).max(axis=1) == 0)
        idx = idx[boolean]
        allx_log_injd[idx] = np.logaddexp(allx_log_injd[idx], log_injd_bins[ii])

    return allx_log_injd

def part2(data):
    x, binunique, ninbin, dx, xrange = data
    keys = binunique.keys()
    ndim = x.shape[-1]
    log_injd_x = -np.inf * np.ones(x.shape[0])
    for cyc in keys:
        log_injd_cyc = np.log(ninbin[cyc]) - np.log(dx[cyc]).sum()
        binid_x = ((x - xrange.T[0]) / dx[cyc]).astype(int)
        label_x = (binid_x * 10 ** np.arange(ndim)).sum(axis=1)
        nbins = binunique[cyc].shape[0]
        for ii in range(nbins):
            bu = binunique[cyc][ii]
            label_bu = (bu * 10 ** np.arange(ndim)).sum()
            idx = np.where(label_x == label_bu)[0]
            boolean = (np.abs(bu - binid_x[idx]).max(axis=1) == 0)
            idx = idx[boolean]
            log_injd_x[idx] = np.logaddexp(log_injd_x[idx], log_injd_cyc[ii])
    return log_injd_x

def remove_early_cycles(allx_log_injd, allx, allloglkl, sampidx, xrange, dx, binunique, ninbin, del_upto_lkl, probmass_lkl, n):
    ''' Removes insignificant cycles, corresponding data and their 
        contribution to injection density
        allx_log_injd: (array) injection density samples from previous cycles
        allx: (multid-array) All samples from previous cycles
        allloglkl: (numpy array) Log likelihood values
        sampidx: (numpy array) The cycle number associated with a sample
        xrange: (numpy ndarray) range of parameters
        dx: (numpy array) the bin widths
        binunique: (dictionary) bins from previous cycles
        ninbin: (dictionary) number of samples in each bin for each cycle
        del_upto_lkl: (dictionary) likelihood threshold at each cycle
        probmass_lkl: the likelihood threshold below which cycles can be removed
    '''

    pool = Pool(processes = n)
    cycles = list(dx.keys())
    for cycle in cycles:
        if del_upto_lkl[cycle] < probmass_lkl:
            log_injd_bins = np.log(ninbin[cycle]) - np.log(dx[cycle]).sum()
            if pool is None:
                allx_log_injd = injd_subtract([allx, allx_log_injd, xrange, dx[cycle], binunique[cycle], log_injd_cycle])
            else:
                n = pool._processes
                data = zip(np.array_split(allx,n),np.array_split(allx_log_injd,n),n*[xrange],n*[dx[cycle]],n*[binunique[cycle]],n*[log_injd_bins])
                res = pool.map(injd_subtract, data)
                allx_log_injd = np.hstack(res)

            print ('~~~~~~', cycle, len(sampidx[sampidx==cycle]), del_upto_lkl[cycle], probmass_lkl)
            del dx[cycle]
            del binunique[cycle]
            del ninbin[cycle]
            idx = np.where(sampidx!=cycle)
            allx_log_injd = allx_log_injd[idx]
            allx = allx[idx]
            allloglkl = allloglkl[idx]
            sampidx = sampidx[idx]

    return dx, binunique, ninbin, ninbin, allx_log_injd, allx, allloglkl, sampidx

def injd_subtract(data):
    allx, allx_log_injd, xrange, dx, bunq, log_injd_bins = data
    ndim = allx.shape[-1]
    nbins = bunq.shape[0]
    binid_allx = ((allx - xrange.T[0]) / dx).astype(int)
    label_allx = (binid_allx * 10 ** np.arange(ndim)).sum(axis=1)
    
    for ii in range(nbins):
        label_bu = (bunq[ii] * 10 ** np.arange(ndim)).sum()
        idx = np.where(label_allx == label_bu)[0]
        boolean = (np.abs(bunq[ii] - binid_allx[idx]).max(axis=1) == 0)
        idx = idx[boolean]
        allx_log_injd[idx] = np.log(np.exp(allx_log_injd[idx]) - np.exp(log_injd_bins[ii]))

    return allx_log_injd

def get_bins(log_deltaV, starting_box, ndim, nones, cycle):

    rndim = ndim - nones
    mbin = np.exp(-log_deltaV/rndim)
    if cycle % 2 == 0:
        nbin = (mbin * 2 ** np.random.uniform(-1, 1, size = (400000, rndim)) + 0.5).astype(int)
    else:
        nbin = np.maximum(1, norm.rvs(loc = mbin, scale = mbin ** 0.5, size = (400000, rndim))).astype(int)
    nbin = np.maximum(1, nbin)
    nn = nbin[np.argmin(np.abs(1 - np.exp(np.log(nbin).sum(axis=1) + log_deltaV)))]
    nbins = np.append(nn, np.ones(nones))
    dx = np.diff(starting_box, axis = 1).flatten() / nbins

    return nbins, dx