import numpy as np
from  scipy.optimize import curve_fit

def _func( x, f0, beta ) :
    return( f0 * x**beta ) 

def _calc_beta( wavs, flambda) :
    """ written by CJP to take a spectrum and return beta, the UV slope 
    following Calzetti et al. 1994 """

    """ 
    f_lambda  ~ lambda^beta

    with fitting windows: 
    """
    fit_windows = [(1268., 1284.),
                       (1309., 1316.),
                       (1342., 1371.),
                       (1407., 1515.),
                       (1562., 1583.),
                       (1677., 1740.),
                       (1760., 1833.),
                       (1866.,1890.),
                       (1930., 1950.),
                       (2400., 2580.)]

    for i in range(len(fit_windows)) : 
        if i==0 :
            ok = ((wavs > fit_windows[i][0]) & (wavs <= fit_windows[i][1]))
        else :
            ok = ok |  ((wavs > fit_windows[i][0]) & (wavs <= fit_windows[i][1]))


    x = wavs[ok]
    y = flambda[ok]
    mny = np.mean(y)
    if mny==0 : mny=1
    y /= mny

    p0 = [ 1.0, -1.0]
    #print(flambda)
    #print(x, y)
    popt, pcov = curve_fit(_func, x, y )

    popt[0] *= mny
    # I'm sure pcov is messed up here
    
    return( popt[1]) # return's beta

