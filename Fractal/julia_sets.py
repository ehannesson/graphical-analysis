# julia_sets.py
import numpy as np
# from mpmath import mp, mpf, mpc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import progressbar
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

def julia_set(f, window, *args, iters=50, res=(1000, 1000), N=2, save=False,
                verbose=True, **kwargs):
    """
    Numerically calculate the Julia set of f, tracking escape times of points
    not contained in the Julia set.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
            **Must be compatible with numpy broadcasting
        window: [r_min, r_max, i_min, i_max] real/imaginary window bounds
        *args: any additional arguments to be passed to f
        iters (int): number of iterations to calculate to decide if a point
            'escapes' or stays bounded
        res (int): resolution of the plot, should be (real_res, imag_res).
            That is, number of pixels in the real/imaginary axes.
        N (int): numeric bound for considering somehting becoming unbounded
        save (bool): if passed a string, pickles X, Y, escape, and iters in a subfolder
            with the name of the argument
        **kwargs: any additional arguments to be passed to f
    """
    # mp.dps = 20
    # create meshgrid
    # x = mp.linspace(window[0], window[1], res[0], endpoint=False)
    # y = list(mp.matrix(mp.linspace(window[2], window[3], res[1], endpoint=False))*1j)
    x = np.linspace(window[0], window[1], res[0], endpoint=False)
    y = np.linspace(window[2], window[3], res[1], endpoint=False)
    X, Y = np.meshgrid(x, y)
    # X, Y = mp.matrix(X), mp.matrix(Y*1j)
    Z = X + Y*1j

    # to track when things escape
    escape = np.zeros((res[1], res[0]), dtype=int)

    # iterate, updating when things have escaped-define registry:temporary-path={}
    bar = progressbar.progressbar(range(iters))
    for iter in bar:
        # get next iteration
        Z = f(Z)
        # update escape values
        mask = np.abs(Z) > N
        escape[mask] = iter+1
        Z[mask] = None

    if save:
        objs = (X, Y, escape, np.max(escape), hash(Z.tobytes()))
        names = ('X', 'Y', 'escape', 'max_iter', 'hash')
        for obj, name in zip(objs, names):
            with open(save + '/' + name, 'wb') as f:
                pickle.dump(obj, f)

    if verbose: print(np.max(escape))
    # TODO: consider returning the max escape, so that when creating high res image,
        # we can compute all the points first, tracking the max, then create the plots
        # with the data and know how to properly normalize the inputs
    return X, Y, escape


# TODO: if above update is implemented, consider renaming this to plot_julia_from_func
    # or something like that and create a new function called plot_julia_from_data which
    # would then be used by the high_res_julia function
def plot_julia(f, window, *args, iters=25, vmax=None, res=(1000, 1000), N=2, cmap='hsv',
                plot=True, savefig=False, dpi=300, save=False, shading='gouraud'):
    if not plot:
        matplotlib.use('Agg')

    # numerically find filled julia set on interval window, with excape speeds
    X, Y, orbits = julia_set(f, window, *args, iters=iters, res=res, N=N, save=save)

    if vmax is None:
        # if unspecified, set vmax to orbit maximum
        vmax = np.max(orbits)

    # mask zero values
    orbits = np.ma.masked_where(orbits == 0, orbits)
    cmap = plt.get_cmap(cmap)

    cmap.set_bad(color='black')

    print('Creating pcolormesh')
    stime = time.time()
    # plt.pcolormesh(X, Y, orbits, cmap=cmap, vmin=0, vmax=vmax, shading=shading, norm=norm)
    plt.pcolormesh(orbits,
                    # vmin=0,
                    # vmax=vmax,
                    shading=shading,
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax),
                    cmap=cmap)


    plt.axis('off')
    plt.gca().axis('image')
    print(time.time() - stime)
    if savefig:
        print('Saving figure')
        stime = time.time()
        plt.savefig('jet' + '-' + savefig, bbox_inches='tight', pad_inches=0, dpi=dpi)

    print(time.time() - stime)
    if plot:
        plt.show()


