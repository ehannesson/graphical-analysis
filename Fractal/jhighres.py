# jhighres.py
import numpy as np
from numba import njit
from julia_sets import plot_julia
import os
import re
import progressbar
import time
import multiprocessing as mp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# @njit(nopython=True)
def high_res_julia(f, window, outfile, splits=3, iters=25, res=(3000, 3000), N=5,
                    cmap='gist_rainbow', dpi=600, sub_path='Graphics/subfigs/',
                    save=False):
    """
    Creates a high resolution image of the julia set of some function f.
    Splits the whole window into many subwindows, computes high resolution images
    for each subwindow, and finally stiches those images together. Saves this
    final image as specified by the outfile argument. Returns None.

    Parameters:
        f (function): the function whose julia set we are creating
        window (tuple(xmin, xmax, ymin, ymax)): x and y bounds for where we are plotting
        outfile (str): filename or path/filename specifying save location

        splits (int): breaks down the computation into splits^2 subwindows, allowing for
        high resolution final output
        iters (int): number of iterations to compute to determine if a point becomes unbounded
        res (tuple(int, int)): resolution for each SUBWINDOW. This means that the final image
            will be composed of (splits^2)(res[0]*res[1]) pixels
        N (int): Bound to consider something 'unbounded'
        cmap (str): color map argument for matplotlib

    Returns:
        None
    """
    # change directories into the subpath
    os.chdir(sub_path)
    # create save directory if saving
    if save and not os.path.exists(save):
        os.mkdir(save)

    # just for tracking our current subwindow (for progress updates)
    c_window = 1

    # STEP 1 SPLIT WINDOW
    xmin, xmax, ymin, ymax = window
    # get the start/endpoints for the new subwindows
    xsplits = np.linspace(xmin, xmax, splits+1)
    ysplits = np.linspace(ymin, ymax, splits+1)
    # extract x- and y-subwindow intervals
    xwindows = [(xsplits[i], xsplits[i+1]) for i in range(splits)]
    ywindows = [(ysplits[i], ysplits[i+1]) for i in range(splits)]

    # STEP 2 CREATE SUBWINDOW IMAGES
    # iterate through our grid left to right, top to bottom creating subfigures
    for yinterval in range(splits):
        for xinterval in range(splits):
            subwindow = xwindows[xinterval] + ywindows[yinterval]
            # if we want to pickle the objects
            if save:
                subsave = save + '/{}{}'.format(yinterval, xinterval)
                if not os.path.exists(subsave): os.mkdir(subsave)
            else:
                subsave = False

            # format temporary save file as temp[row][column]
            _tempsave = 'temp{}{}'.format(yinterval, xinterval)
            print('Subwindow {}/{}'.format(c_window, int(splits**2)))
            c_window += 1
            # use multiprocessing to avoid memory leaks
            _args = (f, subwindow)
            _kwargs = {'iters': iters, 'res': res, 'N': N, 'cmap': cmap, 'plot': False,
                        'savefig': _tempsave, 'dpi': dpi, 'save': subsave}

            p = mp.Process(target=plot_julia, args=_args, kwargs=_kwargs)
            p.daemon=True
            p.start()
            p.join()

            # plot_julia(f, subwindow, iters=iters, res=res, N=N, cmap=cmap,
            #             plot=False, savefig=_tempsave, dpi=dpi, save=subsave)
            # plt.clf()

    # STEP 3 STITCH IMAGES BACK TOGETHER
    print('Combining images into rows...')
    # path to hard-disk temp storage
    temp_path = '-define registry:temporary-path=/run/media/erik/8CEC90F6EC90DC30/temp'
    for row in range(splits):
        # string to match images in this row
        row_match = 'temp{}*.png'.format(row)
        _args = 'convert {} +append {} temp{}.png'.format(row_match, temp_path, row)

        # stitch images into a row
        os.system(_args)
        time.sleep(5)

        # use multiprocessing to avoid memorey leaks
        # p = mp.Process(target=os.system, args=_args)
        # p.daemon=True
        # p.start()
        # p.join()

    print('Combining rows into final image...')
    # stitch first two rows together
    os.system('convert {} temp1.png temp0.png -append _temp.png'.format(temp_path))

    # now stitch the rest of the rows to the _temp.png image
    for row in range(2, splits):
        time.sleep(5)
        # use multiprocessing to avoid memory leaks
        _args = 'convert {} temp{}.png _temp.png -append _temp.png'.format(temp_path, row)
        os.system(_args)

        # p = mp.Process(target=os.system, args=_args)
        # p.daemon=True
        # p.start()
        # p.join()

        # os.system('convert {} temp{}.png _temp.png -append _temp.png'.format(temp_path, row))

    # rename the temp file as outfile, clean all temp files
    os.system('mv _temp.png ../final/{}'.format(outfile))
    os.system('rm temp*.png')
    os.chdir('../..')

    return
