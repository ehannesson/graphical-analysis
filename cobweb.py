# cobweb.py
"""
Erik Hannesson
Script for creating cobweb plots.
20 September 2019
"""
import matplotlib.pyplot as plt
import numpy as np

def iterate_dynamics(f, x0, iters=10, xlim=None, ylim=None, *args, **kwargs):
    """
    Numerically calculates the first iters points of the orbit of x0 under f and
    plots a corresponding cobweb graphical analysis.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
        x0 (float): orbital seed
        iters (int): number of orbit points to calculate/plot
            For iters=n, this finds/plots the list [x0, f(x0), ... , fn(x0)]
        xlim (2-tuple): should be a 2-tuple of floats of the form (x_min, x_max)
            This sets the x-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        ylim (2-tuple): should be a 2-tuple of floats of the form (y_min, y_max)
            This sets the y-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        *args: any additional arguments to be passed to f
        **kwargs: any additional arguments to be passed to f

    Returns:
        orbit (list): orbit of x0 under f: [x0, f(x0), ... , f^n(x0)]
            If there is an overflow error when calculating f^i(x0), the function
                will instead return a the tuple (orbit, i-1), where i-1 is the
                last point successfully calculated.
    """
    orbit = [x0]
    overflow = False

    # find the first iters orbit points
    for n in range(iters):
        try:
            # calculate next point on orbit
            x0 = f(x0, *args, **kwargs)
            orbit.append(x0)

        except OverflowError as e:
            # Print error message and terminate iteration
            print('WARN: OverflowError encountered at iteration {}.'.format(n))
            print('Terminating function iteration.')
            last_iter = n - 1
            break

    # prepare cobweb plot
    
