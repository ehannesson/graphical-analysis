# cobweb.py
"""
Erik Hannesson
Script for creating cobweb plots.
20 September 2019
"""
import matplotlib.pyplot as plt
import numpy as np

def iterate_dynamics(f, x0, iters=10, *args, **kwargs):
    """
    Numerically calculate the first iters points of the orbit of x0 under f.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
        x0 (float): orbital seed
        iters (int): number of orbit points to calculate
            For iters=n, this finds the list [x0, f(x0), ... , fn(x0)]
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
            overflow = True
            break

    if overflow:
        return orbit, last_iter
    else:
        return orbit

def cobweb_plot(f, x0, iters=10, xlim=None, ylim=None, cmap='viridis', *args, **kwargs):
    """
    Creates a "cobweb" graphical analysis plot of the orbit of x0 under f.
    Only the first iters points of the orbit are calculated and plotted.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
        x0 (float): orbital seed;
####    TODO: if the orbit is already known, pass in a list to avoid computing the orbit
        iters (int): number of orbit points to calculate
            For iters=n, this finds the list [x0, f(x0), ... , fn(x0)]
        xlim (2-tuple): should be a 2-tuple of floats of the form (x_min, x_max)
            This sets the x-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        ylim (2-tuple): should be a 2-tuple of floats of the form (y_min, y_max)
            This sets the y-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        cmap (str): name of matplotlib color map; used to visually represent
            time (iterations) with cobwebbing lines.
            Suggested color maps include:
                viridis, plasma, inferno, magma, and cividis
        *args: any additional arguments to be passed to f
        **kwargs: any additional arguments to be passed to f
    """
    # get orbit of x0 under f
    orbit = iterate_dynamics(f, x0, iters=iters, *args, **kwargs)
    # check for OverflowError
    if type(orbit) is list:
        orbit = orbit[0]

    # (x0, 0) --> (x0, f(x0)) --> (f(x0), f(x0)) --> (f(x0), f(f(x0)))
    # (0, _)  --> (0, 1) --> (1, 1) --> (1, 2)
    #
    # [(orbit[0], 0), (orbit[0], orbit[1])]
    # [(orbit[0], orbit[1]), (orbit[1], orbit[1])]
    # [(orbit[1], orbit[1]), (orbit[1], orbit[2])]
    # [(orbit[1], orbit[2]), (orbit[2], orbit[2])]




    # create color mapping for cobwebbing lines
    # c_map = plt.get_cmap(cmap)
    # colors = [c_map(i) for i in np.linspace(0, 1, num_lines)]
    #
    # for line, color in zip(lines, colors):
    #     plt.plot(line, color=color)
