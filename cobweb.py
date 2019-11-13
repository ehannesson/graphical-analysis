# cobweb.py
"""
Erik Hannesson
Script for creating cobweb plots.
20 September 2019
"""
import matplotlib.pyplot as plt
import numpy as np

def iterate_dynamics(f, x0, *args, iters=25, **kwargs):
    """
    Numerically calculate the first iters points of the orbit of x0 under f.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
            **Must be compatible with numpy broadcasting
        x0 (float): orbital seed
        iters (int): number of orbit points to calculate
            For iters=n, this finds the list [x0, f(x0), ..., f^n(x0)]
        *args: any additional arguments to be passed to f
        **kwargs: any additional arguments to be passed to f
    Returns:
        orbit (list): orbit of x0 under f: [x0, f(x0), ..., f^n(x0)]
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
            print('WARN: OverflowError encounteblack at iteration {}.'.format(n+1))
            print('Terminating function iteration.')
            last_iter = n
            overflow = True
            break

    # if array of orbits, reshape (leaves single orbit as expected)
    orbit = np.array(orbit).T

    if overflow and len(orbit.shape) == 1:
        return orbit, last_iter
    else:
        return orbit

def cobweb_plot(f, x0, *args, iters=25, xlim=None, ylim=None, cmap='magma', title=None, **kwargs):
    """
    Creates a "cobweb" graphical analysis plot of the orbit of x0 under f.
    Only the first iters points of the orbit are calculated and plotted.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
            **Must be compatible with numpy broadcasting
        x0 (float): orbital seed;
####    TODO: if the orbit is already known, pass in a list to avoid computing the orbit
        *args: any additional arguments to be passed to f
        iters (int): number of orbit points to calculate
            For iters=n, this finds the list [x0, f(x0), ..., f^n(x0)]
        xlim (2-tuple): should be a 2-tuple of floats of the form (x_min, x_max)
            This sets the x-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        ylim (2-tuple): should be a 2-tuple of floats of the form (y_min, y_max)
            This sets the y-axis bounds for plotting.
            If None, uses the matplotlib defaults.
        cmap (str): name of matplotlib color map; used to visually represent
            time (iterations) with cobwebbing lines.
            Suggested color maps include:
                viridis, cool, inferno, magma, and cividis
        title (str): graph title
        **kwargs: any additional arguments to be passed to f
    """
    cmap_num = 0
    cmaps = [cmap, 'viridis', 'magma', 'gist_rainbow', 'inferno_r', 'cool']
    cmaps = ['inferno_r'] + ['cool']*26 + ['inferno_r']

    # get orbit of x0 under f
    orbits = iterate_dynamics(f, x0, iters=iters, *args, **kwargs)
    # check for OverflowError
    if type(orbits) is tuple:
        orbits = orbits[0]

    min_orb = np.min(orbits)
    max_orb = np.max(orbits)

    if len(orbits) == 1:
        # if there is only one orbit, nest it so we can "loop" over all orbits
        orbits = [orbits]
    for orbit in orbits:
        # each element of lines is a tuple
            # each tuple contains two lists [x1, x2], [y1, y2]
            # containing the x- and y-coords of each cobweb line
        lines = [([orbit[0], orbit[0]], [0, orbit[1]])]
        lines.append(([orbit[0], orbit[1]], [orbit[1], orbit[1]]))

        # extract remaining cobweb lines
        for i in range(1, len(orbit) - 1):
            # line (x,x) --> (x, f(x))
            lines.append(([orbit[i], orbit[i]], [orbit[i], orbit[i+1]]))
            # line (x, f(x)) --> (f(x), f(x))
            lines.append(([orbit[i], orbit[i+1]], [orbit[i+1], orbit[i+1]]))

        # create color mapping for cobwebbing lines
        c_map = plt.get_cmap(cmaps[cmap_num%len(cmaps)]) # cycle cmaps for mult. orbits
        cmap_num += 1
        colors = [c_map(i) for i in np.linspace(0, 1, len(lines))]
        # cobweb lines get thinner as we iterate further
        lwidth = np.linspace(1.5, 1, len(lines))
        # draw cobweb lines
        for line, color, lw in zip(lines, colors, lwidth):
            plt.plot(line[0], line[1], color=color, lw=lw)

    # draw x and y axes
    plt.gca().axhline(color='black', lw=1)
    plt.gca().axvline(color='black', lw=1)

    # set the x and y limits
    if xlim:
        domain = np.linspace(xlim[0], xlim[1], 500)
        plt.xlim(xlim)
    else:
        xlim = plt.gca().get_xlim()
        domain = np.linspace(xlim[0], xlim[1], 500)

    if ylim:
        yrange = np.linspace(ylim[0], ylim[1], 500)
        plt.ylim(ylim)
    else:
        ylim = plt.gca().get_ylim()
        yrange = np.linspace(ylim[0], ylim[1], 500)

    # plot line y=x
    plt.plot(domain, domain, color='gray', zorder=-1)
    # plot function y=f(x)
    plt.plot(domain, f(domain, *args, **kwargs), color='gray', zorder=-1)

    if title:
        plt.title(title)

    # plt.show()

if __name__ == '__main__':
    end = 1.25
    f = lambda x: 3*(x-x**3)/2
    x = np.linspace(-end, end, 28)
    x[2], x[-3] = -1.1, 1.1

    cobweb_plot(f, x,
                iters=12,
                xlim=(-1.5,1.5),
                ylim=(-1.5,1.5),
                title=r'Orbits of $f(x)=\frac{3}{2}(x-x^3)$')
    cmap = ['inferno_r']*3 + ['jet']*10 + ['inferno_r']*10 + ['jet']*3
    plt.plot([-1.45]*2, [0, 10], color='black', lw=1.5)
    plt.plot([-1.35]*2, [0, 10], color='black', lw=1.5)
    plt.plot([1.45]*2, [0, -10], color='black', lw=1.5)
    plt.plot([1.35]*2, [0, -10], color='black', lw=1.5)
    plt.show()
# cmaps = ['cool'] + ['inferno_r']*2 + ['cool']*11 + ['inferno_r']*11 + ['cool']*2 + ['inferno_r']
# cmaps = ['inferno_r'] + ['cool']*26 + ['inferno_r']
