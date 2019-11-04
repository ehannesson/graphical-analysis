import numpy as np

def iterate_dynamics(f, x0, *args, iters=25, **kwargs):
    """
    Numerically calculate the first iters points of the orbit of x0 under f.

    Parameters:
        f (func): callable function of the form f(x0, *args, **kwargs)
            **Must be compatible with numpy broadcasting
        x0 (float): orbital seed
        *args: any additional arguments to be passed to f
        iters (int): number of orbit points to calculate
            For iters=n, this finds the list [x0, f(x0), ..., f^n(x0)]
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
            print('WARN: OverflowError encountered at iteration {}.'.format(n))
            print('Terminating function iteration.')
            last_iter = n - 1
            overflow = True
            break

    if overflow:
        return orbit, last_iter
    else:
        return orbit
