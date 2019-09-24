# plot_bifurcations.py
"""
Erik Hannesson
23 September 2019
Math 534 - Dynamical Systems
"""
from cobweb import iterate_dynamics
from asymptotic_behavior import asym_orbit_behavior
import numpy as np
import matplotlib.pyplot as plt

def plot_bifurcation(f, iters=300, asym=200, max_cycle=16, tol=1e-5, n=500,
                        cmap='viridis_r', *args, **kwargs):

    behavior = []

    # for c_vals in [c1, c2, c3, c4, c5, c6]:
        # for c in c_vals:
    for c in np.linspace(-2.001, 0.25, n, endpoint=False):
        orbit = iterate_dynamics(f, 0, iters=iters, c=c)

        if type(orbit) is tuple:
            orbit=orbit[0]
        # (c, (cycle_length, [cycle_values])) OR (c, (Flase, [filled_intervals])) if no cycle found
        behavior.append((c, asym_orbit_behavior(orbit, asym=asym, max_cycle=max_cycle, tol=tol, **kwargs)))

    valid_cycle_lengths = {length[1][0] for length in behavior if length[1][0] is not False}
    color_map = {clen:i for i, clen in enumerate(valid_cycle_lengths)}
    c_map = plt.get_cmap(cmap)
    colors = [c_map(i) for i in np.linspace(0, 1, len(valid_cycle_lengths))]

    for orb in behavior:
        if orb[1][0] is not False:
            # cyclic orbits
            plt.plot([orb[0]]*orb[1][0], orb[1][1], linestyle=' ', marker='o',
                        color=colors[color_map[orb[1][0]]], ms=2)
        else:
            plt.plot([orb[0]]*len(orb[1][1]), orb[1][1], linestyle=' ', marker=',', color='black')

    plt.xlabel(r'$Parameter Value$')
    plt.ylabel(r'$x-Value$')
    plt.show()
