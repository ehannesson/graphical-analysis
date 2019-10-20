# asymptotic_behavior.py
"""
Erik Hannesson
23 September 2019
Asymptotic Dynamical System Behavior
"""
import numpy as np

def asymptotic_behavior(orbit, asym=150, max_cycle=16, tol=1e-5):
    """
    Numerically determines the asymptotic behavior of a given orbit.
    That is, it determines if orbit approaches a fixed point, two-cycle,
    three-cycle, etc., or if it is chaotic.

    Parameters:
        orbit (list): orbit of a dynamical system
        asym (int): orbit[asym:] is considered the 'asymptotic behavior'
        max_cycle (int): longest cycle length to check for before considering
            the orbit to be chaotic
        tol (float): tolerance for considering two numbers equivalent
    Returns:
        cycle_length (int or bool): if a cycle is detected, returns the length
            of the cycle; otherwise, returns False (i.e. orbit is chaotic)
        cycle_points (list): if a cycle is detected, returns the cycle's points;
            otherwise, returns orbit

    ----------------------------------------------------------------------------
    Important Notes
    ----------------------------------------------------------------------------
    This algorithm is not well suited for orbits that converge very slowly to
    fixed points or orbits (for example: 3x(1-x) with x0=0.9). In general, this
    algorithm will classify such orbits as chaotic.
    """
    # check for cycles of length cycle_length
    for cycle_length in range(1, max_cycle+1):
        is_cycle = True        # track if this is a valid cycle

        # check that each point in the potential cycle is fixed
        for i in range(cycle_length):
            pot_cyc = orbit[asym+i::cycle_length]

            if not np.allclose(pot_cyc, pot_cyc[-1]):
                # if all ith points of the potential cycle are not equivalent
                is_cycle = False
                break

        if is_cycle:
            return cycle_length, orbit[-cycle_length:]

    # # if no cycle was found
    return False, orbit
