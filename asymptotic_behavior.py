# asymptotic_behavior.py
"""
Erik Hannesson
23 September 2019
Asymptotic Dynamical System Behavior
"""
import numpy as np

def asym_orbit_behavior(orbit, asym=150, max_cycle= 16, tol=1e-5, bins=20, int_tol=5):
    # first check if it is eventually fixed
    if np.allclose(orbit[asym:], orbit[-1], rtol=tol):
        return 1, orbit[-1]

    # check for cyc_len-cycles
    for cyc_len in range(2, max_cycle+1):
        cycle = True        # track if valid cycle

        for i in range(cyc_len):
            pot_cyc = orbit[asym+i::cyc_len]
            if not np.allclose(pot_cyc, pot_cyc[-1]):
                cycle = False
                break
        if cycle:
            return cyc_len, orbit[-cyc_len:]

    # # if no cycle was found
    return False, orbit
