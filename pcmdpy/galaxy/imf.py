# imf.py
# Ben Cook (bcook@cfa.harvard.edu)
__all__ = ['salpeter_IMF', 'kroupa_IMF', 'salpeter_meanmass',
           'kroupa_meanmass', 'salpeter_mass', 'salpeter_num',
           'kroupa_mass', 'kroupa_num']

import numpy as np


def salpeter_mass(alpha=-2.35, min_mass=0.1, max_mass=300.):
    return (np.power(max_mass, alpha+2) -
            np.power(min_mass, alpha+2)) / (alpha+2)


def salpeter_num(alpha=-2.35, min_mass=0.1, max_mass=300.):
    return (np.power(max_mass, alpha+1) -
            np.power(min_mass, alpha+1)) / (alpha+1)


def kroupa_mass(alpha_lower=-1.3, alpha_upper=-2.3, min_mass=0.1,
                max_mass=300., break_mass=0.5):
    mass_lower = (np.power(break_mass, alpha_lower+2) -
                  np.power(min_mass, alpha_lower+2)) / (alpha_lower+2)
    mass_upper = (np.power(max_mass, alpha_upper+2) -
                  np.power(break_mass, alpha_upper+2)) / (alpha_upper+2)
    return mass_lower + mass_upper


def kroupa_num(alpha_lower=-1.3, alpha_upper=-2.3, min_mass=0.1,
               max_mass=300., break_mass=0.5):
    num_lower = (np.power(break_mass, alpha_lower+1) -
                 np.power(min_mass, alpha_lower+1)) / (alpha_lower + 1)
    num_upper = (np.power(max_mass, alpha_upper+1) -
                 np.power(break_mass, alpha_upper+1)) / (alpha_upper+1)
    return num_lower + num_upper


def salpeter_IMF(mass, alpha=-2.35, lower=0.1, upper=300.,
                 norm_by_mass=False, **kwargs):
    mids = 0.5 * (mass[1:] + mass[:-1])  # midpoints between masses
    m_low = np.append([mass[0]], mids)  # (lowest bin stays same)
    m_high = np.append(mids, [mass[-1]])  # (highest bin stays same)
    imf = (np.power(m_high, alpha+1) - np.power(m_low, alpha+1)) / (alpha+1)
    imf[mass < lower] = 0.
    min_mass = max(lower, mass[0])
    max_mass = upper
    # Normalize so imf*mass sums to 1 (form 1 solar mass)
    if norm_by_mass:
        imf /= salpeter_mass(alpha=alpha, min_mass=min_mass, max_mass=max_mass)
    # Normalize so imf sums to 1 (form 1 star)
    else:
        imf /= salpeter_num(alpha=alpha, min_mass=min_mass, max_mass=max_mass)
    return imf


def kroupa_IMF(mass, alpha_lower=-1.3, alpha_upper=-2.3, lower=0.08,
               upper=300., break_mass=0.5, norm_by_mass=False, **kwargs):
    mids = 0.5 * (mass[1:] + mass[:-1])  # midpoints between masses
    m_low = np.append([mass[0]], mids)  # (lowest bin stays same)
    m_high = np.append(mids, [mass[-1]])  # (highest bin stays same)
    imf_lower = (np.power(m_high, alpha_lower+1) -
                 np.power(m_low, alpha_lower+1)) / (alpha_lower + 1)
    imf_upper = (np.power(m_high, alpha_upper+1) -
                 np.power(m_low, alpha_upper+1)) / (alpha_upper + 1)
    imf = imf_lower
    imf[mass >= break_mass] = imf_upper[mass >= break_mass]
    imf[mass < lower] = 0.
    min_mass = max(lower, mass[0])
    max_mass = upper

    # Normalize so imf*mass sums to 1 (form 1 solar mass)
    if norm_by_mass:
        imf /= kroupa_mass(alpha_lower=alpha_lower, alpha_upper=alpha_upper,
                           min_mass=min_mass, max_mass=max_mass,
                           break_mass=break_mass)
    # Normalize so imf sums to 1 (form 1 star)
    else:
        imf /= kroupa_num(alpha_lower=alpha_lower, alpha_upper=alpha_upper,
                          min_mass=min_mass, max_mass=max_mass,
                          break_mass=break_mass)
    return imf


def salpeter_meanmass(**kwargs):
    return salpeter_mass(**kwargs) / salpeter_num(**kwargs)


def kroupa_meanmass(**kwargs):
    return kroupa_mass(**kwargs) / kroupa_num(**kwargs)
