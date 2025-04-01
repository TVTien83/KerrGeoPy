"""Module containing functions for finding resonance orbits
"""
from .frequencies import *
from .frequencies import _ellippi
from .constants import *
from .constants import _standardize_params
from scipy.optimize import fsolve, newton
from scipy.special import elliprf
from numpy import sqrt, pi, sign

def valid_frequencyRatio(ratio):
    if (0 < ratio) & (ratio < 1):
        return True
    else:
        return False

def _rtheta_frequencyRatio(a, p, e, x):
    """Ratio of r-frequency and theta-frequency  

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    
    
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")

    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    z_minus, z_plus = stable_polar_roots(a, p, e, x)

    y1 = (p**2-p*(r3+r4))/(1-e**2)+r3*r4
    y2 = a**2/2*(2*z_plus-z_minus)
    d1 = e*p*(r4-r3)/(1-e**2)
    d2 = -a**2*z_minus/2

    return sign(x)*sqrt(y1/y2)*elliprf(0, 1+d2/y2, 1-d2/y2)/elliprf(0, 1+d1/y1, 1-d1/y1)
    
def _rphi_frequencyRatio(a, p, e, x):
    """Ratio of r-frequency and phi-frequency 

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")
    
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    z_minus, z_plus = stable_polar_roots(a, p, e, x)
    E, L, Q = constants_of_motion(a, p, e, x)
    
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    k_theta = sqrt(z_minus / z_plus)
    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))
    
    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
    
    #theta-r ratio
    y1 = (p**2-p*(r3+r4))/(1-e**2)+r3*r4
    y2 = a**2/2*(2*z_plus-z_minus)
    d1 = e*p*(r4-r3)/(1-e**2)
    d2 = -a**2*z_minus/2
    thetar_ratio = sign(x)*sqrt(y2/y1)*elliprf(0, 1+d1/y1,1-d1/y1)/elliprf(0, 1+d2/y2, 1-d2/y2)
    
    #coefficients
    Btheta = 2 / (pi * sqrt(e0zp)) * _ellippi(z_minus, k_theta**2)
    Br = 2 * a / (
        pi * (r_plus - r_minus) * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
    ) * (
        (2 * E * r_plus - a * L)
        / (r3 - r_plus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
        - (2 * E * r_minus - a * L)
        / (r3 - r_minus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
    )
    return 1/(Btheta*thetar_ratio+Br)

def _phitheta_frequencyRatio(a, p, e, x):
    """Ratio of r-frequency and phi-frequency 

    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    double
    """
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    if not is_stable(a, p, e, x):
        raise ValueError("Not a stable orbit")
    
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    z_minus, z_plus = stable_polar_roots(a, p, e, x)
    E, L, Q = constants_of_motion(a, p, e, x)
    
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)

    k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    k_theta = sqrt(z_minus / z_plus)
    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))
    
    # simplified form of epsilon0*z_plus
    e0zp = (a**2 * (1 - E**2) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
    
    #theta-r ratio
    y1 = (p**2-p*(r3+r4))/(1-e**2)+r3*r4
    y2 = a**2/2*(2*z_plus-z_minus)
    d1 = e*p*(r4-r3)/(1-e**2)
    d2 = -a**2*z_minus/2
    rtheta_ratio = sign(x)*sqrt(y1/y2)*elliprf(0, 1+d2/y2,1-d2/y2)/elliprf(0, 1+d1/y1, 1-d1/y1)
    
    #coefficients
    Btheta = 2 / (pi * sqrt(e0zp)) * _ellippi(z_minus, k_theta**2)
    Br = 2 * a / (
        pi * (r_plus - r_minus) * sqrt((1 - E**2) * (r1 - r3) * (r2 - r4))
    ) * (
        (2 * E * r_plus - a * L)
        / (r3 - r_plus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_plus) * _ellippi(h_plus, k_r**2))
        - (2 * E * r_minus - a * L)
        / (r3 - r_minus)
        * (ellipk(k_r**2) - (r2 - r3) / (r2 - r_minus) * _ellippi(h_minus, k_r**2))
    )
    return Btheta+Br*rtheta_ratio

# Finding resonance
## In general 
def rtheta_resonance(ratio, a, e, x):
    """Return p of r-theta resonant orbit
    
    Parameters
    ----------
    ratio: double
        r-theta frequency ratio
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    
    if not valid_frequencyRatio(ratio):
        raise ValueError("The ratio must be between 0 and 1")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
    try:
        p0 = 6/(1-ratio**2)
        p1 = separatrix(a, e, x)+1e-10
        p_sol = newton(lambda p: abs(_rtheta_frequencyRatio(a, p, e, x))-ratio, x0=p0, x1=p1)
    except ValueError:
        p0 = separatrix(a, e, x)+1e-10
        p_sol = newton(lambda p: abs(_rtheta_frequencyRatio(a, p, e, x))-ratio, x0=p0)
        return p_sol    
    else:
        return p_sol

def rphi_resonance(ratio, a, e, x):
    """Return p of r-phi resonant orbit

    Parameters
    ----------
    ratio: double
        r-phi frequency ratio
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    
    if not valid_frequencyRatio(ratio):
        raise ValueError("The ratio must be between 0 and 1")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    try:
        p0 = 6/(1-ratio**2)
        p1 = separatrix(a, e, x)+1e-10
        p_sol = newton(lambda p: abs(_rphi_frequencyRatio(a, p, e, x))-ratio, x0=p0, x1=p1)
    except ValueError:
        p0 = separatrix(a, e, x)+1e-10
        p_sol = newton(lambda p: abs(_rphi_frequencyRatio(a, p, e, x))-ratio, x0=p0)
        return p_sol    
    else:
        return p_sol

def phitheta_resonance(ratio, a, e, x):
    """Return p of phi-theta resonant orbit

    Parameters
    ----------
    ratio: double
        r-phi frequency ratio
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    x : double
        cosine of the orbital inclination

    Returns
    -------
    p : double
        orbital semi-latus rectum
    """
    
    a, x = _standardize_params(a, x)
    
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = separatrix(a, e, x)+1e-10
    p_sol = newton(lambda p: abs(_phitheta_frequencyRatio(a, p, e, x))-ratio, x0=p0)
    return p_sol

## In the weak-field limit (only r-theta and r-phi)
def rtheta_resonance_weakFieldLimit(ratio, a, p, e):
    """Return approximated cos^2(theta) of r-theta resonant orbit in the weak-field limit

    Parameters
    ----------
    ratio: double
        r-phi frequency ratio
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity

    Returns
    -------
    z_minus : double
        squared cosine of the polar angle
    """
    a = abs(a)
    
    if not valid_frequencyRatio(ratio):
        raise ValueError("The ratio must be between 0 and 1")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, .5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
    p0 = 6/(1-ratio**2)
    return (
        (4 * a**2 + e**2 - (6 * e**2 + 4 * p**2) / p0) / (a**2 * (5 - 6 * e**2 / p0)) 
        - (12 * p * (2 * e**2 / p0 + 1)) / (a**2 * (5 - 6 * e**2 / p0)**2)
        + (8 / a**2) * np.sqrt(
            (p * (a**2 - e**2 + (6 * e**2 * (1 - a**2) + 4 * p**2) / p0)) / (5 - 6 * e**2 / p0)**3 
            - (4 * p**2 * (1 - 6 * e**2 / p0)) / (5 - 6 * e**2 / p0)**4
        ))

def rphi_resonance_weakFieldLimit(ratio, a, p, e, is_prograde=True):
    """Return approximated p of r-phi resonant orbit in the weak-field limit

    Parameters
    ----------
    ratio: double
        r-phi frequency ratio
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    p : double
        orbital semi-latus rectum

    Returns
    -------
    
    z_minus : double
        squared cosine of the polar angle
    """
    
    a = abs(a)
    
    if not valid_frequencyRatio(ratio):
        raise ValueError("The ratio must be between 0 and 1")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
    p0 = 6/(1-ratio**2)
    k_rtheta = lambda x: (2*(-6+p)*p+24*a*sqrt(p)*x
                          +3*a**2*(1-5*x**2+e**2*(-1+x**2))
                          )/(
                              2*p**2+3*e**2*(1+a**2*(-1+x**2)))
    Lz_sign = (int(is_prograde)+1)/2
    equation = lambda x: 1/abs(Lz_sign/sqrt(k_rtheta(x))+2*a/p**(3/2)+(-a**2*x-5*e**2*a**2*x/4)/p**2)-ratio
    x_sol = newton(equation, x0=0.5*Lz_sign)
    return 1-x_sol**2
    
# Frequency ratio of r-theta and r-phi in special limits (support finding triple resonance)
## In the polar limit
def rtheta_frequencyRatio_polarLimit(a, p, e, is_prograde=True):
    """Ratio of r-frequency and theta-frequency in the polar limit
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde

    Returns
    -------
    double
    """
    Lz_sign = -1+2*int(is_prograde)
    P = (a**2*(a**4*(-1+e**2)*2+p**4+2*a**2*p*(-2+p+e**2*(2+p)))
         )/(
             a**4*(-1+e**2)**2+2*a**2*(1+e**2)*p**2+(-4+p)*p**3)
    S = (2*(a**2*(-1+e**2)+p**2)**2
         )/(
            a**4*(-1+e**2)**2+2*a**2*(1+e**2)*p**2+(-4+p)*p**3)
    r1 = p/(1-e)
    r2 = p/(1+e)
    r3 = (S+sqrt(S**2-4*P))/2
    r4 = (S-sqrt(S**2-4*P))/2
    E2 = 1+2*(-1+e**2)/(2*p+S-e**2*S)
    Q = 2*p**2*P/(a**2*(2*p+S-e**2*S))
    #L2 = 0
    #z2 = 1
    #z1 = Q/(1-E2)/a**2/z2
    k_r = (r1-r2)/(r1-r3)*(r3-r4)/(r2-r4)
    k_theta = (1-E2)*a**2/Q
    factor = Lz_sign*sqrt((r1-r3)*(r2-r4)*2*(1-e**2)*a**2/(2*p**2*P))
    return factor*ellipk(k_theta)/ellipk(k_r)

def rphi_frequencyRatio_polarLimit(a, p, e, is_prograde=True):
    """Ratio of r-frequency and phi-frequency in the polar limit
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde

    Returns
    -------
    double
    """
    Lz_sign = -1+2*int(is_prograde)
    P = (a**2*(a**4*(-1+e**2)*2+p**4+2*a**2*p*(-2+p+e**2*(2+p)))
         )/(
             a**4*(-1+e**2)**2+2*a**2*(1+e**2)*p**2+(-4+p)*p**3)
    S = (2*(a**2*(-1+e**2)+p**2)**2
         )/(
            a**4*(-1+e**2)**2+2*a**2*(1+e**2)*p**2+(-4+p)*p**3)
    r1 = p/(1-e)
    r2 = p/(1+e)
    r3 = (S+sqrt(S**2-4*P))/2
    r4 = (S-sqrt(S**2-4*P))/2
    E2 = 1+2*(-1+e**2)/(2*p+S-e**2*S)
    Q = 2*p**2*P/(a**2*(2*p+S-e**2*S))
    #L2 = 0
    #z2 = 1
    #z1 = Q/(1-E2)/a**2/z2
    
    k_r = (r1-r2)/(r1-r3)*(r3-r4)/(r2-r4)
    k_theta = (1-E2)*a**2/Q
    factor1 = Lz_sign*sqrt((r1-r3)*(r2-r4)*2*(1-e**2)*a**2/(2*p**2*P))
    rtheta_ratio = factor1*ellipk(k_theta)/ellipk(k_r)
    
    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    h_r = (r1-r2)/(r1-r3)
    h_plus = h_r*(r3-r_plus)/(r2-r_plus)
    h_minus = h_r*(r3-r_minus)/(r2-r_minus)
    factor2 = 2*a/(pi*(r_plus-r_minus)*sqrt((1-E2)*(r1-r3)*(r2-r4)))*2*sqrt(E2)
    B_r_plus = r_plus/(r3-r_plus)*(ellipk(k_r)-(r2-r3)/(r2-r_plus)*_ellippi(h_plus, k_r))
    B_r_minus = r_minus/(r3-r_minus)*(ellipk(k_r)-(r2-r3)/(r2-r_minus)*_ellippi(h_minus, k_r))
    B_r = factor2*(B_r_plus-B_r_minus)
    #B_theta = 1
    return 1/(B_r+1/rtheta_ratio)

## In the equatorial limit
def rtheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde=True):
    """Ratio of r-frequency and theta-frequency in the equatorial limit
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde

    Returns
    -------
    double
    """
    
    Lz_sign = -1+2*int(is_prograde)
    P = 0
    S = 2*(a**4*(-1+e**2)*p+(-4+p)*p**3+a**2*p**2*(3+e**2+p)
           -Lz_sign*2*sqrt(a**2*p**3*(a**4*(-1+e**2)**2+(-4*e**2+(-2+p)**2)*p**2+2*a**2*p*(-2+p+e**2*(2+p))))
           )/(
               a**4*(-1+e**2)**2+(-4+p)**2*p**2+2*a**2*(-1+e**2)*p*(4+p)
           )
    P = 0
    r1 = p/(1-e)
    r2 = p/(1+e)
    r3 = (S+sqrt(S**2-4*P))/2
    r4 = (S-sqrt(S**2-4*P))/2
    E2 = 1+2*(-1+e**2)/(2*p+S-e**2*S)
    L2 = (2*(a**2*(-1+e**2)+p**2)*(a**2-P)+4*a**2*p*S)/(a**2*(2*p+S-e**2*S))
    #Q = 0
    #z_minus = 0
    z_plus = -(p**2+2*p*S)/(a**2*(-1+e**2))
    k_r = (r1-r2)/(r1-r3)*(r3-r4)/(r2-r4)
    k_theta = 0
    factor = Lz_sign*sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))
    return factor*ellipk(k_theta)/ellipk(k_r)

def rphi_frequencyRatio_equatorialLimit(a, p, e, is_prograde=True):
    """Ratio of r-frequency and phi-frequency in the equatorial limit
    
    Parameters
    ----------
    a : double
        dimensionless spin of the black hole
    p : double
        orbital semi-latus rectum
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde

    Returns
    -------
    double
    """
    Lz_sign = -1+2*int(is_prograde)
    P = 0
    S = 2*(a**4*(-1+e**2)*p+(-4+p)*p**3+a**2*p**2*(3+e**2+p)
           -Lz_sign*2*sqrt(a**2*p**3*(a**4*(-1+e**2)**2+(-4*e**2+(-2+p)**2)*p**2+2*a**2*p*(-2+p+e**2*(2+p))))
           )/(
               a**4*(-1+e**2)**2+(-4+p)**2*p**2+2*a**2*(-1+e**2)*p*(4+p)
           )
    P = 0
    r1 = p/(1-e)
    r2 = p/(1+e)
    r3 = (S+sqrt(S**2-4*P))/2
    r4 = (S-sqrt(S**2-4*P))/2
    E2 = 1+2*(-1+e**2)/(2*p+S-e**2*S)
    L2 = (2*(a**2*(-1+e**2)+p**2)*(a**2-P)+4*a**2*p*S)/(a**2*(2*p+S-e**2*S))
    #Q = 0
    z_minus = 0
    z_plus = -(p**2+2*p*S)/(a**2*(-1+e**2))
    k_r = (r1-r2)/(r1-r3)*(r3-r4)/(r2-r4)
    k_theta = 0
    factor1 = Lz_sign*sqrt((r1-r3)*(r2-r4)/(a**2*z_plus))
    rtheta_ratio = factor1*ellipk(k_theta)/ellipk(k_r)

    r_plus = 1+sqrt(1-a**2)
    r_minus = 1-sqrt(1-a**2)
    h_r = (r1-r2)/(r1-r3)
    h_plus = h_r*(r3-r_plus)/(r2-r_plus)
    h_minus = h_r*(r3-r_minus)/(r2-r_minus)
    factor2 = 2*a/(pi*(r_plus-r_minus)*sqrt((1-E2)*(r1-r3)*(r2-r4)))
    B_theta = 2*_ellippi(z_minus, k_theta)*sqrt(L2)/(pi*sqrt((1-E2)*a**2*z_plus))
    B_r_plus = (2*sqrt(E2)*r_plus-Lz_sign*a*sqrt(L2))/(r3-r_plus)*(ellipk(k_r)-(r2-r3)/(r2-r_plus)*_ellippi(h_plus, k_r))
    B_r_minus = (2*sqrt(E2)*r_minus-Lz_sign*a*sqrt(L2))/(r3-r_minus)*(ellipk(k_r)-(r2-r3)/(r2-r_minus)*_ellippi(h_minus, k_r))
    B_r = factor2*(B_r_plus-B_r_minus)
    return 1/(B_r+B_theta/rtheta_ratio)    

# Finding triple resonance
def tripleResonance_phiMode(rMode, thetaMode, a, e, is_prograde=True):
    """Find possible phi-modes sastify rMode*omega_r=thetaMode*omega_theta=phiMode*omega_phi.

    Parameters
    ----------
    rMode : double
        radial mode
    thetaMode : double
        polar-angle mode
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde
    
    Returns
    ------- 
    (double, double)
        Boundaries of the posible phi-mode.
    """
        
    a = abs(a)
    rtheta_ratio = thetaMode/rMode
    Lz_sign = int(is_prograde)*2-1
    
    if not valid_frequencyRatio(rtheta_ratio):
        raise ValueError("r-mode must be larger than theta-mode")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
    p1 = newton(lambda p: abs(rtheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rtheta_ratio,
                6/(1-rtheta_ratio**2))
    p2 = newton(lambda p: abs(rtheta_frequencyRatio_polarLimit(a, p, e, is_prograde))-rtheta_ratio,
                6/(1-rtheta_ratio**2))
    k1 = newton(lambda rphi_ratio: abs(rphi_frequencyRatio_equatorialLimit(a, p1, e, is_prograde))-rphi_ratio,
                rtheta_ratio)
    k2 = newton(lambda rphi_ratio: abs(rphi_frequencyRatio_polarLimit(a, p2, e, is_prograde))-rphi_ratio,
                rtheta_ratio)
    
    return sorted([k1*rMode, k2*rMode])

def tripleResonance_thetaMode(rMode, phiMode, a, e, is_prograde=True):
    """Find possible theta-modes sastify rMode*omega_r=thetaMode*omega_theta=phiMode*omega_phi.

    Parameters
    ----------
    rMode : double
        radial mode
    phiMode : double
        azimuthal-angle mode
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde
    
    Returns
    ------- 
    (double, double)
        Boundaries of the posible theta-mode.
    """
        
    a = abs(a)
    rphi_ratio = phiMode/rMode
    Lz_sign = int(is_prograde)*2-1
    
    if not valid_frequencyRatio(rphi_ratio):
        raise ValueError("r-mode must be larger than phi-mode")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
    p1 = newton(lambda p: abs(rphi_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rphi_ratio,
                6/(1-rphi_ratio**2))
    p2 = newton(lambda p: abs(rphi_frequencyRatio_polarLimit(a, p, e, is_prograde))-rphi_ratio,
                6/(1-rphi_ratio**2))
    k1 = newton(lambda rtheta_ratio: abs(rtheta_frequencyRatio_equatorialLimit(a, p1, e, is_prograde))-rtheta_ratio,
                rphi_ratio)
    k2 = newton(lambda rtheta_ratio: abs(rtheta_frequencyRatio_polarLimit(a, p2, e, is_prograde))-rtheta_ratio,
                rphi_ratio)
    
    return sorted([k1*rMode, k2*rMode])

def tripleResonance(rMode, thetaMode, phiMode, a, e, is_prograde=True):
    """Find (p, x) from resonant modes

    Parameters
    ----------
    rMode : double
        radial mode
    thetaMode : double
        polar-angle mode
    phiMode : double
        azimuthal-angle mode
    a : double
        dimensionless spin of the black hole
    e : double
        orbital eccentricity
    is_prograde : bool
        True (default) if the orbit is prograde. Otherwise, the orbit is retrograde
    
    Returns
    ------- 
    (p, x) : (double, double)
        Return orbital semi-latus rectum and cosine of the orbital inclination, respectively
    """
    
    a = abs(a)
    rtheta_ratio = thetaMode/rMode
    rphi_ratio = phiMode/rMode
    Lz_sign = int(is_prograde)*2-1
    
    
    if not valid_frequencyRatio(rtheta_ratio):
        raise ValueError("Require r-mode > theta-mode > 0")
    if not valid_frequencyRatio(rphi_ratio):
        raise ValueError("Require r-mode > phi-mode > 0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")  

    p1_rtheta = newton(lambda p: abs(rtheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rtheta_ratio,
                       6/(1-rtheta_ratio**2))
    p2_rtheta = newton(lambda p: abs(rtheta_frequencyRatio_polarLimit(a, p, e, is_prograde))-rtheta_ratio,
                       6/(1-rtheta_ratio**2))
    p1_rphi = newton(lambda p: abs(rphi_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rphi_ratio,
                    6/(1-rphi_ratio**2))
    p2_rphi = newton(lambda p: abs(rphi_frequencyRatio_polarLimit(a, p, e, is_prograde))-rphi_ratio,
                    6/(1-rphi_ratio**2))

    check_if_having_solution = (p1_rtheta-p1_rphi)*(p2_rtheta-p2_rphi)

    
    if (check_if_having_solution > 0) | (check_if_having_solution == nan):
        return [nan, nan]

    elif check_if_having_solution < 0:
        if abs(p1_rtheta-p1_rphi) < abs(p2_rtheta-p2_rphi):
            p0 = (p1_rtheta+p1_rphi)/2
        else: 
            p0 = (p2_rtheta+p2_rphi)/2
        x0 = Lz_sign*0.5
        sol = fsolve(lambda X: [abs(_rtheta_frequencyRatio(a, X[0], e, X[1]))-rtheta_ratio,
                                abs(_rphi_frequencyRatio(a, X[0], e, X[1]))-rphi_ratio], [p0, x0])
        return sol
    
    else:
        if (p1_rtheta-p1_rphi) == 0:
            return [p1_rtheta, Lz_sign]
        else:
            return [p2_rtheta, 0]