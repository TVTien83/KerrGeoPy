"""Module containing functions for finding resonance orbits
"""
from .frequencies import *
from .frequencies import _ellippi
from .constants import *
from .constants import _standardize_params
from scipy.optimize import fsolve, newton, root_scalar
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
    
    #simplified form of epsilon0*z_plus
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

def _doubleResonance_solver(equation, p0, sep):
    '''Try `scipy.newton` with two different initial guess, one of them is very near the separatrix. Also try solving by using `scipy.root_scalar(method="brentq")`
    
    Parameters
    ----------
    equation: callable
        resonant condition equation needed to be solved
    p0 : double
        initial guess p of the first attemp
    sep : double
        initial guess p of the second attemp, which also is p at the separatrix
    
    Returns
    -------
    p : double
        orbital semi-latus rectum
    '''
    try:
        p_sol = newton(equation, p0)
    except RuntimeError:
        try:
            p_sol = root_scalar(equation, method="brentq", bracket=(sep+1e-5, p0)).root
        except:
            try:
                p_sol = newton(equation, sep+1e-5)
            except RuntimeError:
                pass
            else:
                return p_sol
        else:
            return p_sol
    else:
        return p_sol
    raise RuntimeError("p is too close to the separatrix")
    
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

    return _doubleResonance_solver(lambda p: abs(_rtheta_frequencyRatio(a, p, e, x))-ratio,
                                   6/(1-ratio**2),
                                   separatrix(a, e, x))

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

    return _doubleResonance_solver(lambda p: abs(_rphi_frequencyRatio(a, p, e, x))-ratio,
                                   6/(1-ratio**2),
                                   separatrix(a, e, x))

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
    if (ratio <= 0) | (ratio == 1):
        raise ValueError("The ratio must be positive and not equal to 1")
    if (ratio > 1) & (x < 0):
        raise ValueError("The ratio must be larger than 1 if the orbit is prograde")
    if (ratio < 1) & (x > 0):
        raise ValueError("The ratio must be lesser than 1 if the orbit is retrograde")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if x == 0:
        raise ValueError("Polar orbits not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    return _doubleResonance_solver(lambda p: abs(_phitheta_frequencyRatio(a, p, e, x))-ratio,
                                   (2*a/abs(ratio-1))**(2/3),
                                   separatrix(a, e, x))

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
    
    a = abs(a)
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
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

def phitheta_frequencyRatio_polarLimit(a, p, e, is_prograde=True):
    """Ratio of phi-frequency and theta-frequency in the polar limit
    
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
    a = abs(a)
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
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
    return B_r*rtheta_ratio+1


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
    a = abs(a)
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
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
    a = abs(a)
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
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

def phitheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde=True):
    """Ratio of phi-frequency and theta-frequency in the equatorial limit
    
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
    a = abs(a)
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    
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
    return B_r*rtheta_ratio+B_theta

# Finding triple resonance
def tripleResonance_rMode(thetaMode, phiMode, a, e, is_prograde=True):
    """Find possible r-modes sastify rMode*omega_r=thetaMode*omega_theta=phiMode*omega_phi.

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
    phitheta_ratio = thetaMode/phiMode
    Lz_sign = int(is_prograde)*2-1
    
    if (phitheta_ratio <= 0) | (phitheta_ratio == 1):
        raise ValueError("The ratio must be positive and not equal to 1")
    if (phitheta_ratio > 1) & (not is_prograde):
        raise ValueError("The ratio must be larger than 1 if the orbit is prograde")
    if (phitheta_ratio < 1) & is_prograde:
        raise ValueError("The ratio must be lesser than 1 if the orbit is retrograde")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p1 = _doubleResonance_solver(lambda p: abs(phitheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-phitheta_ratio,
                                 (2*a/abs(phitheta_ratio-1))**(2/3),
                                 separatrix(a, e, Lz_sign))
    rtheta_ratio1 = abs(rtheta_frequencyRatio_equatorialLimit(a, p1, e, is_prograde))
    
    p2 = _doubleResonance_solver(lambda p: abs(phitheta_frequencyRatio_polarLimit(a, p, e, is_prograde))-phitheta_ratio,
                                 (2*a/abs(phitheta_ratio-1))**(2/3),
                                 separatrix(a, e, 0))
    rtheta_ratio2 = abs(rtheta_frequencyRatio_polarLimit(a, p2, e, is_prograde))
    
    return sorted([thetaMode/rtheta_ratio1, thetaMode/rtheta_ratio2])

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

    p1 = _doubleResonance_solver(lambda p: abs(rphi_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rphi_ratio,
                                 6/(1-rphi_ratio**2),
                                 separatrix(a, e, Lz_sign))
    rtheta_ratio1 = abs(rtheta_frequencyRatio_equatorialLimit(a, p1, e, is_prograde))
    
    p2 = _doubleResonance_solver(lambda p: abs(rphi_frequencyRatio_polarLimit(a, p, e, is_prograde))-rphi_ratio,
                                 6/(1-rphi_ratio**2),
                                 separatrix(a, e, 0))
    rtheta_ratio2 = abs(rtheta_frequencyRatio_polarLimit(a, p2, e, is_prograde))
    
    return sorted([rtheta_ratio1*rMode, rtheta_ratio2*rMode])

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

    p1 = _doubleResonance_solver(lambda p: abs(rtheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rtheta_ratio,
                                 6/(1-rtheta_ratio**2),
                                 separatrix(a, e, Lz_sign))
    rphi_ratio1 = abs(rphi_frequencyRatio_equatorialLimit(a, p1, e, is_prograde))
    
    p2 = _doubleResonance_solver(lambda p: abs(rtheta_frequencyRatio_polarLimit(a, p, e, is_prograde))-rtheta_ratio,
                                 6/(1-rtheta_ratio**2),
                                 separatrix(a, e, 0))
    rphi_ratio2 = abs(rphi_frequencyRatio_polarLimit(a, p2, e, is_prograde))
    
    return sorted([rphi_ratio1*rMode, rphi_ratio2*rMode])

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

    sep1 = separatrix(a, e, Lz_sign)
    sep2 = separatrix(a, e, 0)
    p0_rtheta = 6/(1-rtheta_ratio**2)
    p0_rphi = 6/(1-rphi_ratio**2)
    p1_rtheta = _doubleResonance_solver(lambda p: abs(rtheta_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rtheta_ratio, p0_rtheta, sep1)
    p1_rphi = _doubleResonance_solver(lambda p: abs(rphi_frequencyRatio_equatorialLimit(a, p, e, is_prograde))-rphi_ratio, p0_rphi, sep1)
    p2_rtheta = _doubleResonance_solver(lambda p: abs(rtheta_frequencyRatio_polarLimit(a, p, e, is_prograde))-rtheta_ratio, p0_rtheta, sep2)
    p2_rphi = _doubleResonance_solver(lambda p: abs(rphi_frequencyRatio_polarLimit(a, p, e, is_prograde))-rphi_ratio, p0_rphi, sep2)    
                                   
    check_if_having_solution = (p1_rtheta-p1_rphi)*(p2_rtheta-p2_rphi)
    
    if (check_if_having_solution > 0) | (check_if_having_solution == nan):
        return [nan, nan]
    elif check_if_having_solution < 0:
        x0 = Lz_sign/((p1_rphi-p1_rtheta)/(p2_rtheta-p2_rphi)+1)
        p0 = p2_rtheta-Lz_sign*x0*(p2_rtheta-p1_rtheta)
        sol = fsolve(lambda X: [abs(_rtheta_frequencyRatio(a, X[0], e, X[1]))-rtheta_ratio,
                                abs(_rphi_frequencyRatio(a, X[0], e, X[1]))-rphi_ratio], [p0, x0])
        return sol
    else:
        if (p1_rtheta-p1_rphi) == 0:
            return [p1_rtheta, Lz_sign]
        else:
            return [p2_rtheta, 0]