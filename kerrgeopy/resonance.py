"""Module containing functions for finding resonance orbits
"""
from .frequencies import *
from .frequencies import _ellippi
from .constants import *
from .constants import _standardize_params
from scipy.optimize import fsolve, newton, root_scalar
from scipy.special import elliprf, ellipk
from numpy import sqrt, pi, sign

def valid_frequencyRatio(ratio):
    if (0 < abs(ratio)) & (abs(ratio) < 1):
        return True
    else:
        return False

def valid_integers(r, phi, theta):
    if r < 0:
        return False
    if (r!=0) & (phi!=0) & (theta!=0):
        if (r<abs(phi)) & (r<abs(theta)):
            return True
    elif (r==0) & (phi!=0) & (theta!=0):
        if phi*theta>0:
            return True
    elif (r!=0) & (phi==0) & (theta!=0):
        if r<abs(theta):
            return True
    elif (r!=0) & (phi!=0) & (theta==0):
        if r<abs(phi):
            return True
    else:
        return False

# Frequency ratio functions
## Needed functions
def _r3r4_prod(a, p, e, x):
    """
    r3*r4
    """
    numerator_term1 = ((-4 + p) * p**7 + a**8 * (-1 + e**2)**4 * (-1 + x**2)**2 + 
         2 * a**6 * (-1 + e**2)**2 * p * (1 - x**2) * ((1 + e**2) * p + (-2 + p + e**2 * (2 + p)) * (1 - x**2)) + 
         a**4 * p**3 * ((-3 + 2 * e**2 + e**4) * p + 4 * (-4 + 3 * p + e**4 * (4 + p)) * (1 - x**2) + 
         (-1 + e**2) * (-4 + e**2 * (-12 + p) + 3 * p) * (-1 + x**2)**2))
    numerator_term2 =  ((-1 - e**2) * p**4 * (-2 + x**2) + 
          8 * (-1 + e**2) * p**2 * (-1 + x**2) + 
          2 * p**3 * (-3 - e**2 + 4 * (1 + e**2) * x**2))
    numerator_term3 = (-1/(a**2*p))*(
        (a**4*(-1 + e**2)**2 + 
        (-4*e**2 + (-2 + p)**2)*p**2 + 
        2*a**2*p*(-2 + p + e**2*(2 + p)))*x**2*(-p**2 + a**2*(-1 + e)**2*(-1 + x**2))*
        (-p**2 + a**2*(1 + e)**2*(-1 + x**2))*(-p**2 + a**2*(-1 + e**2)*(-1 + x**2)))
    if x>=0:
        numerator = a**2*(1-x**2)*(numerator_term1+2*a**2*p**2*(numerator_term2-4*sqrt(numerator_term3)))
    else:
        numerator = a**2*(1-x**2)*(numerator_term1+2*a**2*p**2*(numerator_term2+4*sqrt(numerator_term3)))
    denominator = ((-4 + p)**2 * p**6 + a**8 * (-1 + e**2)**4 * (-1 + x**2)**2 + 
         2 * a**2 * p**5 * ((-1 + e**2) * (4 + p) + (-4 + e**2 * (-12 + p) + 3 * p) * (1 - x**2)) + 
         2 * a**6 * (-1 + e**2)**2 * p**2 * (-1 + x**2) * (-2 + 3 * x**2 + e**2 * (-2 + x**2)) + 
         a**4 * p**3 * (-8 * (-1 + e**2)**2 * (1 - 3 * x**2 + 2 * x**4) + 
         p * ((-1 + e**2)**2 - 4 * (-1 + e**4) * (-1 + x**2) + (3 + e**2)**2 * (-1 + x**2)**2)))

    return numerator/denominator

def _r3r4_prod_reduced(a, p, e, x):
    """
    r3*r4/(1-x^2)
    """
    numerator_term1 = ((-4 + p) * p**7 + a**8 * (-1 + e**2)**4 * (-1 + x**2)**2 + 
         2 * a**6 * (-1 + e**2)**2 * p * (1 - x**2) * ((1 + e**2) * p + (-2 + p + e**2 * (2 + p)) * (1 - x**2)) + 
         a**4 * p**3 * ((-3 + 2 * e**2 + e**4) * p + 4 * (-4 + 3 * p + e**4 * (4 + p)) * (1 - x**2) + 
         (-1 + e**2) * (-4 + e**2 * (-12 + p) + 3 * p) * (-1 + x**2)**2))
    numerator_term2 =  ((-1 - e**2) * p**4 * (-2 + x**2) + 
          8 * (-1 + e**2) * p**2 * (-1 + x**2) + 
          2 * p**3 * (-3 - e**2 + 4 * (1 + e**2) * x**2))
    numerator_term3 = (-1/(a**2*p))*(
        (a**4*(-1 + e**2)**2 + 
        (-4*e**2 + (-2 + p)**2)*p**2 + 
        2*a**2*p*(-2 + p + e**2*(2 + p)))*x**2*(-p**2 + a**2*(-1 + e)**2*(-1 + x**2))*
        (-p**2 + a**2*(1 + e)**2*(-1 + x**2))*(-p**2 + a**2*(-1 + e**2)*(-1 + x**2)))
    if x>=0:
        numerator = a**2*(numerator_term1+2*a**2*p**2*(numerator_term2-4*sqrt(numerator_term3)))
    else:
        numerator = a**2*(numerator_term1+2*a**2*p**2*(numerator_term2+4*sqrt(numerator_term3)))
    denominator = ((-4 + p)**2 * p**6 + a**8 * (-1 + e**2)**4 * (-1 + x**2)**2 + 
         2 * a**2 * p**5 * ((-1 + e**2) * (4 + p) + (-4 + e**2 * (-12 + p) + 3 * p) * (1 - x**2)) + 
         2 * a**6 * (-1 + e**2)**2 * p**2 * (-1 + x**2) * (-2 + 3 * x**2 + e**2 * (-2 + x**2)) + 
         a**4 * p**3 * (-8 * (-1 + e**2)**2 * (1 - 3 * x**2 + 2 * x**4) + 
         p * ((-1 + e**2)**2 - 4 * (-1 + e**4) * (-1 + x**2) + (3 + e**2)**2 * (-1 + x**2)**2)))
    return numerator/denominator

def _r3r4_sum(a, p, e, x):
    if a != 0:
        return (-p**2+a**2*(1-e**2)*(1-x**2))*(a**2-_r3r4_prod_reduced(a,p,e,x))/(2*a**2*p)
    else:
        return 2*p/(p-4)

def _r3(a, p, e, x):
    delta = _r3r4_sum(a,p,e,x)**2-4*_r3r4_prod(a,p,e,x)
    return (_r3r4_sum(a,p,e,x)+sqrt(delta))/2

def _r4(a, p, e, x):
    delta = _r3r4_sum(a,p,e,x)**2-4*_r3r4_prod(a,p,e,x)
    return (_r3r4_sum(a,p,e,x)-sqrt(delta))/2

def _E2(a, p, e, x):
    """
    Squared energy
    """
    return 1-2*(1-e**2)/(2*p+(1-e**2)*_r3r4_sum(a,p,e,x))

def _L2(a, p, e, x):
    """
    Squared angular momentum
    """
    numerator = 2*(a**2*(-1+e**2)+p**2)*(a**2-_r3r4_prod(a,p,e,x))+4*a**2*p*_r3r4_sum(a,p,e,x)
    denominator = a**2*(2*p+(1-e**2)*_r3r4_sum(a,p,e,x))
    return numerator/denominator

def _del1y1(a, p, e, x):
    numerator = -e*p*sqrt(_r3r4_sum(a,p,e,x)**2-4*_r3r4_prod(a,p,e,x))
    denominator = (1-e**2)*_r3r4_prod(a,p,e,x)+p**2-p*_r3r4_sum(a,p,e,x)
    return numerator/denominator

def _del2y2(a, p, e, x):
    numerator = (1-e**2)*a**4*(1-x**2)
    denominator = (1-e**2)*a**4*(1-x**2)-2*p**2*_r3r4_prod_reduced(a,p,e,x)
    return numerator/denominator

def _y1y2(a, p, e, x):
    numerator = 2*a**2*((1-e**2)*_r3r4_prod(a,p,e,x)+p**2-p*_r3r4_sum(a,p,e,x))
    denominator = a**4*(e**2-1)*(1-x**2)+2*p**2*_r3r4_prod_reduced(a,p,e,x)
    return numerator/denominator

def _kr(a,p,e,x):
    numerator = (p-p*(1-e)/(1+e))*(_r3(a,p,e,x)-_r4(a,p,e,x))
    denominator = (p-_r3(a,p,e,x)*(1-e))*(p/(1+e)-_r4(a,p,e,x))
    return numerator/denominator

def _ktheta(a,p,e,x):
    return (1-x**2)*(1-e**2)*a**4/(p**2*_r3r4_prod_reduced(a,p,e,x))

def _rPlus(a):
    return 1+sqrt(1-a**2)

def _rMinus(a):
    return 1-sqrt(1-a**2)

def _hPlus(a,p,e,x):
    numerator = (p-p*(1-e)/(1+e))*(_r3(a,p,e,x)-_rPlus(a))
    denominator = (p-_r3(a,p,e,x)*(1-e))*(p/(1+e)-_rPlus(a))
    return numerator/denominator

def _hMinus(a,p,e,x):
    numerator = (p-p*(1-e)/(1+e))*(_r3(a,p,e,x)-_rMinus(a))
    denominator = (p-_r3(a,p,e,x)*(1-e))*(p/(1+e)-_rMinus(a))
    return numerator/denominator

def _phiTerm_thetaPrefactor(a,p,e,x):
    numerator = (a**2*(-1+e**2)+p**2)*(a**2-_r3r4_prod(a,p,e,x))+2*a**2*p*_r3r4_sum(a,p,e,x)
    denominator = p**2*_r3r4_prod_reduced(a,p,e,x)
    return 2/pi*sqrt(numerator/denominator)

def _phiTerm_rPrefactor(a,p,e,x):
    term1 = 2*a/(pi*(_rPlus(a)-_rMinus(a)))
    term2_numerator = 2*p+(1-e**2)*_r3r4_sum(a,p,e,x)
    term2_denominator = 2*(p-_r3(a,p,e,x)*(1-e))*(p-_r4(a,p,e,x)*(1+e))
    return term1*sqrt(term2_numerator/term2_denominator)

def _phiTerm_rPrefactorPlus(a,p,e,x):
    term1 = (2*sqrt(_E2(a,p,e,x))*_rPlus(a)-a*sign(x)*sqrt(abs(_L2(a,p,e,x))))/(_r3(a,p,e,x)-_rPlus(a))
    term2 = ellipk(_kr(a,p,e,x))-(p/(1+e)-_r3(a,p,e,x))/(p/(1+e)-_rPlus(a))*_ellippi(_hPlus(a,p,e,x),_kr(a,p,e,x))
    return term1*term2

def _phiTerm_rPrefactorMinus(a,p,e,x):
    term1 = (2*sqrt(_E2(a,p,e,x))*_rMinus(a)-a*sign(x)*sqrt(abs(_L2(a,p,e,x))))/(_r3(a,p,e,x)-_rMinus(a))
    term2 = ellipk(_kr(a,p,e,x))-(p/(1+e)-_r3(a,p,e,x))/(p/(1+e)-_rMinus(a))*_ellippi(_hMinus(a,p,e,x),_kr(a,p,e,x))
    return term1*term2

def _phiTerm_thetaTerm(a,p,e,x):
    return _phiTerm_thetaPrefactor(a,p,e,x)*_ellippi(1-x**2,_ktheta(a,p,e,x))

def _phiTerm_rTerm(a,p,e,x):
    return _phiTerm_rPrefactor(a,p,e,x)*(_phiTerm_rPrefactorPlus(a,p,e,x)-_phiTerm_rPrefactorMinus(a,p,e,x))

## frequency ratios
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
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    return sqrt(_y1y2(a,p,e,x))*elliprf(0, 1+_del2y2(a,p,e,x), 1-_del2y2(a,p,e,x))/elliprf(0, 1+_del1y1(a,p,e,x), 1-_del1y1(a,p,e,x))

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
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    if x != 0: 
        return sign(x)/(sign(x)*_phiTerm_thetaTerm(a,p,e,x)/_rtheta_frequencyRatio(a,p,e,x)+_phiTerm_rTerm(a,p,e,x))
    else:
        return [1/(1/_rtheta_frequencyRatio(a,p,e,x)+_phiTerm_rTerm(a,p,e,x)), -1/(-1/_rtheta_frequencyRatio(a,p,e,x)+_phiTerm_rTerm(a,p,e,x))]

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
    if not is_stable(a,p,e,x):
        raise ValueError("Not a stable orbit")
    if x != 0: 
        return _phiTerm_thetaTerm(a,p,e,x)+sign(x)*_rtheta_frequencyRatio(a,p,e,x)*_phiTerm_rTerm(a,p,e,x)
    else:
        return [1+_rtheta_frequencyRatio(a,p,e,x)*_phiTerm_rTerm(a,p,e,x),1-_rtheta_frequencyRatio(a,p,e,x)*_phiTerm_rTerm(a,p,e,x)]

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
    if p0 < sep:
        try:
            return newton(equation, sep+1e-5)
        except (RuntimeError, ValueError):
            pass
    try:
        return newton(equation, p0)
    except (RuntimeError, ValueError):
        try:
            return root_scalar(equation, method="brentq", bracket=(sep+1e-5, p0)).root
        except:
            try:
                return newton(equation, sep+1e-5)
            except (RuntimeError, ValueError):
                pass
    print("p is too close to the separatrix or no resonance. The result is set to sep+1e-5")
    return sep+1e-5

# find resonance
## r-theta resonance
def _rtheta_resonance_p(a, e, x, rInteger, thetaInteger):
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
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/thetaInteger)")
    return _doubleResonance_solver(lambda p: _rtheta_frequencyRatio(a, p, e, x)-abs(ratio),
                                   6/(1-ratio**2),
                                   separatrix(a, e, x))

def _rtheta_resonance_e(a, p, x, rInteger, thetaInteger):
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
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/thetaInteger)")
    p0 = _rtheta_resonance_p(a, 0, x, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, 1, x, rInteger, thetaInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    elif p == p1:
        return 1
    else:
        e0 = (p-p0)/(p1-p0)
        return newton(lambda e: _rtheta_frequencyRatio(a, p, e, x)-abs(ratio), e0)

def _rtheta_resonance_x(a, p, e, rInteger, thetaInteger):
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
    
    a = abs(a)
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported") 
    ratio = rInteger/thetaInteger
    Lz_sign = sign(ratio)
    p0 = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, e, Lz_sign, rInteger, thetaInteger)
    if (ratio>0)&((p<p1)|(p0<p)):
        print(f"p must be in [{p1:f},{p0:f}]")
        return nan
    if (ratio<0)&((p<p0)|(p1<p)):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    elif p == p1:
        return 1
    else:
        x0 = Lz_sign*(p-p0)/(p1-p0)
        return newton(lambda x: _rtheta_frequencyRatio(a, p, e, x)-abs(ratio), x0)

## r-phi resonance

def _rphi_resonance_p(a, e, x, rInteger, phiInteger):
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
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    if x == 0:
        if ratio > 0:
            resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, 0)[0]-abs(ratio)
        elif ratio < 0:
            resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, 0)[1]-abs(ratio)
    else:
        if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/phiInteger)")
        resonantEquation = lambda p: _rphi_frequencyRatio(a, p, e, x)-abs(ratio)
    p0 = 6/(1-ratio**2)
    pSep = separatrix(a, e, x)
    return _doubleResonance_solver(resonantEquation, p0, pSep)

def _rphi_resonance_e(a, p, x, rInteger, phiInteger):
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
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    p0 = _rphi_resonance_p(a, 0, x, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, 1, x, rInteger, phiInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    e0 = (p-p0)/(p1-p0)
    if x == 0:
        if ratio > 0:
            resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)[0]-abs(ratio)
        elif ratio < 0:
            resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)[1]-abs(ratio)
    else:
        if sign(x)*sign(ratio)<0:
            raise ValueError("Require sign(x) = sign(rInteger/phiInteger)")
        resonantEquation = lambda e: _rphi_frequencyRatio(a, p, e, x)-abs(ratio)
    return newton(resonantEquation, e0)

def _rphi_resonance_x(a, p, e, rInteger, phiInteger):
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
    
    a = abs(a)
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = rInteger/phiInteger
    Lz_sign = sign(ratio)
    p0 = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, e, Lz_sign, rInteger, phiInteger)
    if (ratio>0)&((p<p1)|(p0<p)):
        print(f"p must be in [{p1:f},{p0:f}]")
        return nan
    if (ratio<0)&((p<p0)|(p1<p)):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    x0 = Lz_sign*(p-p0)/(p1-p0)
    resonantEquation = lambda x: _rphi_frequencyRatio(a, p, e, x)-abs(ratio)
    return newton(resonantEquation, x0)

## phi-theta resonance

def _phitheta_resonance_p(a, e, x, thetaInteger, phiInteger):
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
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if not valid_params(a, e, x):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")
    ratio = phiInteger/thetaInteger
    if x == 0:
        if ratio > 1:
            resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, 0)[0]-ratio
        elif ratio < 1:
            resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, 0)[1]-ratio
    else:
        if sign(x)*sign(ratio-1)<0:
            raise ValueError("Require sign(x) = sign(phiInteger/thetaInteger-1)")
        resonantEquation = lambda p: _phitheta_frequencyRatio(a, p, e, x)-ratio
    p0 = (2*a/abs(ratio-1))**(2/3)
    pSep = separatrix(a, e, x)
    return _doubleResonance_solver(resonantEquation, p0, pSep)

def _phitheta_resonance_e(a, p, x, thetaInteger, phiInteger):
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
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = phiInteger/thetaInteger
    p0 = _phitheta_resonance_p(a, 0, x, thetaInteger, phiInteger)
    p1 = _phitheta_resonance_p(a, 1, x, thetaInteger, phiInteger)
    if (p<p0)|(p1<p):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    e0 = (p-p0)/(p1-p0)
    if x == 0:
        if ratio > 1:
            resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)[0]-abs(ratio)
        elif ratio < 1:
            resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)[1]-abs(ratio)
    else:
        if sign(x)*sign(ratio-1)<0:
            raise ValueError("Require sign(x) = sign(phiInteger/thetaInteger-1)")
        resonantEquation = lambda e: _phitheta_frequencyRatio(a, p, e, x)-abs(ratio)
    return newton(resonantEquation, e0)

def _phitheta_resonance_x(a, p, e, thetaInteger, phiInteger):
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
    
    a = abs(a)
    
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    ratio = phiInteger/thetaInteger
    Lz_sign = sign(ratio-1)
    p0 = _phitheta_resonance_p(a, e, 0, thetaInteger, phiInteger)
    p1 = _phitheta_resonance_p(a, e, Lz_sign, thetaInteger, phiInteger)
    if (ratio>1)&((p<p1)|(p0<p)):
        print(f"p must be in [{p1:f},{p0:f}]")
        return nan
    if (ratio<1)&((p<p0)|(p1<p)):
        print(f"p must be in [{p0:f},{p1:f}]")
        return nan
    if p == p0:
        return 0
    if p == p1:
        return 1
    x0 = Lz_sign*(p-p0)/(p1-p0)
    resonantEquation = lambda x: _phitheta_frequencyRatio(a, p, e, x)-abs(ratio)
    return newton(resonantEquation, x0)

# Finding triple resonance
def _tripleResonance_rInteger(thetaInteger, phiInteger, a, e):
    """Find possible r-integers sastify omega_r:omega_theta:omega_phi = r-integer:theta-integer:phi-integer.

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
    phitheta_ratio = phiInteger/thetaInteger
    Lz_sign = sign(phitheta_ratio-1)
    
    if not valid_integers(0, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _phitheta_resonance_p(a, e, 0, thetaInteger, phiInteger)
    p1 = _phitheta_resonance_p( a, e, Lz_sign, thetaInteger, phiInteger)
    rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
    rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, e, Lz_sign)
    
    return sorted([thetaInteger*rtheta_ratio0, thetaInteger*rtheta_ratio1])

def _tripleResonance_thetaInteger(rInteger, phiInteger, a, e):
    """Find possible theta-integers sastify omega_r:omega_theta:omega_phi = r-integer:theta-integer:phi-integer.

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
    rphi_ratio = rInteger/phiInteger
    Lz_sign = sign(rphi_ratio)
    
    if not valid_integers(rInteger, 0, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1 = _rphi_resonance_p(a, e, Lz_sign, rInteger, phiInteger)
    rtheta_ratio0 = _rtheta_frequencyRatio(a, p0, e, 0)
    rtheta_ratio1 = _rtheta_frequencyRatio(a, p1, e, Lz_sign)
    
    return sorted([rInteger/rtheta_ratio0, rInteger/rtheta_ratio1])

def _tripleResonance_phiInteger(rInteger, thetaInteger, a, e):
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
    rtheta_ratio = rInteger/thetaInteger
    Lz_sign = rtheta_ratio
    
    if not valid_integers(rInteger, thetaInteger, 0):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")

    p0 = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1 = _rtheta_resonance_p(a, e, Lz_sign, rInteger, thetaInteger)
    if Lz_sign > 0:
        rphi_ratio0 = _rphi_frequencyRatio(a, p0, e, 0)[0]
    elif Lz_sign < 0:
        rphi_ratio0 = _rphi_frequencyRatio(a, p0, e, 0)[1]
    rphi_ratio1 = _rphi_frequencyRatio(a, p1, e, Lz_sign)
    
    return sorted([rInteger/rphi_ratio0, rInteger/rphi_ratio1])

def _tripleResonance(a, e, rInteger, thetaInteger, phiInteger):
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
    rtheta_ratio = rInteger/thetaInteger
    rphi_ratio = rInteger/phiInteger
    Lz_sign = sign(rtheta_ratio)
    
    if not valid_integers(rInteger, thetaInteger, phiInteger):
        raise ValueError("Require |rInteger| > |thetaInteger|, |rInteger| > |phiInteger| and thetaInteger*phiInteger>0")
    if a == 1:
        raise ValueError("Extreme Kerr not supported")
    if e == 1:
        raise ValueError("Marginally bound orbits not supported")
    if not valid_params(a, e, 0.5):
        raise ValueError("a^2, e and x^2 must be between 0 and 1")  

    p0_rtheta = _rtheta_resonance_p(a, e, 0, rInteger, thetaInteger)
    p1_rtheta = _rtheta_resonance_p(a, e, Lz_sign, rInteger, thetaInteger)
    p0_rphi = _rphi_resonance_p(a, e, 0, rInteger, phiInteger)
    p1_rphi = _rphi_resonance_p(a, e, Lz_sign, rInteger, phiInteger)
                                   
    check_if_having_solution = (p0_rtheta-p0_rphi)*(p1_rtheta-p1_rphi)
    
    if (check_if_having_solution > 0) | (check_if_having_solution == nan):
        rs = _tripleResonance_rInteger(thetaInteger, phiInteger, a, e)
        thetas = _tripleResonance_phiInteger(rInteger, thetaInteger, a, e)
        phis = _tripleResonance_thetaInteger(rInteger, phiInteger, a, e)
        print("No triple resonance, try changing one of the integers:")
        print(f"Choose r-integer in [{rs[0]:f}, {rs[1]:f}]")
        print(f"Choose phi-integer in [{thetas[0]:f}, {thetas[1]:f}]")
        print(f"Choose theta-integer in [{phis[0]:f}, {phis[1]:f}]")
        return [nan, nan]
    elif check_if_having_solution < 0:
        x0 = Lz_sign/((p1_rphi-p1_rtheta)/(p0_rtheta-p0_rphi)+1)
        p0 = p0_rtheta-Lz_sign*x0*(p0_rtheta-p1_rtheta)
        sol = fsolve(lambda X: [_rtheta_frequencyRatio(a, X[0], e, X[1])-abs(rtheta_ratio),
                                _rphi_frequencyRatio(a, X[0], e, X[1])-abs(rphi_ratio)], [p0, x0])
        return sol
    else:
        if (p0_rtheta-p0_rphi) == 0:
            return [p0_rtheta, Lz_sign]
        else:
            return [p1_rtheta, 0]
        
# Generic function

def findResonance(a, p=None, e=None, x=None, integers=[0,0,0]):
    rInteger, thetaInteger, phiInteger = integers
    if (rInteger!=0) & (thetaInteger!=0) & (phiInteger!=0):
        if e != None:
            return _tripleResonance(a, e, rInteger, thetaInteger, phiInteger)
        else: 
            raise ValueError("Only support finding (p,x) with given (a,e)")
        
    elif (rInteger==0) & (thetaInteger!=0) & (phiInteger!=0):
        if (p==None) & (e!=None) & (x!=None):
            return _phitheta_resonance_p(a, e, x, thetaInteger, phiInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _phitheta_resonance_e(a, p, x, thetaInteger, phiInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _phitheta_resonance_x(a, p, e, thetaInteger, phiInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    elif (rInteger!=0) & (thetaInteger==0) & (phiInteger!=0):
        if (p==None) & (e!=None) & (x!=None):
            return _rphi_resonance_p(a, e, x, rInteger, phiInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _rphi_resonance_e(a, p, x, rInteger, phiInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _rphi_resonance_x(a, p, e, rInteger, phiInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    elif (rInteger!=0) & (thetaInteger!=0) & (phiInteger==0):
        if (p==None) & (e!=None) & (x!=None):
            return _rtheta_resonance_p(a, e, x, rInteger, thetaInteger)
        elif (p!=None) & (e==None) & (x!=None):
            return _rtheta_resonance_e(a, p, x, rInteger, thetaInteger)
        elif (p!=None) & (e!=None) & (x==None):
            return _rtheta_resonance_x(a, p, e, rInteger, thetaInteger)
        else: 
            raise ValueError("Require 2 of (p, e, x)")
        
    else:
        raise ValueError("At least 2 of the integers are non-zero")        