'''
Spin-dependent potential, based off from the OBEPot class from thre-boson case
'''

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
from scipy.special import legendre
from sympy.physics.quantum.cg import CG, Wigner9j, Wigner6j
import math


class OBEpotSpin:
    '''
    OBE potential including spin state of mediator

    This just means to include the Wigner9j and 6j symbols 
    and consider for higher powers of q since we now also consider for l=1 states.
    - This means we also need an additional coefficient C1
    '''

    # this are common parameters for all instances 
    hbarc=197.327
    
    # init interaction
    def __init__(self, cutoff=500.0, C0=1.0, C1=1.0, nx=12,mpi=138.0,A=-1.0):
        """Defines the one boson exchange for a given regulator, coupling strength and short distance parameter
        
        Parameters:
        cutoff -- regulator in MeV
        C0 -- strength of the short distance counter term (in s-wave) 
        A -- strength of OBE
        nx -- number of angular grid points for numerical integration
        mpi -- mass of exchange boson in MeV"""
        
        self.mpi = mpi/self.hbarc
        self.cutoff = cutoff/self.hbarc
        self.C0=C0
        self.C1 = C1
        self.A=A
        self.nx=nx
        
        self.xp=np.empty((self.nx),dtype=np.double)
        self.xw=np.empty((self.nx),dtype=np.double)
        self.xp,self.xw=leggauss(self.nx)
    
    
    
    # function defines the x integral 
    def _g(self,pp,p,k,m):
        """Calculates g function of the partial wave decomposition of OBE. 
        
           pp -- outgoing momentum 
           p -- incoming momentum
           k -- angular momentum
           m --  powers of q in long range part"""
        
        # define prefact 
        # required to add factor from spherical harmonics
        prefact=np.sqrt(2*k+1.)/2*(-1)**k*4*math.pi
        # get the corresponding legendre polynomial 
        Pk = legendre(k)
        # define momentum transfer dependent on angles 
        qval=np.sqrt(p**2+pp**2-2*p*pp*self.xp)
        
        # build integral of regularized OBE 
        return float(prefact*np.sum(Pk(self.xp)*qval**(2-m)/((qval**2+self.mpi**2))*self.xw*np.exp(-(qval**2+self.mpi**2)/self.cutoff**2)))
        
    # determines complete, regularized interaction     
    def v(self,pp,p,l,s,j,t):
        """Potential matrix element in fm**2
        
           pp -- outgoing momentum in fm**-1
           p -- incoming momentum in fm**-1
           l -- angular momentum
           s, j, t -- spin, total ang momentum, isospin
           """ 
        
        # first overall prefact of 1pi exchange part  (cancel 2pi factors!)
        prefact=self.A
        
        # need to consider Wigner symbols here
        
        # iterate over different powers in q
        # only two lowest powers needed since we consider up to l = 1
        mat = 0
        for m in [0,2]:   
            # wigner symbols for the powers
            wigner_m = float(Wigner9j(1, 1, m, 1, 1, m, 0, 0, 0).doit()) \
                 *float(CG(1,1,m,0,0,0).doit())
            # wigner symbols due to mixing of quantum numbers
            wigner_mixes = np.sqrt((2*j+1)*(2*l+1)*(2*s+1))*float(Wigner9j(j, j, 0, l, l, m, s, s, m).doit()) \
                 *(-1)**l*6*np.sqrt((2*m+1)/(2*l+1)*(2*s+1))*(2*m+1) \
                 *float(Wigner9j(s, s, m, 1/2, 1/2, 1, 1/2, 1/2, 1).doit())
            
            # add factor due to isospin here
            pow_factor = prefact * (2*t + 1) * (3*np.sqrt((2*m+1)/(4*math.pi))) * wigner_m * wigner_mixes

            # iterations over possible values of l3
            # set by the power of q
            for lam1 in range(m+1):
              lam2=m-lam1 # for conservation purposes
              # spherical harmonic factor for different lam states
              sph_harm_factor = np.sqrt(4*math.pi*math.factorial(2*m+1.)/(math.factorial(2*lam1)*math.factorial(2*lam2)))
              lam_factor=pow_factor*sph_harm_factor*pp**lam1*(-p)**lam2

              # allowed angular momentum states to contribute to the potential
              for k in range(max(abs(l-lam1),abs(l-lam2)),min(l+lam1,l+lam2)+1):
                  # wigner symbols due to coupling from k and lam1, lam2
                wigner_k = float(Wigner9j(k, k, 0., lam1, lam2, m, l, l, m).doit())*(2*k+1) *float(CG(k,lam1,l,0,0,0).doit())*\
                           float(CG(k,lam2,l,0,0,0).doit())
                angmom_factor=lam_factor * wigner_k

                # add this factor to the legendre polynomials as defined with spinless case
                mat+=angmom_factor*self._g(pp,p,k,m)

        # add the short-range potential part
        if (j==0): 
          mat+=self.C0*np.exp(-(pp**2+p**2)/self.cutoff**2) 
        elif (j==1):   
          mat+=self.C1*np.exp(-(pp**2+p**2)/self.cutoff**2)

        return mat

class OBEpot:
    """Provides a method for the partial wave representation of the OBE potential. 
    
       The matrix elements are obtained by numerical intergration.
       The mass of the exchanged boson, the strength of the 
       interaction and the couter term is given on initialization. 
       The interaction is regularized using a cutoff that is also 
       given on init.
    """
    
    # this are common parameters for all instances 
    hbarc=197.327
    
    # init interaction
    def __init__(self, cutoff=500.0, C0=1.0, nx=12,mpi=138.0,A=-1.0):
        """Defines the one boson exchange for a given regulator, coupling strength and short distance parameter
        
        Parameters:
        cutoff -- regulator in MeV
        C0 -- strength of the short distance counter term (in s-wave) 
        A -- strength of OBE
        nx -- number of angular grid points for numerical integration
        mpi -- mass of exchange boson in MeV"""
        
        self.mpi = mpi/self.hbarc
        self.cutoff = cutoff/self.hbarc
        self.C0=C0
        self.A=A
        self.nx=nx
        
        self.xp=np.empty((self.nx),dtype=np.double)
        self.xw=np.empty((self.nx),dtype=np.double)
        self.xp,self.xw=leggauss(self.nx)
    
    
    
    # function defines the x integral 
    def _g(self,pp,p,k):
        """Calculates g function of the partial wave decomposition of OBE. 
        
           pp -- outgoing momentum 
           p -- incoming momentum
           k -- angular momentum"""
        
        # define prefact 
        # get the corresponding legendre polynomial 
        Pk = legendre(k)
        # define momentum transfer dependent on angles 
        qval=np.sqrt(p**2+pp**2-2*p*pp*self.xp)
        
        # build integral of regularized OBE 
        return float(np.sum(Pk(self.xp)/((qval**2+self.mpi**2))*self.xw*np.exp(-(qval**2+self.mpi**2)/self.cutoff**2)))
        
    # determines complete, regularized interaction     
    def v(self,pp,p,l,s=0,j=0,t=0):
        """Potential matrix element in fm**2
        
           pp -- outgoing momentum in fm**-1
           p -- incoming momentum in fm**-1
           l -- angular momentum""" 
        
        # first overall prefact of 1pi exchange part  (cancel 2pi factors!)
        prefact=self.A
        
        mat=prefact*self._g(pp,p,l)

        if (l==0):   # add s-wave counter term 
          mat+=self.C0*np.exp(-(pp**2+p**2)/self.cutoff**2)  # 4pi is take into account by spherical harmonics for l=0
                    
        return mat