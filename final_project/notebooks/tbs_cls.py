'''
This file contains the classes used for our analysis.

In particular, we contain the classes TwoBody and OBEpot that do not 
directly interfere with the modifications we apply on our problem.

We also contain the TwoBodyTMat and ThreeBody class (for bosons) so that
we have easy access to them when performing comparisons.

'''

import numpy as np
import math as m
from numpy.polynomial.legendre import leggauss
from scipy.special import legendre, spherical_jn, sph_harm
from sympy.physics.quantum.cg import CG, Wigner9j, Wigner6j

from scipy.interpolate import interp1d
import timeit

# one-boson exchange potential, the potential we deal with with all of our problems.
# contains both long range and short ranged part
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
    def v(self,pp,p,l):
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


# the TwoBody class, directly copied from code for Ex6
class TwoBody:
    """Methods to obtain eigenvalues and eigenvectors for the bound state problem and for searches of the binding energy."""
    # define hbarc for unit conversion 
    hbarc=197.327  
    
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, mred=938.92/2,l=0,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            np1four=200,np2four=100):
        """Initialization of two-body solver. 
        
           The initialization defines the momentum grids and the interaction and partial wave to be used. 
           At this time, also the grid for Fourier transformation and for the Fourier transformed 
           wave function is given. 
           
           Parameters:
           pot -- object that defines the potential matrix elements (e.g. of class OBEpot).
           np1 -- number of grid points in interval [0,pb] 
           np2 -- number of grid points in interval [pb,pc]
           pa  -- half of np1 points are in interval [0,pa]
           pb  -- interval boundary as defined above 
           pc  -- upper integration boundary for the solution of the integral equation 
           mred -- reduces mass of the two bosons in MeV
           
           nr1 -- number of r points in interval [0,rb] 
           nr2 -- number of r points in interval [rb,rc]
           ra  -- half of np1 points are in interval [0,pa]
           rb  -- interval boundary as defined above 
           rc  -- upper integration boundary for the solution of the integral equation 
           
           np1four -- number of p points in interval [0,pb] for Fourier trafo
           np2four -- number of p points in interval [pb,pc] for Fourier trafo"""
        
        # measure also time for preparation (mostly from potential)
        self.preptime=-timeit.default_timer() 

        # store parameters (if necessary convert to fm)
        self.np1 = np1
        self.np2 = np2
        self.npoints  = np1+np2 
        self.mred=mred/self.hbarc
        self.pa=pa
        self.pb=pb
        self.pc=pc
        self.l=l 

        self.nr1 = nr1
        self.nr2 = nr2
        self.nrpoints  = nr1+nr2 
        self.ra=ra
        self.rb=rb
        self.rc=rc

        self.np1four = np1four
        self.np2four = np2four
        self.npfour  = np1four+np2four 

        # store grid points and weights for integral equations
        self.pgrid,self.pweight = self._trns(self.np1,self.np2,self.pa,self.pb,self.pc)
 
        # store grid points and weights for r space wave functions
        self.rgrid,self.rweight = self._trns(self.nr1,self.nr2,self.ra,self.rb,self.rc)
        
        # store grid points and weights for Fourier trafo 
        self.pfourgrid,self.pfourweight = self._trns(self.np1four,self.np2four,self.pa,self.pb,self.pc)
        
        # store underlying interaction
        self.pot=pot
        
        # and actual potential matrix elements 
        self.vmat=np.empty((self.npoints,self.npoints),dtype=np.double)
        for i in range(self.npoints):
          for j in range(self.npoints): 
            self.vmat[i,j]=self.pot.v(self.pgrid[i],self.pgrid[j],self.l)

        self.preptime+=timeit.default_timer() 
    
            
        # fix timer for solution of the eigenvalue equations
        self.runtime=0.0
        
    def _trns(self,np1,np2,pa,pb,pc):
      """Auxilliary method that provides transformed Gauss-Legendre grid points and integration weights.
      
         This is using a hyperbolic trafo shown in the lecture. 
         Parameter: 
         np1 --  grid points in ]0,pb[
         np2 --  grid points are distributed in ]pb,pc[ using a linear trafo
         
         pa  -- half of np1 points are in interval [0,pa]
         pb  -- interval boundary as defined above 
         pc  -- upper integration boundary """ 
    
      x1grid,x1weight=leggauss(np1)
      x2grid,x2weight=leggauss(np2)

      # trafo (1.+X) / (1./P1-(1./P1-2./P2)*X) for first interval 
      p1grid=(1.+x1grid) / (1./pa-(1./pa-2./pb)*x1grid)
      p1weight=(2.0/pa-2.0/pb)*x1weight / (1./pa-(1./pa-2./pb)*x1grid)**2

      # linear trafo 
      p2grid=(pc+pb)/2.0 + (pc-pb)/2.0*x2grid
      p2weight=(pc-pb)/2.0*x2weight
   
      pgrid=np.empty((self.npoints),dtype=np.double)
      pweight=np.empty((self.npoints),dtype=np.double)
    
      pgrid = np.concatenate((p1grid, p2grid), axis=None)
      pweight = np.concatenate((p1weight, p2weight), axis=None)
   
      return pgrid,pweight 

# set up set of equations and calculate eigenvalues 

    def eigv(self,E,neigv):
      """Solve two-body integral equation and return n-th eigenvalue, momentum grid and wave function. 

         Parameters:
         E -- energy used in the integral equation in fm**-1 
         neigv -- number of the eigenvalue to be used"""
   
    # measure timeing (compare for loop and einsum)
      self.runtime-=timeit.default_timer() 
    
    # set up the matrix amat for which eigenvalues have to be calculated
    
      amat=np.einsum('i,ij,j->ij', 1.0/(E-self.pgrid**2/(2*self.mred)),self.vmat,self.pweight*self.pgrid**2)   
      # replaces less performant for loops   
      #amat=np.empty((self.npoints,self.npoints),dtype=np.double)
      #for i in range(self.npoints):
      #  for j in range(self.npoints): 
      #    amat[i,j]=np.real(1.0/(E-self.pgrid[i]**2/(2*self.mred))*self.vmat[i,j]*self.pweight[j]*self.pgrid[j]**2)

    # determine eigenvalues using numpy's eig method        
      evalue,evec=np.linalg.eig(amat)
    
    # I now assume that the relevant eigenvalues are real to avoid complex arithmetic 
      evalue=np.real(evalue)
        
    # remove neigv-1 largest eigenvalues 
      for n in range(neigv-1):
        maxpos=np.argmax(evalue)
        evalue[maxpos]=0.0
    
    # take the next one 
      maxpos=np.argmax(evalue)
      eigv=evalue[maxpos]
    # define solution as unnormalized wave function 
      wf=evec[:,maxpos]
    # and normalize 
      norm=np.sum(wf**2*self.pweight[0:self.npoints]*self.pgrid[0:self.npoints]**2)
      wf=1/np.sqrt(norm)*wf
    
    # measure timeing (compare for loop and einsum)
      self.runtime+=timeit.default_timer()
    
      return eigv,self.pgrid[0:self.npoints],wf

    
    def esearch(self,neigv=1,e1=-0.01,e2=-0.0105,elow=0.0,tol=1e-8):
        """Perform search for energy using the secant method. 
        
           Parameters:
           neigv -- number of the eigenvalue to be used
           e1 -- first estimate of binding energy (should be negative)
           e2 -- second estimate of binding energy (should be negative)
           elow -- largest energy to be used in search (should be negative)
           tol -- if two consecutive energies differ by less then tol, the search is converged
           
           Energies are given in fm**-1. """
        
        # determine eigenvalues for starting energies        
        eta1,pgrid,wf=self.eigv(e1,neigv)
        eta2,pgrid,wf=self.eigv(e2,neigv)
        
        while abs(e1-e2) > tol: 
          # get new estimate (taking upper value into account)   
          enew=e2+(e1-e2)/(eta1-eta2)*(1-eta2) 
          enew=min(elow,enew)
       
          # get new eigenvalue and replace e1 and e2 for next iteration
          eta,pgrid,wf=self.eigv(enew,neigv)
          e2=e1
          eta2=eta1
          e1=enew
          eta1=eta 
                  
        return e1,eta1,pgrid,wf 
           
    def fourier(self,wfp):
        """Calculates the Fourier transform of the partial wave representation of the wave function.
        
           Parameter: 
           wfp -- wave function in momentum space
            
           Note that the factor I**l is omitted."""
        
        # calculate spherical bessel functions based dense Fourier trafo momentum grid and rgrid
        # prepare matrix based on r,p points  
        rpmat = np.outer(self.rgrid,self.pfourgrid)
        # evaluate jl     
        jlmat = spherical_jn(self.l,rpmat)
        
        # interpolate of wave to denser Fourier trafo grid
        wfinter = interp1d(self.pgrid, wfp, kind='cubic',fill_value="extrapolate")
        # interpolate wf and multiply my p**2*w elementwise 
        wfdense = wfinter(self.pfourgrid)*self.pfourgrid**2*self.pfourweight*np.sqrt(2/m.pi)
        
        # now the Fourier trafo is a matrix-vector multiplication 
        wfr = jlmat.dot(wfdense)
        
        return self.rgrid,wfr
    
    
    def rms(self,wfr):
        """Calculates the norm and rms radius for the given r-space wave function.
        
           Normalization of the wave function is assumed. 
           Parameter: 
           wfr -- wave function in r-space obtained by previous Fourier trafo"""
        
        
        norm=np.sum(wfr**2*self.rweight*self.rgrid**2)
        rms=np.sum(wfr**2*self.rweight*self.rgrid**4)

            
        rms=np.sqrt(rms)
        
        return norm,rms

# next extend the class for twobody to scattering 
from scipy.special import legendre
import timeit 

class TwoBodyTMat(TwoBody):
    """This class defines the off-shell t-matrix for a three-body bound state.
    
       The class assumes three identical particles. It also contains the 
       initialization of grid points for the three-body problem. 
    """
    
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, 
                            nq1=20, nq2=10, qa=1.0, qb=5.0, qc=20.0, 
                            mass=938.92,lmax=0,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            nrho1=20, nrho2=10, rhoa=1.0, rhob=5.0, rhoc=20.0, 
                            np1four=200,np2four=100):
        """Initialization of grid points and interaction for the solution of the three-body problem. 
        
           Parameter: 
           pot -- object that defines the potential matrix elements (e.g. of class OBEpot).
           np1 -- number of p grid points in interval [0,pb] 
           np2 -- number of p grid points in interval [pb,pc]
           pa  -- half of np1 points are in interval [0,pa]
           pb  -- interval boundary as defined above 
           pc  -- upper integration boundary for the solution of the integral equation 
           nq1 -- number of q grid points in interval [0,qb] 
           nq2 -- number of q grid points in interval [qb,qc]
           qa  -- half of np1 points are in interval [0,qa]
           qb  -- interval boundary as defined above 
           qc  -- upper integration boundary for the solution of the integral equation 
           mass -- particle mass of the three identical bosons in MeV
           
           nr1 -- number of r (related to p) points in interval [0,rb] 
           nr2 -- number of r (related to p) points in interval [rb,rc]
           ra  -- half of np1 points are in interval [0,ra]
           rb  -- interval boundary as defined above 
           rc  -- upper integration boundary for the solution of the integral equation 
           nrho1 -- number of rho (related to q) points in interval [0,rhob] 
           nrho2 -- number of rho (related to q) points in interval [rhob,rhoc]
           rhoa  -- half of np1 points are in interval [0,rhoa]
           rhob  -- interval boundary as defined above 
           rhoc  -- upper integration boundary for the solution of the integral equation 
           
           np1four -- number of p or q  points in interval [0,pb] or[0,qb]   for Fourier trafo
           np2four -- number of p or q points in interval [pb,pc] or [qb,qc] for Fourier trafo
           
           lmax  -- maximal two-body angular momentum to be taken into account
           """
        
        
        # first use the TwoBody class to keep the main parameters 
        super().__init__(pot,np1,np2,pa,pb,pc,mass/2,0,nr1,nr2,ra,rb,rc,np1four,np2four)

        self.nq1 = nq1
        self.nq2 = nq2
        self.nqpoints  = nq1+nq2 
        self.mass=mass/self.hbarc
        self.qa=qa
        self.qb=qb
        self.qc=qc
        self.lmax=lmax 

        self.nrho1 = nrho1
        self.nrho2 = nrho2
        self.nrhopoints  = nrho1+nrho2 
        self.rhoa=rhoa
        self.rhob=rhob
        self.rhoc=rhoc


        # store grid points and weights for integral equations
        self.qgrid,self.qweight = self._trns(self.nq1,self.nq2,self.qa,self.qb,self.qc)
 
        # store grid points and weights for r space wave functions
        self.rhogrid,self.rhoweight = self._trns(self.nrho1,self.nrho2,self.rhoa,self.rhob,self.rhoc)
        
        # store grid points and weights for Fourier trafo 
        self.qfourgrid,self.qfourweight = self._trns(self.np1four,self.np2four,self.qa,self.qb,self.qc)
        
        # and prepare actual potential matrix elements for all angular momenta
        self.vmatpw=np.empty((self.npoints,self.npoints,self.lmax+1),dtype=np.double)
        for l in range(self.lmax+1):
         for i in range(self.npoints):
          for j in range(self.npoints):
            if l==0:
              self.vmatpw[i,j,l]=self.vmat[i,j]  
            else:    
              self.vmatpw[i,j,l]=self.pot.v(self.pgrid[i],self.pgrid[j],l)
        
        self.tmattime=0.0
        
# now turn to scattering and solve for LS equation to get tmatrix (on- and offshell)
    def prep_tmat(self,E):
      """Prepares all necessary t-matrix elements up to l=lmax.
      
         Starts the calculation of the t-matrix for a given three-body energy. 
      """  
      self.tmattime-=timeit.default_timer() 
      # prepare off-shell energies for t-matrix 
      etmat=E-0.75*self.qgrid**2/self.mass   # note that this is a negative energy < E_b(two-body) 
             
      # prepare numpy array that keeps all tmatrix elements 
      tmat=np.empty((self.lmax+1,self.nqpoints,self.npoints,self.npoints),dtype=np.double)
      
      # now I need to solve the Lippmann-Schwinger equation for each etmat and each l =0,..,lmax
      for l in range(self.lmax+1):
        for ie in range(self.nqpoints): 
            
          # define matrix for set of equations 
          # predefine the Kronecker deltas 
          amat=np.identity(self.npoints,dtype=np.double)
          # now add the second part of the definition of the matrix   
          #for i in range(self.npoints):
          #  for j in range(self.npoints):
          #     amat[i,j]+=-self.vmatpw[i,j,l]*self.pgrid[j]**2 \
          #                     /(etmat[ie]-self.pgrid[j]**2/(2*self.mred))*self.pweight[j]  \  

          amat-=np.einsum("ij,j->ij",self.vmatpw[:,:,l],self.pgrid[:]**2 \
                               /(etmat[ie]-self.pgrid[:]**2/(2*self.mred))*self.pweight[:])
            
              
        
          # now define the rhs   
#          bmat=np.empty((self.npoints,self.npoints),dtype=np.double)
#          for i in range(self.npoints):
#           for j in range(self.npoints):   
#             bmat[i,j]=self.vmatpw[i,j,l]

          bmat=self.vmatpw[:,:,l]
    
          # finally solve set of equations and store in complete array 
          tmat[l,ie,:,:]=np.linalg.solve(amat,bmat)
        
      self.tmattime+=timeit.default_timer() 
      # return offshell matrix for all energies and angular momenta   
      return tmat
            


# prepare interpolation using cubic hermitian splines 

class Cubherm:
    """Prepares spline functions for cubic hermitian splines. 
    
    see Hueber et al. FBS 22,107 (1997). 
    
    The function spl returns the the spline function for a given x. 
    If x is below the smallest grid point, extrapolation is used. 
    If x is after largest grid point, then the function evaluates to zero. 
    """
    
        
    def spl(xold,xin):
        """Calculates spline functions for given values xold and xnew.
        
           Parameters:
           xold -- set of grid points where function is given. xold needs to be one dimensional.
           xnew -- set of grid points to interpolate to. xnew can be multidimensional. 
           
           On return spline functions will be given that have the shape of xnew and xold combined. 
        """
        
        # first determine the base value of the index for each xnew.
        
        nold=len(xold)
        if nold<4:
          raise(ValueError("Interpolation requires at least 4 grid points.")) 
        
        xnew=xin.reshape((-1))        
        indx=np.empty((len(xnew)),dtype=int)
        
        for i in range(len(xnew)):
          # do not extrapolated beyond largest grid point
          if xnew[i] > xold[nold-1]: 
            indx[i]=-1
          else:  
            for j in range(nold):
              if xnew[i] <= xold[j]:
                break          
            if j < 1:
              indx[i]=0
            elif j > nold-3:
              indx[i]=nold-3
            else:
              indx[i]=j-1  

        # then prepare phi polynomials for each x 
        
        phi1=np.zeros((len(xnew)),dtype=np.double)
        phi2=np.zeros((len(xnew)),dtype=np.double)
        phi3=np.zeros((len(xnew)),dtype=np.double)
        phi4=np.zeros((len(xnew)),dtype=np.double)
        
        for i in range(len(xnew)):
          if indx[i]>0:  
            phi1[i] = (xold[indx[i] + 1] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 3 * (xold[indx[i] + 1] - 3 * xold[indx[i]] + 2 * xnew[i])
            phi2[i] = (xold[indx[i]] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 3 * (3 * xold[indx[i] + 1] - xold[indx[i]] - 2 * xnew[i])
            phi3[i] = (xnew[i] - xold[indx[i]]) * (xold[indx[i] + 1] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 2
            phi4[i] = (xnew[i] - xold[indx[i] + 1]) * (xold[indx[i]] - xnew[i]) ** 2 / (xold[indx[i] + 1] - xold[indx[i]]) ** 2
        
        # now we are ready to prepare the spline functions 
        # most are zero 
        splfu=np.zeros((len(xold),len(xnew)),dtype=np.double)
        for i in range(len(xnew)):
          if indx[i]>0:  
            splfu[indx[i]-1,i] = \
               -phi3[i]*(xold[indx[i]+1]-xold[indx[i]])/(
                        (xold[indx[i]]-xold[indx[i]-1])*(xold[indx[i]+1]-xold[indx[i]-1]))
            
            splfu[indx[i],i] = phi1[i] \
                +phi3[i]*((xold[indx[i]+1]-xold[indx[i]])/ (xold[indx[i]]-xold[indx[i]-1]) \
                         -(xold[indx[i]]-xold[indx[i]-1])/ (xold[indx[i]+1]-xold[indx[i]]))/(xold[indx[i]+1]-xold[indx[i]-1]) \
                -phi4[i]*(xold[indx[i]+2]-xold[indx[i]+1])/ (xold[indx[i]+1]-xold[indx[i]])/(xold[indx[i]+2]-xold[indx[i]])

            splfu[indx[i]+1,i] = phi2[i] \
                +phi3[i]*(xold[indx[i]]-xold[indx[i]-1])/ (xold[indx[i]+1]-xold[indx[i]])/(xold[indx[i]+1]-xold[indx[i]-1]) \
                +phi4[i]*((xold[indx[i]+2]-xold[indx[i]+1])/ (xold[indx[i]+1]-xold[indx[i]]) \
                         -(xold[indx[i]+1]-xold[indx[i]])/ (xold[indx[i]+2]-xold[indx[i]+1]))/(xold[indx[i]+2]-xold[indx[i]])
            
            
            splfu[indx[i]+2,i] = \
                phi4[i]*(xold[indx[i]+1]-xold[indx[i]])/(
                        (xold[indx[i]+2]-xold[indx[i]+1])*(xold[indx[i]+2]-xold[indx[i]]))
          elif indx[i]>=0:
            # do linear interpolation at the origin 
            splfu[indx[i],i] = (xnew[i]-xold[indx[i]+1])/(xold[indx[i]]-xold[indx[i]+1]) 
            splfu[indx[i]+1,i] = (xold[indx[i]]-xnew[i])/(xold[indx[i]]-xold[indx[i]+1]) 

        retshape=[nold] 
        for n in list(np.shape(xin)):
          retshape.append(n)
        
        return splfu.reshape(retshape)
    

# three-body bound state for bosons
# we still need to add wavefunction evaluation and KE evaluation in here

# definition of a ThreeBody class for the calculation of bound states



class ThreeBody(TwoBodyTMat):
    """Provides routines for the implementation of the permutation operator and application of the bound state kernel."""
        
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, 
                            nq1=20, nq2=10, qa=1.0, qb=5.0, qc=20.0, 
                            nx=12,
                            mass=938.92,lmax=0,bl=0,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            nrho1=20, nrho2=10, rhoa=1.0, rhob=5.0, rhoc=20.0, 
                            np1four=200,np2four=100,maxiter=10,eigtol=1.0E-6):     
        """Initializes the permutation operator for the three-body calculation and prepares application of Faddeev kernel.
        
           Parameters: 
           
           pot -- object that defines the potential matrix elements (e.g. of class OBEpot).
           np1 -- number of p grid points in interval [0,pb] 
           np2 -- number of p grid points in interval [pb,pc]
           pa  -- half of np1 points are in interval [0,pa]
           pb  -- interval boundary as defined above 
           pc  -- upper integration boundary for the solution of the integral equation 
           nq1 -- number of q grid points in interval [0,qb] 
           nq2 -- number of q grid points in interval [qb,qc]
           qa  -- half of np1 points are in interval [0,qa]
           qb  -- interval boundary as defined above 
           qc  -- upper integration boundary for the solution of the integral equation 
           
           nx -- angular grid points for the permutation operator
           
           mass -- particle mass of the three identical bosons in MeV
           
           nr1 -- number of r (related to p) points in interval [0,rb] 
           nr2 -- number of r (related to p) points in interval [rb,rc]
           ra  -- half of np1 points are in interval [0,ra]
           rb  -- interval boundary as defined above 
           rc  -- upper integration boundary for the solution of the integral equation 
           nrho1 -- number of rho (related to q) points in interval [0,rhob] 
           nrho2 -- number of rho (related to q) points in interval [rhob,rhoc]
           rhoa  -- half of np1 points are in interval [0,rhoa]
           rhob  -- interval boundary as defined above 
           rhoc  -- upper integration boundary for the solution of the integral equation 
           
           np1four -- number of p or q  points in interval [0,pb] or[0,qb]   for Fourier trafo
           np2four -- number of p or q points in interval [pb,pc] or [qb,qc] for Fourier trafo
           
           lmax  -- maximal two-body angular momentum to be taken into account    
           bl    -- total orbital angular momentum L ("big l")           
        """
        
        # first initialize the tmatrix class (do not calc the tmatrix yet)
        super().__init__(pot,np1,np2,pa,pb,pc,nq1,nq2,qa,qb,qc,
                         mass,lmax,
                         nr1,nr2,ra,rb,rc,nrho1,nrho2,rhoa,rhob,rhoc,
                         np1four,np2four)
        
        # prepare angular grid points for the permutation operator 
        self.nx=nx
        self.xp,self.xw = leggauss(nx)
        
        # store parameter for iterative solver 
        self.maxiter=maxiter 
        self.eigtol=eigtol 
        
        # prepare partial wave channels for total angular momentum bl
        # the preparation of a table of allowed combinations is useful 
        # for L != 0 (then l=l12 and lam=l3 can be different from each other)
        self.bl=bl
        self.lammax=lmax+bl
        self.qnalpha=[]
        alpha=0
        for l in range(lmax+1):
          if(l%2==0):   # take only symmetric pw (Pauli)  
           for lam in range(abs(l-bl),l+bl+1):
             self.qnalpha.append({"alpha":alpha,"l":l,"lam":lam,"bl":bl})
             alpha+=1
        self.nalpha=len(self.qnalpha)
        
        self.gpreptime=-timeit.default_timer()   
        
        # split time measurements in several pieces to find relevant loops 
        self.timepermangle=0
        self.timeylam=0
        self.timeyl=0
        self.timeystarl=0
        self.timeclebsch=0
        self.timeylylam=0
        self.timegcalc=0
        self.timespl=0
        self.timepmat=0
        
        # this prepares the G function and splines to be used for the preparation of the 
        # kernel later (pmat = permutation matrix)
        self.pmat=self._prep_perm()
        self.gpreptime+=timeit.default_timer()

        self.fadpreptime=0
        self.fadsolvetime=0
        self.numiter=0
        
        
        
    def _angle(self,px,py,pz):
        """Auxiliary routine to determine magnitude, phi, and theta of three component vector. 
        
           Parameters:
           px,py,pz -- cartesian components of a vector 
           
           returns magntitude, theta and phi angles.
        """
    
        pmag=np.sqrt(px**2+py**2+pz**2)
        theta=np.where(pmag!=0.0,np.arccos(pz/pmag),0.0)
             
        phi=theta # copy shape of theta to phi 
        phi=1.5*m.pi # and set to constant value
        
        # prepare bool arrays for px,py > 0 < 0  with shape of phi 
        
        pxgt0=(phi==0)  # copy shape
        pxgt0=(px>0)    # test 

        pxlt0=(phi==0)  # copy shape
        pxlt0=(px<0)    # test 

        pxeq0=(phi==0)  # copy shape
        pxeq0=(px==0)   # test 

        pygt0=(phi==0)  # copy shape
        pygt0=(py>0)    # test 
                      
        np.where(pxgt0 & pygt0,np.arctan(py/px),phi)
        np.where(pxgt0 & np.invert(pxgt0),2*m.pi-np.arctan(-py/px),phi)
        np.where(pxlt0 & pygt0,m.pi-np.arctan(-py/px),phi)
        np.where(pxlt0 & np.invert(pygt0),m.pi+np.arctan(py/px),phi)
        np.where(pxeq0 & pygt0,0.5*m.pi,phi)
            
        return pmag,theta,phi     
    
    
    def _lmindx(self,l,m):
        """Combined unique index for l and m.
        
           Nice trick: since quantum numbers lm are linked to each other, this combined 
           index allows one to store the results depending on lm using the memory more efficiently. 
        """        
        return l**2+l+m
      
        
    def _prep_perm(self):
        """Prepares and return an array for the application of the permutation operator.
        
           The matrix is based on G_{alpha,alphap}(q,qp,x) and is combined to be 
           directly applicable to be summed  with tmatrix.
        """
        
        
        
        self.timepermangle-=timeit.default_timer()
        # prepare shifted momenta and angles for the symmetric permutation 
        pip=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)        
        pi=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)        
        
        thetap=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        theta=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        thetapp=np.empty((self.nx),dtype=np.double)
        
        for ix in range(self.nx):
          xval=self.xp[ix] 
          thetapp[ix]=np.arccos(xval)
          for jq in range(self.nqpoints):
            qpval=self.qgrid[jq]
            for iq in range(self.nqpoints):
              qval=self.qgrid[iq]
            
              px=qpval*np.sqrt(1.0-xval**2)
              py=0.0
              pz=0.5*qval+qpval*xval 
              pi[iq,jq,ix],theta[iq,jq,ix],phi=self._angle(px,py,pz)
                
              px=-0.5*qpval*np.sqrt(1.0-xval**2)
              py=0.0
              pz=-qval-0.5*qpval*xval 
              pip[iq,jq,ix],thetap[iq,jq,ix],phi=self._angle(px,py,pz)

        self.timepermangle+=timeit.default_timer()
        
        # prepare spherical harmonics and store based on lmindx 
        # number of lam,mu und l,mu combinations 
        nlamindx=self._lmindx(self.lammax,self.lammax)+1
        nlindx=self._lmindx(self.lmax,self.lmax)+1
        
        self.timeylam-=timeit.default_timer()
        # array for Y_{lam mu}(hat qp) (real is sufficient since phi=0)
        ylam=np.empty((nlamindx,self.nx),dtype=np.double)
        for lam in range(self.lammax+1):
          for mu in range(-lam,lam+1):
            ylam[self._lmindx(lam,mu),:]=np.real(sph_harm(mu,lam, 0, thetapp))
        self.timeylam+=timeit.default_timer()
        
        
        self.timeyl-=timeit.default_timer()
        # array for Y_{l mu}(-q-0.5qp) (real is sufficient since phi=0)
        yl=np.empty((nlindx,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for l in range(self.lmax+1):
          for mu in range(-l,l+1):
            yl[self._lmindx(l,mu),:,:,:]=np.real(sph_harm(mu,l, 0, thetap))
        self.timeyl+=timeit.default_timer()
        
        self.timeystarl-=timeit.default_timer()
        # array for Y*_{l mu}(0.5q+qp) (real is sufficient since phi=0)
        ystarl=np.empty((nlindx,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for l in range(self.lmax+1):
          for mu in range(-l,l+1):
            ystarl[self._lmindx(l,mu),:,:,:]=np.real(sph_harm(mu,l, 0, theta))
        self.timeystarl+=timeit.default_timer()
        
        # now prepare the necessary Clebsch-Gordan coefficients
        # we need (l lam L, M 0 M)  and (l lam L,mu M-mu,M)
        # I assume that L is smaller than the lmax or lammax therefore M=-L,L
        # the smallest index for storage 
        self.timeclebsch-=timeit.default_timer()
        
        cg=np.zeros((self.nalpha,2*self.bl+1),dtype=np.double)
        cgp=np.zeros((self.nalpha,2*self.lmax+1,2*self.bl+1),dtype=np.double)
        
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          for bm in range(-self.bl,self.bl+1):
            cg[qnset["alpha"],bm+self.bl]=float(CG(qnset["l"],bm,qnset["lam"],0,self.bl,bm).doit())
            for mu in range(-qnset["l"],qnset["l"]+1):
              cgp[qnset["alpha"],mu+qnset["l"],bm+self.bl]=float(CG(qnset["l"],mu,qnset["lam"],bm-mu,self.bl,bm).doit())
        self.timeclebsch+=timeit.default_timer()

        self.timeylylam-=timeit.default_timer()
        # now we can perform the mu summation for the combination of coupled spherical harmonics 
        ylylam=np.zeros((self.nalpha,2*self.bl+1,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alphap=qnset["alpha"]
          l=qnset["l"]
          lam=qnset["lam"]
          for bm in range(-self.bl,self.bl+1):
            for mu in range(-l,l+1):
              lmindx=self._lmindx(l,mu)
              if abs(bm-mu)<=lam:
                lamindx=self._lmindx(lam,bm-mu)
                ylylam[alphap,bm+self.bl,:,:,:]+=cgp[alphap,mu+l,bm+self.bl]*yl[lmindx,:,:,:]*ylam[lamindx,:]
        self.timeylylam+=timeit.default_timer()
                
        # bm summation then gives G 
        self.timegcalc-=timeit.default_timer()
        gfunc=np.zeros((self.nalpha,self.nalpha,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alpha=qnset["alpha"]
          l=qnset["l"]
          lam=qnset["lam"]
          for qnsetp in self.qnalpha:  # go through allowed l,lam combinations
            alphap=qnsetp["alpha"]
            for bm in range(-self.bl,self.bl+1):
              if(abs(bm)<=l):  
                lmindx=self._lmindx(l,bm) 
                gfunc[alpha,alphap,:,:,:]+=8*m.pi**2*np.sqrt((2*lam+1)/(4*m.pi))/(2*self.bl+1) \
                   *ystarl[lmindx,:,:,:]*ylylam[alphap,bm+self.bl,:,:,:]
        self.timegcalc+=timeit.default_timer()
            
        #  now we assume that there is a function on p on the left defined by p**l and on the right devided by p'**l' 
        # that is interpolated using Cubherm to pi and pip 
        
        # set spline elements based on grid points and shifted momenta 
        self.timespl-=timeit.default_timer()
        splpi=Cubherm.spl(self.pgrid,pi)
        splpip=Cubherm.spl(self.pgrid,pip)
        
        # interpolation fspl=np.sum(spl*fold,axis=1) first axis is pgrid 
        # prepare splines multiplied by p**l factors (splalpha also includes the integration weights for q' and x integral)
        
        splalpha=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
        splalphap=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
        
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alpha=qnset["alpha"]
          l=qnset["l"]
            
          # ijkl : iq ip jq ix  
          splalpha[alpha,:,:,:,:]=np.einsum("jikl,ikl,j,l,k->ijkl",splpi,pi**l,1.0/self.pgrid**l,self.xw,self.qweight*self.qgrid**2)
          splalphap[alpha,:,:,:,:]=np.einsum("jkil,kil,j->ijkl",splpip,pip**l,1.0/self.pgrid**l)
        
#          for ip in range(self.npoints): 
#           for iq in range(self.nqpoints):
#             indxpmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
             #for jq in range(self.nqpoints):
             #   splalpha[indxpmat,jq,:]=splpi[ip,iq,jq,:]*(pi[iq,jq,:]/self.pgrid[ip])**l*self.xw[:]*self.qweight[jq]*self.qgrid[jq]**2
             #   splalphap[indxpmat,jq,:]=splpip[ip,jq,iq,:]*(pip[jq,iq,:]/self.pgrid[ip])**l
#             splalpha[indxpmat,:,:]=np.einsum("ij,j,i->ij",splpi[ip,iq,:,:]*(pi[iq,:,:]/self.pgrid[ip])**l,self.xw[:],self.qweight[:]*self.qgrid[:]**2)
#             splalphap[indxpmat,:,:]=splpip[ip,:,iq,:]*(pip[:,iq,:]/self.pgrid[ip])**l
        self.timespl+=timeit.default_timer()
            
        
        self.timepmat-=timeit.default_timer()
        #pmat=np.zeros((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha),dtype=np.double)
        
        # also generate views with separated indices 
        #pmatsingle=pmat.reshape((self.nalpha,self.nqpoints,self.npoints,self.nalpha,self.nqpoints,self.npoints))
        
        #splalphapsingle=splalphap.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
        #splalphasingle=splalpha.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
        
        # ijk : alpha iq ip (indxpmat)
        # lmn : alphap jq jp (indxpmatp)
        # o   : ix 
        
        pmatsingle=np.einsum("ijkmo,iljmo,lmnjo->ijklmn",splalpha,gfunc,splalphap)
        pmat=pmatsingle.reshape((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha))
        
#        for qnset in self.qnalpha:  # go through allowed l,lam combinations
#          alpha=qnset["alpha"]
#          for qnsetp in self.qnalpha:  # go through allowed l,lam combinations
#            alphap=qnsetp["alpha"]
#            for ip in range(self.npoints): 
#             for iq in range(self.nqpoints):
#              indxpmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
#              for jp in range(self.npoints): 
#               for jq in range(self.nqpoints):
#                indxpmatp=self.npoints*self.nqpoints*alphap+self.npoints*jq+jp
#                pmat[indxpmat,indxpmatp]=np.sum(splalpha[indxpmat,jq,:]
#                              *gfunc[alpha,alphap,iq,jq,:]
#                              *splalphap[indxpmatp,iq,:])                   
        self.timepmat+=timeit.default_timer()
                                      
        
        return pmat
        
    def prep_faddeev(self,ener):
        """Prepares the Faddeev kernel as a matrix using only two-body interactions.
        
           Parameter:
           ener -- three-body energy in fm-1
        """
 
        # get tmatrix for given energy
        self.tmat=self.prep_tmat(ener)
        
        self.fadpreptime-=timeit.default_timer()
        
        # use matrix multiplication of preprepared permutation matrix 
        # self.pmat[indxpmat,indxpmatp] contains permutation matrix 
        # indexpmat is alpha,iq,ip 

        self.kfadmat=np.zeros(self.pmat.shape,dtype=np.double)
        
        for qnset in self.qnalpha:
          l=qnset["l"] 
          alpha=qnset["alpha"]
          for iq in range(self.nqpoints):
            for ip in range(self.npoints):
              indxkmat=ip+self.npoints*iq+self.npoints*self.nqpoints*alpha
              for jp in range(self.npoints):
                indxpmat=jp+self.npoints*iq+self.npoints*self.nqpoints*alpha
                self.kfadmat[indxkmat,:]+=self.tmat[l,iq,ip,jp]*2*self.pmat[indxpmat,:]
    
        # now multiply with G0
        
        G0=np.empty((self.nqpoints,self.npoints),dtype=np.double)
        for iq in range(self.nqpoints):
          for ip in range(self.npoints):
            G0[iq,ip]=1.0/(ener-0.75*self.qgrid[iq]**2/self.mass-self.pgrid[ip]**2/self.mass )
        
        for alpha in range(self.nalpha):
          for iq in range(self.nqpoints):
            for ip in range(self.npoints):
              indxkmat=ip+self.npoints*iq+self.npoints*self.nqpoints*alpha
              self.kfadmat[indxkmat,:]*=G0[iq,ip]  

        self.fadpreptime+=timeit.default_timer()

# set up set of equations and calculate eigenvalues 

    def eigv(self,E,neigv):
      """Solve three-body Faddev equation and return n-th eigenvalue and Faddeev component. 

         Parameters:
         E -- energy used in the integral equation in fm**-1 
         neigv -- number of the eigenvalue to be used"""
   
    # set up the matrix for the Faddeev equations
      self.prep_faddeev(E)
      self.fadsolvetime-=timeit.default_timer()
        
    # determine eigenvalues using numpy's eig method        
      evalue,evec=np.linalg.eig(self.kfadmat)
    
    # I now assume that the relevant eigenvalues are real to avoid complex arithmetic 
      evalue=np.real(evalue)
        
    # remove neigv-1 largest eigenvalues 
      for n in range(neigv-1):
        maxpos=np.argmax(evalue)
        evalue[maxpos]=0.0
    
    # take the next one 
      maxpos=np.argmax(evalue)
      eigv=evalue[maxpos]
    
    # define solution as unnormalized Faddeev component 
      fadcomp=np.real(evec[:,maxpos])
          
    # and normalize using permutation again 
      fadtmp=2.0*self.pmat.dot(fadcomp)
        
      norm=0.0  
      for alpha in range(self.nalpha):
        for iq in range(self.nqpoints):
          for ip in range(self.npoints):
            indxkmat=ip+self.npoints*iq+self.npoints*self.nqpoints*alpha
            norm+=fadcomp[indxkmat]*fadtmp[indxkmat]*self.qweight[iq]*self.qgrid[iq]**2
            
      fadcomp=fadcomp.reshape((self.nalpha,self.nqpoints,self.npoints))     
      norm+=self.skalp(fadcomp,fadcomp)  
      norm*=3.0
        
      fadcomp=(1/np.sqrt(norm))*fadcomp
    
      self.fadsolvetime+=timeit.default_timer()
      return eigv,fadcomp
                
    def esearch(self,neigv=1,e1=-0.05,e2=-0.06,elow=-0.02,tol=1e-8,nitermax=20):
        """Perform search for energy using the secant method. 
        
           Parameters:
           neigv -- number of the eigenvalue to be used
           e1 -- first estimate of binding energy (should be negative)
           e2 -- second estimate of binding energy (should be negative)
           elow -- largest energy to be used in search (should be negative)
           tol -- if two consecutive energies differ by less then tol, the search is converged
           
           Energies are given in fm**-1. """
        
        # determine eigenvalues for starting energies        
        eta1,fadcomp=self.eigv(e1,neigv)
        eta2,fadcomp=self.eigv(e2,neigv)
        niter=0
        
        # start = timeit.default_timer()
        while abs(e1-e2/e1) > tol: 
          # get new estimate (taking upper value into account)   
          enew=e2+(e1-e2)/(eta1-eta2)*(1-eta2) 
          enew=min(elow,enew)
       
          # get new eigenvalue and replace e1 and e2 for next iteration
          eta,fadcomp=self.eigv(enew,neigv)
          e2=e1
          eta2=eta1
          e1=enew
          eta1=eta 

          # print(e1, e2, eta)

          # break if loop is taking too long
          niter+=1
          if niter > nitermax:
            break

        # end=timeit.default_timer()

        # print("time for energy search: ", end-start)
            
        return e1,eta1,fadcomp 
  
      # define the wavefunction
    def wavefunc(self, fadcomp):
      '''
      Wavefunction, evaluated from permutation operator and Faddeev component
      '''

      # first evaluate the identity part of the wave function evaluation
      # this means to evaluate the spline integrals without the permutation operator

      # prepare shifted momenta and angles for the symmetric permutation 
      pip=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)        
      pi=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)        
      
      thetap=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
      theta=np.empty((self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
      thetapp=np.empty((self.nx),dtype=np.double)
      
      for ix in range(self.nx):
        xval=self.xp[ix] 
        thetapp[ix]=np.arccos(xval)
        for jq in range(self.nqpoints):
          qpval=self.qgrid[jq]
          for iq in range(self.nqpoints):
            qval=self.qgrid[iq]
          
            px=qpval*np.sqrt(1.0-xval**2)
            py=0.0
            pz=0.5*qval+qpval*xval 
            pi[iq,jq,ix],theta[iq,jq,ix],phi=self._angle(px,py,pz)
              
            px=-0.5*qpval*np.sqrt(1.0-xval**2)
            py=0.0
            pz=-qval-0.5*qpval*xval 
            pip[iq,jq,ix],thetap[iq,jq,ix],phi=self._angle(px,py,pz)

      splpi=Cubherm.spl(self.pgrid,pi)
      splpip=Cubherm.spl(self.pgrid,pip)
      
      # interpolation fspl=np.sum(spl*fold,axis=1) first axis is pgrid 
      # prepare splines multiplied by p**l factors (splalpha also includes the integration weights for q' and x integral)
      
      splalpha=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
      splalphap=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
      
      for qnset in self.qnalpha:  # go through allowed l,lam combinations
        alpha=qnset["alpha"]
        l=qnset["l"]
          
        # ijkl : iq ip jq ix  
        # we dont integrate over qj so we remove qweights from splalpha
        splalpha[alpha,:,:,:,:]=np.einsum("jikl,ikl,j,l->ijkl",splpi,pi**l,1.0/self.pgrid**l,self.xw)
        splalphap[alpha,:,:,:,:]=np.einsum("jkil,kil,j->ijkl",splpip,pip**l,1.0/self.pgrid**l)
      
        # for ip in range(self.npoints): 
        #   for iq in range(self.nqpoints):
        #     indxpmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
        #     for jq in range(self.nqpoints):
        #       # splalpha[indxpmat,jq,:]=splpi[ip,iq,jq,:]*(pi[iq,jq,:]/self.pgrid[ip])**l*self.xw[:]*self.qweight[jq]*self.qgrid[jq]**2
        #       # splalphap[indxpmat,jq,:]=splpip[ip,jq,iq,:]*(pip[jq,iq,:]/self.pgrid[ip])**l
        #     splalpha[indxpmat,:,:]=np.einsum("ij,j->ij",splpi[ip,iq,:,:]*(pi[iq,:,:]/self.pgrid[ip])**l,self.xw[:])
        #     splalphap[indxpmat,:,:]=splpip[ip,:,iq,:]*(pip[:,iq,:]/self.pgrid[ip])**l

      # idmat = np.zeros((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha), dtype=np.double)

      # constructing the identity matrix element between the spline functions
      idmatsingle=np.einsum("ijkmo,lmnjo->ijklmn",splalpha,splalphap)
      idmat=idmatsingle.reshape((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha))
      
      # also generate views with separated indices 
      #pmatsingle=pmat.reshape((self.nalpha,self.nqpoints,self.npoints,self.nalpha,self.nqpoints,self.npoints))
      
      #splalphapsingle=splalphap.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
      #splalphasingle=splalpha.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
      
      # ijk : alpha iq ip (indxpmat)
      # lmn : alphap jq jp (indxpmatp)
      # o   : ix 
      
      # pmatsingle=np.einsum("ijkmo,iljmo,lmnjo->ijklmn",splalpha,gfunc,splalphap)
      # pmat=pmatsingle.reshape((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha))

      # for qnset in self.qnalpha:  # go through allowed l,lam combinations
      #   alpha=qnset["alpha"]
      #   for qnsetp in self.qnalpha:  # go through allowed l,lam combinations
      #     alphap=qnsetp["alpha"]
      #     for ip in range(self.npoints): 
      #       for iq in range(self.nqpoints):
      #         indxidmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
      #         for jp in range(self.npoints): 
      #           for jq in range(self.nqpoints):
      #             indxidmatp=self.npoints*self.nqpoints*alphap+self.npoints*jq+jp
      #             idmat[indxidmat,indxidmatp]=np.sum(splalpha[indxidmat,jq,:]
      #                           *splalphap[indxidmatp,iq,:])    

      # now get the wave function

      wf = np.zeros((self.nalpha*self.nqpoints*self.npoints))
      # first reshape faddeev so that it is in terms of one index only
      fadcomp=fadcomp.reshape((self.nalpha*self.nqpoints*self.npoints))   

      # now perform summation over one index
      wf = np.einsum("ij,j->i", idmat + alpha * self.pmat, fadcomp)
      print(wf.shape)

      wf = wf.reshape((self.nalpha,self.nqpoints,self.npoints))

      return wf
    
    # application of the Faddeev kernel for the iterative solver 
    
    def faddeev(self,psiin):
        """Applies the Faddeev kernel using only two-body interactions.
             
           Parameter:
           psiin -- incoming Faddeev component

           It assumed that the tmat and G0 have been prepared for the given energy 
           before. 
           
        """
        
        # use matrix multiplication and prepared permutation matrix
        psitmp=2.0*self.pmat.dot(psiin.reshape(-1)).reshape(psiin.shape)
        
        psiout=np.empty(psiin.shape)
        
        for qnset in self.qnalpha:
          l=qnset["l"] 
          alpha=qnset["alpha"]
          psiout[alpha,:,:]=np.einsum("ijk,ik->ij",self.tmat[l,:,:,:],psitmp[alpha,:,:])  
          #for iq in range(self.nqpoints):
          #  for ip in range(self.npoints):
          #    psiout[alpha,iq,ip]=np.sum(self.tmat[l,iq,ip,:]*psitmp[alpha,iq,:])
        
        # now multiply with G0        
        for alpha in range(self.nalpha):
          psiout[alpha,:,:]=psiout[alpha,:,:]*self.G0  
        
        return psiout
        
    
    # define a scalar product for testing and for the iterative solver 
    
    def skalp(self,psi1,psi2):
        """Calculate scalar product of two wave functions."""
        
        # multiply with integation weights and sum all
        psitmp=np.zeros((self.nalpha,self.nqpoints,self.npoints),dtype=np.double) 
        
        skaval=np.einsum("ijk,k,j->",psi2*psi1,self.pweight*self.pgrid**2,self.qweight*self.qgrid**2)
        #for alpha in range(self.nalpha):
        # for iq in range(self.nqpoints):
        #  for ip in range(self.npoints):
        #   psitmp[alpha,iq,ip]=psi2[alpha,iq,ip] \
        #                           *self.pweight[ip]*self.pgrid[ip]**2  \
        #                           *self.qweight[iq]*self.qgrid[iq]**2
        
        return skaval

# preparation for G0 for a given energy 

    def prep_G0(self,E):
      """Prepares G0 for a given energy.
      
          E -- three-body energy in fm**-1 
          
          returns G0 
      """  
    
      G0=np.empty((self.nqpoints,self.npoints),dtype=np.double)  
      for iq in range(self.nqpoints):
        for ip in range(self.npoints):
          G0[iq,ip]=1.0/(E-0.75*self.qgrid[iq]**2/self.mass-self.pgrid[ip]**2/self.mass)
        
      return G0  
    
    
# set up set of equations and calculate eigenvalues iteratively 

    def eigv_iter(self,E,neigv):
        """Solve three-body Faddev equation and return n-th eigenvalue and Faddeev component. 

         Parameters:
         E -- energy used in the integral equation in fm**-1 
         neigv -- number of the eigenvalue to be used"""

        self.fadsolvetime-=timeit.default_timer()
    
        # get tmatrix for given energy
        self.tmat=self.prep_tmat(E)
        # calculate G0 for E 
        self.G0=self.prep_G0(E) 
        
        # define a start vector of a constant value in each component
        # and normalize   
        psistart=np.ones((self.nalpha,self.nqpoints,self.npoints),dtype=np.double)
        norm=self.skalp(psistart,psistart)
        psistart=(1/np.sqrt(norm))*psistart
        
        # define array for basis vectors, first one is start vector 
        psiv=np.empty((self.maxiter+1,self.nalpha,self.nqpoints,self.npoints),dtype=np.double)
        psiv[0,:,:,:]=psistart[:,:,:]
        
        # define array for < v_i | K | v_j > 
        bmat=np.zeros((self.maxiter+1,self.maxiter+1),dtype=np.double)
        # for comparison to check convergence 
        lasteta=0.0   
        
        for n in range(self.maxiter):  # start iteration 
          # apply kernel   
          psiw=self.faddeev(psiv[n,:,:,:])
          # count iterations for stastics 
        
          self.numiter+=1  
          # orthogonalize 
          psitildew=np.empty(psiw.shape,dtype=np.double)
          psitildew[:,:,:]=psiw[:,:,:] 
        
          for k in range(n+1):
            skaprod=self.skalp(psiv[k,:,:,:],psiw)    
            psitildew-=skaprod*psiv[k,:,:,:]
            # keep relevant matrix elements in bmat 
            bmat[k,n]=skaprod
          
          # now normalize the orthogonal wtilde and store new basis vector 
          skaprod=self.skalp(psitildew,psitildew)
          psitildew=(1/np.sqrt(skaprod))*psitildew 
          psiv[n+1,:,:,:]=psitildew[:,:,:]
          # finallay store last new element of bmat           
          bmat[n+1,n]=np.sqrt(skaprod)  
          
          # in each step eigenvalues of bmat can be calculated and compared to previous eigenvalues
          # determine eigenvalues using numpy's eig method (only use dimensions already defined)
          evalue,evec=np.linalg.eig(bmat[0:n+1,0:n+1])
    
          # I now assume that the relevant eigenvalues are real to avoid complex arithmetic 
          evalue=np.real(evalue)
        
          # remove neigv-1 largest eigenvalues 
          for n in range(neigv-1):
            maxpos=np.argmax(evalue)
            evalue[maxpos]=0.0
    
          # and take the next one 
          maxpos=np.argmax(evalue)
          eigv=evalue[maxpos]
          if (np.abs(lasteta-eigv)<self.eigtol): # converged, stop iteration 
            break 
            
          lasteta=eigv   
        
        # now we assume a converged eigenvalue
        # use the corresponding eigenvector to obtaine the Faddeev component
        fadcomp=np.einsum("l,lijk->ijk",np.real(evec[:,maxpos]),psiv[0:n+1,:,:,:])
        self.fadsolvetime+=timeit.default_timer()

        return eigv,fadcomp 
    
# the following routines can be omitted in first reading. They are only useful for testing purposes.

    def testperm(self,psi1,psi2):
        """Test permutation matrix."""
        
        # first scalar product without permutation 
        product=self.skalp(psi1,psi2)
        
        # now apply permutation 
        psitmp=self.pmat.dot(psi2.reshape(-1)).reshape(psi2.shape)

        # and build the scalar product (only q integration is necessary)
        for iq in range(self.nqpoints):
           psitmp[:,iq,:] = psitmp[:,iq,:] * self.qgrid[iq]**2*self.qweight[iq]
        
        permprod=np.sum(psitmp*psi1)
    
        print("Permutation test:  {0:15.6e}   {1:15.6e}".format(product,permprod))
    
    def testfu(self):
        """Prepares a fully symmetrical wave function."""
        
        psitmp=np.zeros((self.nalpha,self.nqpoints,self.npoints),dtype=np.double)
        for qnset in self.qnalpha:
         if qnset["l"]==0 and qnset["lam"]==0:  
           alpha=qnset["alpha"] 
           for iq in range(self.nqpoints):
            for ip in range(self.npoints):
             x=self.pgrid[ip]**2 +0.75*self.qgrid[iq]**2      
             psitmp[alpha,iq,ip]=np.exp(-0.05*x)/(0.05*x**2+10.0)
                
                
        return psitmp
    
    def printwf(self,psi):
        """Prints wave function."""
     
        psitest=psi.reshape(-1)        
        for j in range(self.npoints*self.nqpoints*self.nalpha):    
            alphap=j//(self.npoints*self.nqpoints)
            jq=(j-alphap*self.npoints*self.nqpoints)//self.npoints
            jp=(j-self.npoints*self.nqpoints*alphap-self.npoints*jq)#

            print("{0:s}   {1:4d}    {2:4d} {3:4d} {4:4d}       {5:15.6e}  {6:15.6e}".format("testfu",j,jp,jq,alphap,psitest[j],psi[alphap,jq,jp]))
    
   
    def comparewf(self,psi1,psi2,tolrel,tolabs):
        """Compares two wave functions."""
     
        for j in range(self.npoints*self.nqpoints*self.nalpha):    
            alphap=j//(self.npoints*self.nqpoints)
            jq=(j-alphap*self.npoints*self.nqpoints)//self.npoints
            jp=(j-self.npoints*self.nqpoints*alphap-self.npoints*jq)
            
            if abs(psi1[alphap,jq,jp]-psi2[alphap,jq,jp])>tolabs \
               or abs(psi1[alphap,jq,jp]-psi2[alphap,jq,jp])/max(abs(psi1[alphap,jq,jp]),tolabs) > tolrel:
            
              print("{0:s}   {1:4d}    {2:4d} {3:4d} {4:4d}       {5:15.6e}  {6:15.6e}    {7:15.6e}".format("Compare:",j,jp,jq,alphap,
                                        psi1[alphap,jq,jp],psi2[alphap,jq,jp],(psi1[alphap,jq,jp]-psi2[alphap,jq,jp])/psi1[alphap,jq,jp]))


    
