'''
This file contains the classes that are modified for our problem, namely
the TwoBodyTMat class and the ThreeBody class. They are added with the suffix
Ferm at the end to distinguish from the boson case.
'''

from tbs_cls import TwoBody, Cubherm

import numpy as np
import math as m
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm
from sympy.physics.quantum.cg import CG, Wigner9j, Wigner6j


import timeit



# next extend the class for twobody to scattering 

class TwoBodyTMatFerm(TwoBody):
    """This class defines the off-shell t-matrix for a three-body bound state.
    
       The class assumes three identical particles. It also contains the 
       initialization of grid points for the three-body problem. 
       
       Here we modify it for fermionic case. We add the following:

    """
    
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, 
                            nq1=20, nq2=10, qa=1.0, qb=5.0, qc=20.0, 
                            mass=938.92,lmax=0,bj=0.5,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            nrho1=20, nrho2=10, rhoa=1.0, rhob=5.0, rhoc=20.0, 
                            np1four=200,np2four=100, spinpot=False):
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
           jmax  -- maximal two-body total angular momentum
           """
        
        
        # first use the TwoBody class to keep the main parameters 
        super().__init__(pot,np1,np2,pa,pb,pc,mass/2,0,nr1,nr2,ra,rb,rc,np1four,np2four, spinpot)

        self.nq1 = nq1
        self.nq2 = nq2
        self.nqpoints  = nq1+nq2 
        self.mass=mass/self.hbarc
        self.qa=qa
        self.qb=qb
        self.qc=qc
        self.lmax=lmax 
        self.smax=1
        self.jmax = lmax + self.smax
        # self.jmax = jmax
        # self.smax= jmax - lmax
        self.bj=bj
        self.tmax = 1

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

        # create alpha matrix for reduced storage

        self.alpha_list = []
        m=0
        for l in range(self.lmax+1):
          for s in range(self.smax+1):
#             for j in range(self.jmax+1):
            j = l + s
            if j < bj:
              for t in range(self.tmax+1):
                  self.alpha_list.append({"m":m, "l":l, "s":s, "j":j, "t":t})
                  m+=1

        print("off-shell matrix quantum numbers:")
        print("{0}  {1}  {2}  {3}  {4}".format(*list(self.alpha_list[0].keys())))

        for alpha_dict in self.alpha_list:
          print("{0}  {1}  {2}  {3}  {4}".format(*list(alpha_dict.values())))

        self.nalpha2N = len(self.alpha_list)
        # and prepare actual potential matrix elements for all angular momenta
        self.vmatpw=np.empty((self.npoints,self.npoints,self.nalpha2N),dtype=np.double)
        for m, alpha in enumerate(self.alpha_list):
          l = alpha["l"]
          s = alpha["s"]
          j = alpha["j"]
          t = alpha["t"]
          for i in range(self.npoints):
              for k in range(self.npoints):
                self.vmatpw[i,k,m]=self.pot.v(self.pgrid[i],self.pgrid[k],l,s,j,t)
        
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
      tmat=np.empty((self.nalpha2N,self.nqpoints,self.npoints,self.npoints),dtype=np.double)
      # print(tmat.shape)
      for m, _ in enumerate(self.alpha_list):
        # print(m)

        for ie in range(self.nqpoints):
          # to deal with delta function
          amat=np.identity(self.npoints,dtype=np.double)
          # now add the second part of the definition of the matrix   
          # below is naive version, go back to this if need to debug
          # for i in range(self.npoints):
          #     for k in range(self.npoints):
          #         amat[i,k]+=-self.vmatpw[i,k,m]*self.pgrid[k]**2 \
          #                       /(etmat[ie]-self.pgrid[k]**2/(2*self.mred))*self.pweight[k]  

          amat-=np.einsum("ij...,j->ij...",self.vmatpw[:,:,m],self.pgrid[:]**2 \
                                /(etmat[ie]-self.pgrid[:]**2/(2*self.mred))*self.pweight[:])
          bmat=self.vmatpw[:,:,m]

          # finally solve set of equations and store in complete array 
          tmat[m,ie,:,:]=np.linalg.solve(amat,bmat)
        
      self.tmattime+=timeit.default_timer() 
      # return offshell matrix for all energies and angular momenta   
      return tmat
            

   
# definition of a ThreeBody class for the calculation of bound states fermions


class ThreeBodyFerm(TwoBodyTMatFerm):
    """Provides routines for the implementation of the permutation operator and application of the bound state kernel."""
        
    def __init__(self, pot, np1=20, np2=10, pa=1.0, pb=5.0, pc=20.0, 
                            nq1=20, nq2=10, qa=1.0, qb=5.0, qc=20.0, 
                            nx=12,
                            mass=938.92,lmax=0, l3max=0, bj=0.5,
                            nr1=20, nr2=10, ra=1.0, rb=5.0, rc=20.0, 
                            nrho1=20, nrho2=10, rhoa=1.0, rhob=5.0, rhoc=20.0, 
                            np1four=200,np2four=100,verbose=False,spinpot=False):     
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
           jmax  --- maxilamal two-body total angular momentum
           l3max -- maximal one-body orbital angular momentum
           bs  -- total spin S, always set to 1.5

           spinpot -- flag to enable spin-dependent potential
        """
        
        # first initialize the tmatrix class (do not calc the tmatrix yet)
        super().__init__(pot,np1,np2,pa,pb,pc,nq1,nq2,qa,qb,qc,
                         mass,lmax,bj,
                         nr1,nr2,ra,rb,rc,nrho1,nrho2,rhoa,rhob,rhoc,
                         np1four,np2four,spinpot)
        
        # prepare angular grid points for the permutation operator 
        self.nx=nx
        self.xp,self.xw = leggauss(nx)
        
        # prepare partial wave channels for total angular momentum bl
        # parameters for two-body system already defined in TwoBodyTMat
        
        # 3rd particle parameters
        # smax, jmax, tmax are all defined in TwoBodyTmat object
        self.lammax = l3max   # max for orbital angular momentum for 3rd particle
        self.s3 = 0.5   # always fixed to be spin-1/2 particles
        self.t3 = 0.5   # isospin state is always 1/2 for proton / neutron

        # total parameters
        self.j3max = self.lammax + self.s3
        self.bj = bj
        self.bjmax = self.jmax + self.j3max
        self.blmax = int(self.bjmax-0.5)
        self.bsmax = self.bj

        # control parameters
        self.verbose = verbose
        
        
        # now construct pertial wave channels, including:
        # l, l3, bl, s, s3, bs, j, j3, bj, t, t3, bt
        self.qnalpha=[]
        alpha=0
        alpha_12 = 0

        start = timeit.default_timer()

        for l in range(lmax+1):
            for s in range(self.smax+1):
              j = l + s
              for t in range(self.tmax+1):
                
                # if(l%2==0):   # take only symmetric pw (Pauli)  
                # for lam in range(abs(l-bl),l+bl+1):
                for lam in range(self.lammax+1):
                  # for j3 in np.arange(abs(j-self.bj),j+self.bj+1):
                  for j3 in np.arange(0.5, self.j3max+1, 1):
                    bl = l + lam
                    bs = s + self.s3
                    bt = t + self.t3
                    bj = bl + bs
                    if bj == self.bj:
                      self.qnalpha.append({"alpha":alpha, "alpha_12":alpha_12,
                                            "l":l,"lam":lam, "bl":bl,
                                            "s":s, "s3":self.s3, "bs":bs,
                                            "j":j, "j3":j3, "bj":bj,
                                            "t":t, "t3":self.t3, "bt":bt
                                          })
                      alpha+=1
                
                alpha_12 += 1


        # array for bl, bs
        self.blbs_list = []
        for bl in range(self.blmax+1):
          bs = bj-bl
          self.blbs_list.append({"bl":bl, "bs":bs})

        self.nalpha=len(self.qnalpha)
        end = timeit.default_timer()
        if self.verbose:
          print("Duration of alpha list construction: ", end-start)

        
        # print partial wave channels
        self.print_channels()
        

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
        
        
        self.gpreptime=-timeit.default_timer()        
        # this prepares the G function and splines to be used for the preparation of the 
        # kernel later (pmat = permutation matrix)
        self.pmat=self._prep_perm()
        self.gpreptime+=timeit.default_timer()

        if self.verbose:
          print("Permutation matrix element computation complete. Time={0:.4e}".format(self.gpreptime))


        self.fadpreptime=0
        self.fadsolvetime=0
        self.numiter=0

    def print_channels(self):
      '''Print partial wave channels'''
      # print states used
      alpha_keys = list(self.qnalpha[0].keys())
      print("three-body quantum numbers:")
      print("{0:>10s}  {1:>6s} {2:>6s} {3:>6s} {4:>6s} {5:>6s} {6:>6s} {7:>6s} {8:>6s} {9:>6s} {10:>6s} {11:>6s} {12:>6s} {13:>6s}".format(*alpha_keys))
      for qnset in self.qnalpha:
        alpha_vals = list(qnset.values())
        print("{0}    {1}   {2}   {3}   {4}   {5}   {6}   {7}   {8}   {9}   {10}   {11}   {12}   {13}".format(*alpha_vals))

        
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
      
    # KW: here the permutation operator is constructed
    # KW: we need to modify here to include spin CG coefficients
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
        ylam=np.empty((nlamindx,self.smax+1, self.jmax+1,self.tmax+1, self.nx),dtype=np.double)
        for lam in range(self.lammax+1):
          for mu in range(-lam,lam+1):
            ylam[self._lmindx(lam,mu),:,:,:,:]=np.real(sph_harm(mu,lam, 0, thetapp))
        self.timeylam+=timeit.default_timer()
        
        self.timeyl-=timeit.default_timer()
        # array for Y_{l mu}(-q-0.5qp) (real is sufficient since phi=0)
        yl=np.empty((nlindx,self.smax+1, self.jmax+1,self.tmax+1,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for l in range(self.lmax+1):
          for mu in range(-l,l+1):
            yl[self._lmindx(l,mu),:,:,:,:,:,:]=np.real(sph_harm(mu,l, 0, thetap))
        self.timeyl+=timeit.default_timer()

        self.timeystarl-=timeit.default_timer()
        # array for Y*_{l mu}(0.5q+qp) (real is sufficient since phi=0)
        ystarl=np.empty((nlindx,self.smax+1, self.jmax+1,self.tmax+1,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for l in range(self.lmax+1):
          for mu in range(-l,l+1):
            ystarl[self._lmindx(l,mu),:,:,:,:,:,:]=np.real(sph_harm(mu,l, 0, theta))
        self.timeystarl+=timeit.default_timer()

        # now prepare the necessary Clebsch-Gordan coefficients
        # we need (l lam L, M 0 M)  and (l lam L,mu M-mu,M)
        # I assume that L is smaller than the lmax or lammax therefore M=-L,L
        # the smallest index for storage 
        self.timeclebsch-=timeit.default_timer()

        cg=np.zeros((self.nalpha,self.blmax+1, 2*self.blmax+1),dtype=np.double)
        cgp=np.zeros((self.nalpha,self.blmax+1,2*self.blmax+1, 2*self.lmax+1),dtype=np.double)

        for blbs in self.blbs_list:
          bl = blbs["bl"]
          for qnset in self.qnalpha:
            alpha=qnset["alpha"]
            bl_alpha = qnset["bl"]

          # make array for each bl over here
          # cg_bl = np.zeros(2*bl+1,dtype=np.double)
          # cgp_bl = np.zeros((2*self.lmax+1, 2*bl+1),dtype=np.double)
          if bl == bl_alpha:
            for bm in range(-bl,bl+1):
              # cg_bl[bm+bl]=float(CG(qnset["l"],bm,qnset["lam"],0,bl,bm).doit())
              # print(qnset["l"],bm,qnset["lam"],0,bl,bm, float(CG(qnset["l"],bm,qnset["lam"],0,bl,bm).doit()))
              cg[qnset["alpha"],bl, bm+bl]=float(CG(qnset["l"],bm,qnset["lam"],0,bl,bm).doit())
              for mu in range(-qnset["l"],qnset["l"]+1):
                # print(qnset["l"],mu,qnset["lam"],bm-mu,bl,bm, float(CG(qnset["l"],mu,qnset["lam"],bm-mu,bl,bm).doit()))
                # cgp_bl[mu+qnset["l"],bm+bl] = float(CG(qnset["l"],mu,qnset["lam"],bm-mu,bl,bm).doit())
                cgp[qnset["alpha"],bl,bm+bl,mu+qnset["l"]]=float(CG(qnset["l"],mu,qnset["lam"],bm-mu,bl,bm).doit())
        
        # print("cg, cgp nonzero ", np.nonzero(cg), np.nonzero(cgp))
        self.timeclebsch+=timeit.default_timer()

        # now we can perform the mu summation for the combination of coupled spherical harmonics 
        self.timeylylam-=timeit.default_timer()

        ylylam=np.zeros((self.nalpha,self.blmax+1,2*self.blmax+1,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for blbs in self.blbs_list:
          bl=blbs["bl"]
          for qnset in self.qnalpha:  # go through allowed l,lam combinations
              alphap=qnset["alpha"]
              l=qnset["l"]
              lam=qnset["lam"]
              s=qnset["s"]
              s3 = qnset["s3"]
              j=qnset["j"]
              j3=qnset["j3"]
              t=qnset["t"]
              t3=qnset["t3"]
              bl_alpha=qnset["bl"]
              # print(l,s,j,t)
              if bl == bl_alpha:
                for bm in range(-bl,bl+1):
                    for mu in range(-l,l+1):
                      lmindx=self._lmindx(l,mu)
                      if abs(bm-mu)<=lam:
                        lamindx=self._lmindx(lam,bm-mu)
                        ylylam[alphap,bl,bm+bl,:,:,:]+=cgp[alphap,bl,bm+bl,mu+l]*yl[lmindx,s,j,t,:,:,:]*ylam[lamindx,s,j,t,:]
        # print("ylylam, ylylamp nonzero ", np.nonzero(ylylam))
        self.timeylylam+=timeit.default_timer()

        self.timegcalc-=timeit.default_timer()


        # evaluate G wiht L first then afterwards perform summation over L and S
        gfunc_bl=np.zeros((self.nalpha,self.nalpha,self.blmax+1,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)


        # gfunc=np.zeros((self.nalpha,self.nalpha,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alpha=qnset["alpha"]
          l=qnset["l"]
          lam=qnset["lam"]
          bl = qnset["bl"]
          s=qnset["s"]
          s3 = qnset["s3"]
          j=qnset["j"]
          j3=qnset["j3"]
          t=qnset["t"]
          t3=qnset["t3"]
          for qnsetp in self.qnalpha:  # go through allowed l,lam combinations
            alphap=qnsetp["alpha"]
            for bm in range(-bl,bl+1):
              if(abs(bm)<=l):  
                lmindx=self._lmindx(l,bm) 
                # print(lmindx)
                # print(gfunc_bl[alpha,alphap,bl,:,:,:].shape)
                # print(ystarl[lmindx,:,:,:].shape)
                # print(ylylam[alphap,bl,bm+bl,:,:,:].shape)
                # print(cg[alpha,bl,bm+ bl].shape)
                gfunc_bl[alpha,alphap,bl,:,:,:]=8*m.pi**2*np.sqrt((2*lam+1)/(4*m.pi))/(2*bl+1) \
                   *ystarl[lmindx,s,j,t,:,:,:]*ylylam[alphap,bl,bm+bl,:,:,:]*cg[alpha,bl,bm+ bl]

        # now sum over all L, S
        gfunc=np.zeros((self.nalpha,self.nalpha,self.nqpoints,self.nqpoints,self.nx),dtype=np.double)


        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alpha=qnset["alpha"]
          l=qnset["l"]
          lam=qnset["lam"]
          s=qnset["s"]
          s3 = qnset["s3"]
          j=qnset["j"]
          j3=qnset["j3"]
          t=qnset["t"]
          t3=qnset["t3"]
          bj=qnset["bj"]
          bt=qnset["bt"]
          bl_alpha = qnset["bl"]
          bs_alpha = qnset["bs"]

          for qnsetp in self.qnalpha:  # go through allowed l,lam combinations

            # not sure how this prime stuff works here
            alphap=qnsetp["alpha"]
            lp = qnsetp["l"]
            lamp = qnsetp["lam"]
            sp = qnsetp["s"]
            s3p = s3
            jp = qnsetp["j"]
            j3p = qnsetp["j3"]
            tp= qnsetp["t"]
            t3 = qnsetp["t3"]
            bl_alphap = qnsetp["bl"]
            bs_alphap = qnsetp["bs"]
            btp = qnsetp["bt"]
          
            if (bt == btp and bl_alpha == bl_alphap and bs_alpha == bs_alphap):


              for blbs in self.blbs_list:
                  bl = blbs["bl"]
                  bs = blbs["bs"]

                  if (bl == bl_alpha and bs == bs_alpha):   
                    # print("unprimed states: ", l, s, j, lam, s3, j3, bl, bs, bj)
                    # print("primed states: ", lp, sp, jp, lamp, s3, j3p, bl, bs, bj)
                    # evaluate wigner 9j symbol
                    c9j = float(Wigner9j(l, s, j, lam, s3, j3, bl, bs, bj).doit())
                    c9jp = float(Wigner9j(lp, sp, jp, lamp, s3, j3p, bl, bs, bj).doit())

                      # evaluate coefficients before that 
                    coeff = (2*bs + 1) * np.sqrt((2*j+1) * (2*jp+1) * (2*j3+1) + (2*j3p + 1))

                    # evaluate spin component -> Wigner 6j symnbols
                    c6j_s = float(Wigner6j(s3, 0.5, sp, s3, bs, s).doit())
                    spin_part = (-1)**s * np.sqrt((2*s + 1) * (2*sp + 1)) * c6j_s

                    # include isospin
                    c6j_t = float(Wigner6j(t3, 0.5, tp, t3, bt, t).doit())
                    isospin_part = (-1)**t * np.sqrt((2*t + 1) * (2*tp + 1)) * c6j_t

                    # print(c9j, c9jp, c6j_s, c6j_t)

                    factor = c9j * c9jp * coeff * spin_part * isospin_part

                    gfunc_bl[alpha,alphap,bl,:,:,:] *= factor

              gfunc[alpha, alphap, :, :, :] = self.bsmax * np.sum(gfunc_bl[alpha,alphap,:,:,:,:], axis=0)

              #         for bm in range(-bl,bl+1):
              #             if(abs(bm)<=l):  
              #                 lmindx=self._lmindx(l,bm) 
              #                 orbital_part = 8*m.pi**2*np.sqrt((2*lam+1)/(4*m.pi))/(2*bl+1) \
              #                     *ystarl[lmindx,s,j,t,:,:,:]*ylylam[alphap,bl,bm+bl,:,:,:]*cg[alphap,bl,bm+bl]
              # #                 print(bm+bl)
              #                 print(np.nonzero(ylylam[alphap,bl,bm+bl,:,:,:]))

              #                 gfunc_bl[alpha, alphap, bl, :,:,:] += coeff * c9j * c9jp * spin_part * isospin_part * orbital_part

      
        # print(gfunc.shape)
        # print("gfunc nonzero", np.nonzero(gfunc))
        
        self.timegcalc+=timeit.default_timer()
        #  now we assume that there is a function on p on the left defined by p**l and on the right devided by p'**l' 
        # that is interpolated using Cubherm to pi and pip 
        #
        # set spline elements based on grid points and shifted momenta 
        self.timespl-=timeit.default_timer()
        splpi=Cubherm.spl(self.pgrid,pi)
        splpip=Cubherm.spl(self.pgrid,pip)
        
        # interpolation fspl=np.sum(spl*fold,axis=1) first axis is pgrid 
        # prepare splines multiplied by p**l factors (splalpha also includes the integration weights for q' and x integral)
        
        splalpha=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
        splalphap=np.empty((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx),dtype=np.double)
        # splalpha=np.empty((self.npoints*self.nqpoints*self.nalpha,self.nqpoints,self.nx),dtype=np.double)
        # splalphap=np.empty((self.npoints*self.nqpoints*self.nalpha,self.nqpoints,self.nx),dtype=np.double)
        
        for qnset in self.qnalpha:  # go through allowed l,lam combinations
          alpha=qnset["alpha"]
          l=qnset["l"]
            
          # ijkl : iq ip jq ix  
          splalpha[alpha,:,:,:,:]=np.einsum("jikl,ikl,j,l,k->ijkl",splpi,pi**l,1.0/self.pgrid**l,self.xw,self.qweight*self.qgrid**2)
          splalphap[alpha,:,:,:,:]=np.einsum("jkil,kil,j->ijkl",splpip,pip**l,1.0/self.pgrid**l)
        
          # for ip in range(self.npoints): 
          #   for iq in range(self.nqpoints):
          #     indxpmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
          #     # for jq in range(self.nqpoints):
          #     #   splalpha[indxpmat,jq,:]=splpi[ip,iq,jq,:]*(pi[iq,jq,:]/self.pgrid[ip])**l*self.xw[:]*self.qweight[jq]*self.qgrid[jq]**2
          #     #   splalphap[indxpmat,jq,:]=splpip[ip,jq,iq,:]*(pip[jq,iq,:]/self.pgrid[ip])**l
          #     splalpha[indxpmat,:,:]=np.einsum("ij,j,i->ij",splpi[ip,iq,:,:]*(pi[iq,:,:]/self.pgrid[ip])**l,self.xw[:],self.qweight[:]*self.qgrid[:]**2)
          #     splalphap[indxpmat,:,:]=splpip[ip,:,iq,:]*(pip[:,iq,:]/self.pgrid[ip])**l
        self.timespl+=timeit.default_timer()
            
        
        self.timepmat-=timeit.default_timer()
        pmat=np.zeros((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha),dtype=np.double)
        
        # also generate views with separated indices 
        #pmatsingle=pmat.reshape((self.nalpha,self.nqpoints,self.npoints,self.nalpha,self.nqpoints,self.npoints))
        
        #splalphapsingle=splalphap.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
        #splalphasingle=splalpha.reshape((self.nalpha,self.nqpoints,self.npoints,self.nqpoints,self.nx))
        
        # ijk : alpha iq ip (indxpmat)
        # lmn : alphap jq jp (indxpmatp)
        # o   : ix 
        
        pmat=np.einsum("ijkmo,iljmo,lmnjo->ijklmn",splalpha,gfunc,splalphap).reshape((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha))
        # pmat=pmatsingle.reshape((self.npoints*self.nqpoints*self.nalpha,self.npoints*self.nqpoints*self.nalpha))
        
        # for qnset in self.qnalpha:  # go through allowed l,lam combinations
        #   alpha=qnset["alpha"]
        #   for qnsetp in self.qnalpha:  # go through allowed l,lam combinations
        #     alphap=qnsetp["alpha"]
        #     for ip in range(self.npoints): 
        #       for iq in range(self.nqpoints):
        #         indxpmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
        #         for jp in range(self.npoints): 
        #           for jq in range(self.nqpoints):
        #             indxpmatp=self.npoints*self.nqpoints*alphap+self.npoints*jq+jp
        #             pmat[indxpmat,indxpmatp]=np.sum(splalpha[indxpmat,jq,:]
        #                           *gfunc[alpha,alphap,iq,jq,:]
        #                           *splalphap[indxpmatp,iq,:])                   
        self.timepmat+=timeit.default_timer()                  
                                      
        return pmat
        
    def prep_faddeev(self,ener):
        """Prepares the Faddeev kernel as a matrix using only two-body interactions.
        
           Parameter:
           ener -- three-body energy in fm-1
        """
 
        # get tmatrix for given energy
        self.tmat=self.prep_tmat(ener)
        if self.verbose:
          print("tmatrix element computation complete. Time={0:.4e}".format(self.tmattime))
        
        self.fadpreptime-=timeit.default_timer()
        
        # use matrix multiplication of preprepared permutation matrix 
        # self.pmat[indxpmat,indxpmatp] contains permutation matrix 
        # indexpmat is alpha,iq,ip 

        self.kfadmat=np.zeros(self.pmat.shape,dtype=np.double)
        
        for qnset in self.qnalpha:
            alpha12=qnset["alpha_12"]
            alpha=qnset["alpha"]
            for iq in range(self.nqpoints):
                for ip in range(self.npoints):
                  indxkmat=ip+self.npoints*iq+self.npoints*self.nqpoints*alpha
                  for jp in range(self.npoints):
                    indxpmat=jp+self.npoints*iq+self.npoints*self.nqpoints*alpha
                    self.kfadmat[indxkmat,:]+=self.tmat[alpha12,iq,ip,jp]*2*self.pmat[indxpmat,:]
    
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

        if self.verbose:
          print("Faddeev element computation complete. Time={0:.4e}".format(self.fadpreptime))

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
      start=timeit.default_timer()    
      evalue,evec=np.linalg.eig(self.kfadmat)
      if self.verbose:
        print("evalue evaluation: ", timeit.default_timer() - start)
    
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
      start=timeit.default_timer()    
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
      if self.verbose:
        print("norm evaluation: ", timeit.default_timer() - start)
        
      fadcomp=(1/np.sqrt(norm))*fadcomp
    
      self.fadsolvetime+=timeit.default_timer()
      if self.verbose:
          print("Faddeev component computation complete. Time={0:.4e}".format(self.fadsolvetime))
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
        print("eta1, eta2 for starting energies: ", eta1, eta2)
        niter=0
        
        start = timeit.default_timer()
        while abs(e1-e2) > tol: 
          # get new estimate (taking upper value into account)   
          enew=e2+(e1-e2)/(eta1-eta2)*(1-eta2) 
          enew=min(elow,enew)
       
          # get new eigenvalue and replace e1 and e2 for next iteration
          eta,fadcomp=self.eigv(enew,neigv)
          e2=e1
          eta2=eta1
          e1=enew
          eta1=eta 
            
          # if self.verbose:
          print("Enew=", e1)
          print("eta_new=", eta)

          # break if loop is taking too long
          niter+=1
          if niter > nitermax:
            break

        end=timeit.default_timer()

        if self.verbose:
          print("Energy search computation complete. Time={0:.4e}".format(end-start))
            
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
      # for qnset in self.qnalpha:  # go through allowed l,lam combinations
      #   alpha=qnset["alpha"]
      #   for ip in range(self.npoints): 
      #     for iq in range(self.nqpoints):
      #       indxidmat=self.npoints*self.nqpoints*alpha+self.npoints*iq+ip
          
      #       wf[indxidmat] = np.dot((idmat[indxidmat,:] + alpha*self.pmat[indxidmat,:]), fadcomp)
      wf = np.einsum("ij,j->i", idmat + alpha * self.pmat, fadcomp)
      # print(wf.shape)

      wf = wf.reshape((self.nalpha,self.nqpoints,self.npoints))

      # normalize the wavefunction
      wf = (1 / np.sqrt(self.skalp(wf, wf))) * wf


      

      return wf

#     def wavefunc(self, fadcomp):
#         '''Wavefunction evaluated from Faddeev component'''
        
    
    def eval_ekin(self, fadout, wf):
        '''
        Evaluate the kinetic energy using the Faddeev component and the 
        wave function, i.e. <T> = <Psi|H0|Psi> = 3*< fad | H0 | Psi>
        '''
        
        Tmat = np.zeros((self.nqpoints, self.npoints), dtype=np.double)
        for iq in range(self.nqpoints):
            for ip in range(self.npoints):
                # multiply the grid points and the momentum values here
                Tval_1 = 0.75 * self.qgrid[iq]**2. / self.mass 
                Tval_2 = self.pgrid[ip]**2. / self.mass
                pint = self.pgrid[ip]**2. * self.pweight[ip]
                qint = self.qgrid[iq]**2. * self.qweight[iq]
                Tmat[iq, ip] = qint * pint * (Tval_1 + Tval_2)
                
        # now evaluate the expectation value
        # this means the sum
        # multiply by 3 due to normalization
        # h0_fad = 3 * np.sum(wf * Tmat * fadout)
        h0 = np.sum(wf**2*Tmat)
        
        return h0



# set up set of equations and calculate eigenvalues iteratively 

    def eigv_iter(self,E,neigv, maxiter=10, eigtol=1e-6):
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
        psiv=np.empty((maxiter+1,self.nalpha,self.nqpoints,self.npoints),dtype=np.double)
        psiv[0,:,:,:]=psistart[:,:,:]
        
        # define array for < v_i | K | v_j > 
        bmat=np.zeros((maxiter+1,maxiter+1),dtype=np.double)
        # for comparison to check convergence 
        lasteta=0.0   
        
        for n in range(maxiter):  # start iteration 
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
          if (np.abs(lasteta-eigv)<eigtol): # converged, stop iteration 
            break 
            
          lasteta=eigv   
        
        # now we assume a converged eigenvalue
        # use the corresponding eigenvector to obtaine the Faddeev component
        fadcomp=np.einsum("l,lijk->ijk",np.real(evec[:,maxpos]),psiv[0:n+1,:,:,:])
        self.fadsolvetime+=timeit.default_timer()

        return eigv,fadcomp 
    
    
    # the following methods are only useful for testing at this point and can be 
    # ignored at first reading
    
    
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
          alpha12 = qnset["alpha_12"]
          psiout[alpha,:,:]=np.einsum("ijk,ik->ij",self.tmat[alpha12,:,:,:],psitmp[alpha,:,:])  
        
        # now multiply with G0        
        for alpha in range(self.nalpha):
          psiout[alpha,:,:]=psiout[alpha,:,:]*self.G0  
        
        return psiout

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
        
    
    # the following routines are useful for testing the code. 
    
    def skalp(self,psi1,psi2):
        """Calculate scalar product of two wave functions."""
        
        # multiply with integation weights 
        psitmp=np.zeros((self.nalpha,self.nqpoints,self.npoints),dtype=np.double)        
        for alpha in range(self.nalpha):
         for iq in range(self.nqpoints):
          for ip in range(self.npoints):
           psitmp[alpha,iq,ip]=psi2[alpha,iq,ip] \
                                   *self.pweight[ip]*self.pgrid[ip]**2  \
                                   *self.qweight[iq]*self.qgrid[iq]**2
        return np.sum(psi1*psitmp)
    
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

   