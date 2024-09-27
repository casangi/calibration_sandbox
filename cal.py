
from enum import Enum,IntEnum
import numpy as np
import pylab as pl
from math import pi

# Ordered VE Enum
class StandardVisEqTypeEnum(IntEnum):
    UNDEF =   0
    K     =  10     # Geometry (Fourier Kernel errors)
    B     =  20     # Bandpass
    G     =  30     # Electronic Gain (per poln)
    J     =  40     # General Jones
    D     =  50     # Instrumental Polarization Leakage
    X     =  60     # (Forward) Cross-hand phase
    C     =  70     # Feed configuration (absolute orientation)
    P     =  80     # Parallactic angle
    E     =  90     # Efficiency (on-axis)    (review order w.r.t. C,P and measurement conv.)
    T     = 100     # Tropospheric Gain (unpol)
    F     = 110     # Ionospheric Gain (disp. delay and birefringence)
    M     = 300     # Multiplicative non-closing
    A     = 400     # Additive non-closing

# Calibration parameter data type
class ParTypeEnum(Enum):
    # Must we support combined Float & Complex case?
    UNDEF   = 0
    FLOAT   = 1
    COMPLEX = 2

# Calibration Jones matrix type
class MatTypeEnum(Enum):
    # Is this needed at all, if all matrices rendered as general?
    # Must we support linearized general?  (ignores 2nd order terms, for traditional poln alg.)
    UNDEF    = 0
    SCALAR   = 1
    DIAGONAL = 2
    GENERAL  = 4

# Visibility polarization basis
class PolBasisEnum(Enum):
    # What about antenna-dependent pol basis?
    UNDEF    = 0
    LINEAR   = 1
    CIRCULAR = 2


# simulation helper function
def calcCheby(sh,ax,nC,C,dC):
    csh=(nC,)+sh[:ax]+sh[ax+1:]
    CC=np.random.uniform(-1,1,csh)
    CC=CC*pl.array(dC).reshape((nC,1,1,1,1))+pl.array(C).reshape((nC,1,1,1,1))
    nSamp=sh[ax]
    x=(2*np.arange(nSamp)-(nSamp-1))/nSamp  # [-1,1], subdivided symmetrically
    tr=tuple(range(ax))+(4,)+tuple(range(ax,4)) # move last to ax
    #print(' tr=',tr)
    Ch=np.polynomial.chebyshev.chebval(x,CC).transpose(tr)  
    #print('osh=',Ch.shape)
    return Ch


# VisJones base class for calibration terms
class VisJones(object):
    globverbose=True
    def __init__(self,
                 visEqType,
                 calTypeName,
                 parType,
                 nPol,
                 nPar,
                 matType,
                 chanDepPar,
                 chanDepMat,
                 initparval=[0],
                 verbose=False):
        self.visEqType=visEqType
        self.calTypeName=calTypeName
        self.parType=parType
        self.nPol=nPol
        self.nPar=nPar
        self.matType=matType
        self.chanDepPar=chanDepPar
        self.chanDepMat=chanDepMat

        self.calFileName='<calfilename(TBD)>'
        self.polBasis=PolBasisEnum.LINEAR      # hardwired here; later, from assoc. data

        self.VEindex=None    # generic index within (and set by) a VisEquation
        
        self.nTimePar=-1
        self.nAntPar=-1
        self.nChanPar=-1
        self.nChanMat=-1
        self.initparval=np.array(initparval).reshape((1,1,1,1,len(initparval)))
        self.pars=np.array([])  # initially empty
        self.Jones=np.array([])  # initially empty
        if verbose: self.info()

    @property
    def parShape(self):
        return (self.nTimePar,
                self.nAntPar,
                self.nChanPar,
                self.nPol,
                self.nPar)

    @property
    def polName(self):
        if self.polBasis==PolBasisEnum.LINEAR:
            if self.nPol==1:
                return ['X & Y']
            else:
                return ['X','Y']
        elif self.polBasis==PolBasisEnum.CIRCULAR:
            if self.nPol==1:
                return ['R & L']
            else:
                return ['R','L']
        else:
            raise Exception('VisJones.polName: PolBasis undefined')
            
    def info(self):
        print('VisJones summary:')
        print(' '+
              'visEqType='+self.visEqType.name,
              'calTypeName='+self.calTypeName,
              'parType='+self.parType.name,
              'matType='+self.matType.name,
              'polBasis='+self.polBasis.name)
        print(' '+
              'nTimePar='+str(self.nTimePar),
              'nAntPar='+str(self.nAntPar))
        print(' '+
              'chanDepPar='+str(self.chanDepPar),
              'chanDepMat='+str(self.chanDepMat),
              'nChanPar='+str(self.nChanPar),
              'nChanMat='+str(self.nChanMat))
        print(' '+
              'nPol='+str(self.nPol),
              'nPar='+str(self.nPar))
        print(' '+
              'pars.shape='+str(self.pars.shape),
              'Jones.shape='+str(self.Jones.shape))
        
        
    def plotSpec(self):
        pl.clf()
        polname=self.polName
        fmt=''
        if self.nChanPar==1:
            fmt='o'
        for ipol in range(self.nPol):
            iplt=ipol+1
            Cpl=self.pars[0,:,:,ipol,0].transpose((1,0))
            pl.subplot(2,self.nPol,iplt)
            pl.plot(pl.absolute(Cpl),fmt)
            pl.title(polname[ipol])
            if iplt%4==1:
                pl.ylabel('Cal Amplitude')
            pl.subplot(2,self.nPol,iplt+self.nPol)
            pl.plot(pl.angle(Cpl)*180/pi,fmt)
            pl.xlabel('Channel')
            if iplt%4==1:
                pl.ylabel('Cal Phase (deg)')

    def plotTime(self):
        pl.clf()
        polname=self.polName
        fmt=''
        if self.nTimePar==1:
            fmt='o'
        for ipol in range(self.nPol):
            iplt=ipol+1
            Cpl=self.pars[:,:,self.nChanPar//2,ipol,0]
            pl.subplot(2,self.nPol,iplt)
            pl.plot(pl.absolute(Cpl),fmt)
            pl.title(polname[ipol])
            if iplt%4==1:
                pl.ylabel('Cal Amplitude')
            pl.subplot(2,self.nPol,iplt+self.nPol)
            pl.plot(pl.angle(Cpl)*180/pi,fmt)
            pl.xlabel('Time (index)')
            if iplt%4==1:
                pl.ylabel('Cal Phase (deg)')


    def plotArgand(self):
        pl.clf()
        fmt=''
        if self.nChanPar==1:
            fmt='o'
        for ipol in range(self.nPol):
            Cpl=self.pars[0,:,:,ipol,0].transpose((1,0))
            pl.plot(Cpl.real,Cpl.imag,fmt)
        pl.axis('equal')
        pl.xlabel('Real')
        pl.ylabel('Imag')
        pl.title('Complex Calibraiton')


    # Generically set parameter axes shapes
    #  (does not inflate pars!)
    def setParShape(self,nTime=1,nAnt=1,nChanPar=1):
        # normally, these will be set by ingesting a calset or data
        self.nTimePar=nTime
        self.nAntPar=nAnt
        self.nChanPar=1
        if self.chanDepPar:
            self.nChanPar=nChanPar

        self.nChanMat=self.nChanPar   # for now  (will need to revise for param'd)

        # discard any existing pars, Jones (probably wrong shape!)
        self.pars=np.array([])
        self.Jones=np.array([])

    # Set parameter axes shapes from visData
    def setParShapeFromVisData(self,visData,doTimeDep=False):
        self.nAntPar=visData.nAnt

        if doTimeDep:
            self.nTimePar=visData.nTime
        else:
            self.nTimePar=1
        
        if self.chanDepPar:
            self.nChanPar=visData.nChan
        else:
            # single-channel par
            #  NB: chan-axis parametrized types (e.g.,FringeJones,Bpoly) will generalize locally
            self.nChanPar=1

        if self.chanDepMat:
            self.nChanMat=visData.nChan
        else:
            self.nChanMat=1

        # discard any existing pars, Jones (probably wrong shape!)
        self.pars=np.array([])
        self.Jones=np.array([])
            
            
    # Parameter formation operations
    ################################
    
    # initialize pars to "perfect" values (according to type-specific initparval 
    def initPar(self):
        typeval=0j
        if self.parType==ParTypeEnum.FLOAT:
            typeval=0.0
        self.pars=np.ones(self.parShape,
                          dtype=type(typeval))
        self.pars=self.pars*self.initparval

        # discard any existing Jones
        self.Jones=np.array([])

    # simulate systematic set of pars
    def simPar(self):
        raise Exception('par simulation not yet implemented for '+self.calTypeName)

    # fill pars from a disk file/table
    def fillpar(self):
        raise Exception('NYI')

    def resampleInFreq(self,vis,freqinterp=[]):
        #print('resampleInFreq',self.visEqType.name)
        # if chanDepPar, for now, just ensure cal has same nchan as data or nchan=1
        # (in future, actually perform interp of pars from cal channelization to vis channelization)
        # (freqinterp will contain interpolation parameters)
        if self.chanDepPar:
            assert(self.nChanPar==1 or self.nChanPar==vis.nChan),'nChan mismatch'

    def resampleInTime(self,vis,timeinterp=[]):
        #print('resampleInTime',self.visEqType.name)
        # for now, just assert cal has only one timestamp, or matches vis's time axis
        # in future, perform time-dep interpolation of pars
        # (timeinterp will contain interpolation parameters)
        assert(self.nTimePar==1 or self.nTimePar==vis.nTime),'nTime mismatch'


    # Jones matrix operations

    # Initialize inflated Jones Identity matrix
    def initJones(self):
        self.Jones=np.zeros((self.nTimePar,self.nAntPar,self.nChanMat,2,2),
                            dtype=complex)
        self.Jones[:,:,:,[0,1],[0,1]]=1+0j

    # 'template' for specialized calcJones (all specializations must explicitly implement)
    def calcJones(self):
        raise Exception('Jones calculation not implemented for '+self.calTypeName)

    # Invert the Jones matrices
    def invertJones(self):
        self.Jones=np.linalg.inv(self.Jones)

    # Empty the Jones member
    def clearJones(self):
        self.Jones=np.array([])

    # multiply this VJ by another "in-place"
    def accumulate(self,other,fromRight=True):
        if fromRight:
            # this = this @ other
            np.matmul(self.Jones,other.Jones,out=self.Jones)
        else:
            # this = other @ this
            np.matmul(other.Jones,self.Jones,out=self.Jones)

    # apply this Jones rightward to specified V, indexed by 1st antenna
    def applyRight(self,visData,V):
        Vc=V.reshape(V.shape[:3]+(2,2))   # cast corr axis as 2x2
        # broadcast antenna axis with a1
        # shapes:  Jones[Nt,Na,Nch,2,2]
        #             Vc[Nt,Nb,Nch,2,2]
        np.matmul(self.Jones[:,visData.a1,:,:,:],Vc,out=Vc)   # "in-place"
            
    # apply conjtrans of this Jones leftward to specified V, indexed by 2nd antenna
    def applyLeft(self,visData,V):
        Vc=V.reshape(V.shape[:3]+(2,2))   # cast corr axis as 2x2
        # broadcast antenna axis with a2
        np.matmul(Vc,np.transpose(np.conj(self.Jones),(0,1,2,4,3))[:,visData.a2,:,:,:],
                  out=Vc)  # "in-place"

            
class BJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.B,
                         calTypeName='BJones',
                         parType=ParTypeEnum.COMPLEX,
                         nPol=2,
                         nPar=1,
                         matType=MatTypeEnum.DIAGONAL,
                         chanDepPar=True,
                         chanDepMat=True,
                         initparval=[1+0j],
                         verbose=verbose)

    def simPar(self):
        self.initPar()
        P=calcCheby(self.parShape,ax=2,
                    nC=5,
                    C=5*[0],         # relative to 0
                    dC=[0]+4*[0.25])  # 
                        
        #print('A:')
        A=calcCheby(sh=self.parShape,ax=2,
                    nC=5,
                    C= [1, -0.25, -0.15, 0.0, -0.05],     # relative to 1
                    dC=[0, 0.25,   0.10, 0.05, 0.05])
        self.pars.real=np.cos(P)
        self.pars.imag=np.sin(P)
        self.pars*=A
        
        
    def calcJones(self):
        super().initJones()
        #
        # [ p[0,0] 0      ]
        # [ 0      p[1,0] ]
        #
        self.Jones[:,:,:,[0,1],[0,1]]=self.pars[:,:,:,[0,1],0]
        

class GJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.G,
                         calTypeName='GJones',
                         parType=ParTypeEnum.COMPLEX,
                         nPol=2,
                         nPar=1,
                         matType=MatTypeEnum.DIAGONAL,
                         chanDepPar=False,
                         chanDepMat=False,
                         initparval=[1+0j],
                         verbose=verbose)

    def simPar(self):
        self.initPar()
        P=np.random.uniform(-pi,pi,(1,self.nAntPar,1,2,1))
        A=np.random.normal(1.0,0.1,(1,self.nAntPar,1,2,1))
        self.pars.real=np.cos(P)
        self.pars.imag=np.sin(P)
        self.pars*=A
            
    def calcJones(self):
        super().initJones()
        #
        # [ p[0,0] 0      ]
        # [ 0      p[1,0] ]
        #
        self.Jones[:,:,:,[0,1],[0,1]]=self.pars[:,:,:,[0,1],0]

        
class JJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.J,
                         calTypeName='JJones',
                         parType=ParTypeEnum.COMPLEX,
                         nPol=2,
                         nPar=2,
                         matType=MatTypeEnum.GENERAL,
                         chanDepPar=True,
                         chanDepMat=True,
                         initparval=[1+0j,0j],
                         verbose=verbose)

        
    def calcJones(self):
        super().initJones()
        #
        # [ p[0,0] p[0,1] ]
        # [ p[1,1] p[1,0] ]
        #
        self.Jones[:,:,:,[0,0,1,1],[0,1,0,1]]=self.pars[:,:,:,[0,0,1,1],[0,1,1,0]]
        
class JOrElJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.J,
                         calTypeName='JOrElJones',
                         parType=ParTypeEnum.FLOAT,
                         nPol=2,
                         nPar=2,
                         matType=MatTypeEnum.GENERAL,
                         chanDepPar=True,
                         chanDepMat=True,
                         initparval=[0,0],
                         verbose=verbose)

        
    def simPar(self):
        super().initPar()
        self.pars=calcCheby(self.parShape,ax=2,
                            nC=4,
                            C=4*[0],          # relative to 0
                            dC=[0]+3*[0.02])  # relative to 0
                        

    def calcJones(self):
        super().initJones()
        #
        # [  cos(dOx)cos(dEx)-i.sin(dOx)sin(dEx)  sin(dOx)cos(dEx)+i.cos(dOx)sin(dEx) ]
        # [ -sin(dOy)cos(dEy)-i.cos(dOy)sin(dEy)  cos(dOy)cos(dEy)-i.sin(dOy)sin(dEy) ]
        #
        J=self.Jones             # reference
        dO=self.pars[:,:,:,:,0]  # reference
        dE=self.pars[:,:,:,:,1]  # reference
        J[:,:,:,0,0].real= np.cos(dO[:,:,:,0])*np.cos(dE[:,:,:,0])
        J[:,:,:,0,0].imag=-np.sin(dO[:,:,:,0])*np.sin(dE[:,:,:,0])
        J[:,:,:,0,1].real= np.sin(dO[:,:,:,0])*np.cos(dE[:,:,:,0])
        J[:,:,:,0,1].imag= np.cos(dO[:,:,:,0])*np.sin(dE[:,:,:,0])
        J[:,:,:,1,0].real=-np.sin(dO[:,:,:,1])*np.cos(dE[:,:,:,1])
        J[:,:,:,1,0].imag=-np.cos(dO[:,:,:,1])*np.sin(dE[:,:,:,1])
        J[:,:,:,1,1].real= np.cos(dO[:,:,:,1])*np.cos(dE[:,:,:,1])
        J[:,:,:,1,1].imag=-np.sin(dO[:,:,:,1])*np.sin(dE[:,:,:,1])

    def plotSpec(self):
        pl.clf()
        polname=self.polName
        fmt=''
        if self.nChanPar==1:
            fmt='o'
        for ipol in range(self.nPol):
            iplt=ipol+1
            dO=self.pars[0,:,:,ipol,0].transpose((1,0))
            pl.subplot(2,self.nPol,iplt)
            pl.plot(dO,fmt)
            if iplt%4==1:
                pl.ylabel('Orientation Error')
            pl.title(polname[ipol])

            dE=self.pars[0,:,:,ipol,1].transpose((1,0))
            pl.subplot(2,self.nPol,iplt+self.nPol)
            pl.plot(dE,fmt)
            if iplt%4==1:
                pl.ylabel('Ellipticity Error')
            pl.xlabel('Channel')

        
        
class DJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.D,
                         calTypeName='DJones',
                         parType=ParTypeEnum.COMPLEX,
                         nPol=2,
                         nPar=1,
                         matType=MatTypeEnum.GENERAL,
                         chanDepPar=True,
                         chanDepMat=True,
                         initparval=[0j],
                         verbose=verbose)
        
    def simPar(self):
        self.pars=np.random.normal(0.0,0.025,(self.nTimePar,
                                              self.nAntPar,
                                              self.nChanPar,
                                              self.nPol,
                                              self.nPar))
        
    def calcJones(self):
        super().initJones()
        #
        # [ 1+0j   p[0,0] ]
        # [ p[1,0] 1+0j   ]
        #
        self.Jones[:,:,:,[0,1],[0,1]]=1+0j       # diag=1
        self.Jones[:,:,:,[0,1],[1,0]]=self.pars[:,:,:,[0,1],0] # off-diags=pars


class CJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.C,
                         calTypeName='CJones',
                         parType=ParTypeEnum.FLOAT,
                         nPol=1,
                         nPar=1,
                         matType=MatTypeEnum.GENERAL,
                         chanDepPar=False,
                         chanDepMat=False,
                         initparval=[0],      # nominal orientation is zero
                         verbose=verbose)

    def calcJones(self):
        super().initJones()
        c=np.cos(self.pars[:,:,:,0,0])
        s=np.sin(self.pars[:,:,:,0,0])

        # NP: Assume linears for now; need to add basis-sensitivity!
        #   (all VisJones should have basis info, even if not needed explicitly)
        
        # Linears:
        # [cos(pa) sin(pa) ]
        # [-sin(pa) cos(pa)]
        self.Jones[:,:,:,0,0]=self.Jones[:,:,:,1,1]=c
        self.Jones[:,:,:,0,1]=s
        self.Jones[:,:,:,1,0]=-s

        # Circulars
        # [exp(-i.p) 0       ] 
        # [0       exp(+i.p)]
        #self.Jones[:,:,:,0,0].real=self.Jones[:,:,:,1,1].real=c
        #self.Jones[:,:,:,0,0].imag=-s
        #self.Jones[:,:,:,1,1].imag=s
        


        
class PJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.P,
                         calTypeName='PJones',
                         parType=ParTypeEnum.FLOAT,
                         nPol=1,
                         nPar=1,
                         matType=MatTypeEnum.GENERAL,
                         chanDepPar=False,
                         chanDepMat=False,
                         initparval=[0],
                         verbose=verbose)

    def simPar(self):
        self.initPar()
        # assume 60 deg linear increase over specified time range, for all antennas
        pa=np.linspace(0,60,self.nTimePar).reshape((self.nTimePar,1,1,1,1))*pi/180.
        self.pars=self.pars+pa
        

    def calcJones(self):
        super().initJones()
        c=np.cos(self.pars[:,:,:,0,0])
        s=np.sin(self.pars[:,:,:,0,0])

        # NP: Assume linears for now; need to add basis-sensitivity!
        #   (all VisJones should have basis info, even if not needed explicitly)
        
        # Linears:
        # [cos(pa) sin(pa) ]
        # [-sin(pa) cos(pa)]
        self.Jones[:,:,:,0,0]=self.Jones[:,:,:,1,1]=c
        self.Jones[:,:,:,0,1]=s
        self.Jones[:,:,:,1,0]=-s

        # Circulars
        # [exp(-i.p) 0       ] 
        # [0       exp(+i.p)]
        #self.Jones[:,:,:,0,0].real=self.Jones[:,:,:,1,1].real=c
        #self.Jones[:,:,:,0,0].imag=-s
        #self.Jones[:,:,:,1,1].imag=s
        


class TJones(VisJones):
    def __init__(self,verbose=VisJones.globverbose):
        super().__init__(visEqType=StandardVisEqTypeEnum.T,
                         calTypeName='TJones',
                         parType=ParTypeEnum.COMPLEX,
                         nPol=1,
                         nPar=1,
                         matType=MatTypeEnum.SCALAR,
                         chanDepPar=False,
                         chanDepMat=False,
                         initparval=[1+0j],
                         verbose=verbose)
    def simPar(self):
        self.initPar()
        p=np.random.normal(0.0,5.0,self.parShape)
        p[0,]=0.0
        p=np.cumsum(p,0)
        p=p+np.random.uniform(-180,180,(1,self.nAntPar,1,1,1))
        p*=(pi/180)
        self.pars.real=np.cos(p)
        self.pars.imag=np.sin(p)
        

    def calcJones(self):
        super().initJones()
        #
        # [ p[0,0] 0      ]
        # [ 0      p[0,0] ]
        #
        self.Jones[:,:,:,[0,1],[0,1]]=self.pars[:,:,:,[0,0],[0,0]]




# TBD:
# o Factory to form type-specific VisCals from filename (w/ type keyword)
# o Explicit accounting of VisJones roles?  (sim, solve, apply, etc.)
        



class VisData(object):
    def __init__(self,nTime,nAnt,nChan,polBasis=PolBasisEnum.LINEAR):
        self.nTime=nTime
        self.nAnt=nAnt
        self.nChan=nChan
        self.polBasis=polBasis
        self.nBln=None   # not None if simulated (or otherwise filled)
        self.doAC=None
        self.a1=[]
        self.a2=[]
        self.S=np.array([])
        self.M=np.array([])
        self.Mcor=np.array([])
        self.Vobs=np.array([])
        self.Vcor=np.array([])
        
    def info(self,showAve=True):
        print('VisData summary:')
        print(' nTime='+str(self.nTime),
              'nAnt='+str(self.nAnt),
              'nBln='+str(self.nBln),
              'nChan='+str(self.nChan),
              'polBasis='+self.polBasis.name,
              'doAC='+str(self.doAC))
        nn=self.nBln
        ell=''
        if len(self.a1)>100:
            nn=100
            ell='...'
        print(' a1='+str(self.a1[:nn])+ell)
        print(' a2='+str(self.a2[:nn])+ell)
        print(' S='+str(self.S))
        print(' M.shape='+str(self.M.shape),
              'Mcor.shape='+str(self.Mcor.shape),
              'Vobs.shape='+str(self.Vobs.shape),
              'Vcor.shape='+str(self.Vcor.shape))

        if showAve:
            if len(self.Vobs)>0:
                print(' mean(Vobs)='+str(np.mean(self.Vobs,(0,1,2))))
            if len(self.Vcor)>0:
                print(' mean(Vcor)='+str(np.mean(self.Vcor,(0,1,2))))

    def plotSpec(self,V,sym='.',clear=True):
        if clear:
            pl.clf()
        corrname=['XX','XY','YX','YY']
        print('Plotting '+str(V.shape[2])+' channels on '+str(V.shape[1])+' baselines...')
        Amax=pl.absolute(V).max()*1.05
        for ipol in range(4):
            iplt=ipol+1
            Vpl=V[0,:,:,ipol].transpose((1,0))
            pl.subplot(2,4,iplt)
            pl.plot(pl.absolute(Vpl),sym)
            ax=list(pl.axis())
            ax[2:]=[0.0,Amax]
            pl.axis(ax)
            pl.title(corrname[ipol])
            if iplt%4==1:
                pl.ylabel('Visibility Amplitude')
            pl.subplot(2,4,iplt+4)
            pl.plot(pl.angle(Vpl)*180/pi,sym)
            ax=list(pl.axis())
            ax[2:]=[-181,181]
            pl.xlabel('Channel')
            if iplt%4==1:
                pl.ylabel('Visibility Phase (deg)')

    def plotArgand(self,V,sym='.',doAve=False,clear=True):
        # TBD: organize colors better (by baseline?)
        if clear:
            pl.clf()
        for ipol in range(4):
            Vpl=V[0,:,:,ipol].transpose((1,0))
            pl.plot(Vpl.real,Vpl.imag,sym)
            if doAve:
                VplM=np.mean(Vpl)
                pl.plot(VplM.real,VplM.imag,'ko')
                
        pl.axis('equal')
        Amax=pl.absolute(pl.array((pl.axis()))).max()
        pl.axis([-Amax,Amax,-Amax,Amax])
        pl.xlabel('Real')
        pl.ylabel('Imag')
        pl.title('Complex Vis')
    
        
    # set point source model from specified Stokes parameters, according to polBasis
    #  (move to VisEquation?  Generalize to FT(I) cases....)
    def setPointModel(self,S=[1,0,0,0]):
        self.S=S
        self.M=np.zeros(shape=(1,1,1,4),dtype=complex)
        M=self.M  # reference
        if self.polBasis==PolBasisEnum.LINEAR:
            M[:,:,:,0]=S[0]+S[1]
            M[:,:,:,1]=S[2]+S[3]*1j
            M[:,:,:,2]=S[2]-S[3]*1j
            M[:,:,:,3]=S[0]-S[1]
        elif self.polBasis==PolBasisEnum.CIRCULAR:
            M[:,:,:,0]=S[0]+S[3]
            M[:,:,:,1]=S[1]+S[2]*1j
            M[:,:,:,2]=S[1]-S[2]*1j
            M[:,:,:,3]=S[0]-S[3]
            

    # Add noise to visibilities
    def addNoise(self,V,sig=0.01):
        V.real+=np.random.normal(0.0,sig,V.shape)
        V.imag+=np.random.normal(0.0,sig,V.shape)

    # Simulate observation according to supplied VisEq, generating meta-data and visibilities
    def observe(self,visEq,S=[1,0,0,0],sig=0.0,doAC=False):
        # inflate meta-info
        self.doAC=doAC
        self.times=range(self.nTime)   # for now, just indices
        self.chans=range(self.nChan)   # for now, just indices
        self.nBln=self.nAnt*(self.nAnt-1)//2  # XCs

        # Handle ACs
        noac=1  # offset for a1, a2 calculation
        if doAC:
            self.nBln+=self.nAnt  # add AC baselines
            noac=0
            
        # generate ordered first and second antennas lists for baselines
        self.a1=sum( list( map(lambda i:[i]*(self.nAnt-i-noac),        range(self.nAnt-noac)) ), [])
        self.a2=sum( list( map(lambda i:list(range(i+noac,self.nAnt)), range(self.nAnt)   ) ), [])

        # Set the visibility model data:
        self.setPointModel(S)
        
        # Initialize fully-inflated (local) Vobs with model
        Vobs=np.ones(shape=(self.nTime,self.nBln,self.nChan,4),dtype=complex)*self.M

        # Corrupt perfect vis via the VisEquation
        visEq.showCorruptVE()
        visEq.corrupt(self,Vobs)

        # Add some noise
        if sig>0.0:
            self.addNoise(Vobs,sig=sig)

        # Assign simulated visibilities to Vobs storage
        self.Vobs=Vobs

    # Correct Vobs, store in Vcor (this is applycal)
    #  TBD: remove to standalone function: applycal
    def correct(self,VisEq):
        assert (len(self.Vobs)!=0), 'No Vobs to correct!'

        # Report correcting VE
        VisEq.showCorrectVE()
        
        # deep copy of Vobs to Vcor
        Vcor=self.Vobs.copy()
        VisEq.correct(self,Vcor)
        self.Vcor=Vcor

    # Solve for calibration (for now, just form Vcor and Mcor
    #  TBD: remove to standalone function
    def solve(self,VisEq):
        assert (len(self.Vobs)!=0), 'No Vobs to correct!'
        assert (len(self.M)!=0), 'No M to corrupt!'

        # Report solving VE
        VisEq.showSolveVE()
        
        # deep copy of Vobs to Vcor, M to Mcor
        Vcor=self.Vobs.copy()
        Mcor=np.ones(shape=(self.nTime,self.nBln,self.nChan,4),dtype=complex)*self.M
        
        VisEq.contractVE(self,Mcor,Vcor)
        self.Mcor=Mcor
        self.Vcor=Vcor

        # pass to solve:
        print('Here we would pass this VisData to a solver for '+VisEq.solveVJ.calTypeName)

    # Form Stokes parameters from (for now) Linear Feed data
    def Stokes(self,Vin):
        # This assert isn't right, since one could pass Vin from a different VisData;
        #  need a better interface for selecting data members....
        assert (self.polBasis==PolBasisEnum.LINEAR), 'Pol Basis not linear!'
        M=np.linalg.inv(np.array([1,1,0,0,0,0,1,1j,0,0,1,-1j,1,-1,0,0]).reshape((4,4))).reshape((1,1,1,4,4))
        # convert to Stokes and average over time, baseline, channel axes
        S=np.real(np.mean(np.matmul(M,Vin.reshape(Vin.shape+(1,))),(0,1,2,4)))

        return S
    
        


class VisEquation(object):
    def __init__(self):
        self.applyVJs=[]
        self.upstreamVJs=[]
        self.downstreamVJs=[]
        self.solveVJ=None
        self.solvePivot=None

    # value function for sorting cal terms
    def __visEqSortVal__(self,vc):
        return vc.visEqType.value
        
    # add a VisJones to the apply list
    def setApply(self,applyVJ=None):
        if self.solveVJ!=None:
            print('Clearing solve term from VisEquation')
            self.solveVJ=None
            self.solvePivot=None
        
        # Add new term (if any) to apply list
        if applyVJ!=None:
            print('Arranging VE to apply: '+
                  applyVJ.visEqType.name+' '+
                  applyVJ.calTypeName+' '+
                  applyVJ.calFileName)
                  
            self.applyVJs.append(applyVJ)

        # If there is at least one term
        if len(self.applyVJs)>0:

            # sort the list according to StandardVisEqTypeEnum
            self.applyVJs.sort(key=self.__visEqSortVal__)

            # Set unique VE index in each VJ
            for ivj in range(len(self.applyVJs)):
                self.applyVJs[ivj].VEindex=ivj
        
        # set up/downstream lists to whole list (same)
        #  (these now appropriate for full corrupt or full correct)
        self.upstreamVJs=self.applyVJs        
        self.downstreamVJs=self.applyVJs      

    # arrange to solve for a VisJones
    def setSolve(self,solveVJ):
        self.solveVJ=solveVJ

        if len(self.applyVJs)>0:
            pivot=0
            more=True
            while more and self.applyVJs[pivot].visEqType.value<self.solveVJ.visEqType.value:
                pivot+=1
                more = pivot < len(self.applyVJs)
            self.solvePivot=pivot

            self.upstreamVJs=self.applyVJs[self.solvePivot:]
            self.downstreamVJs=self.applyVJs[0:self.solvePivot]

        else:
            self.solvePivot=-1

        print('Arranging VE to solve for '+
              self.solveVJ.calTypeName+' ('+
              self.solveVJ.visEqType.name+')')

        

    def showSolveVE(self):
        print('Solving ( ',end='')
        for idown in range(self.solvePivot-1,-1,-1):
            print(self.applyVJs[idown].visEqType.name+
                  '['+str(self.applyVJs[idown].VEindex)+']'+"'.",end='')
        print ('Vobs ) = ',end='')
        print(self.solveVJ.visEqType.name+' ( ',end='')
        if self.solvePivot>-1:
            for iup in range(self.solvePivot,len(self.applyVJs)):
                print(self.applyVJs[iup].visEqType.name+
                      '['+str(self.applyVJs[iup].VEindex)+']'+".",end='')
        print('Vmod )')
        print('...for '+
              self.solveVJ.calTypeName+' ('+
              self.solveVJ.visEqType.name+')')
        print('Pre-applying: ')
        for iapply in range(len(self.applyVJs)):
            print('    '+
                  self.applyVJs[iapply].visEqType.name+
                  '['+str(self.applyVJs[iapply].VEindex)+'] '+
                  self.applyVJs[iapply].calTypeName+' '+
                  self.applyVJs[iapply].calFileName)

        
    def showCorruptVE(self):
        print('Corrupting VE:  Vobs = ',end='')
        for idown in range(len(self.applyVJs)):
            print(self.applyVJs[idown].visEqType.name+
                  '['+str(self.applyVJs[idown].VEindex)+']'+".",end='')
        print ('Vtrue ')
        print(' Applying: ')
        for iapply in range(len(self.applyVJs)-1,-1,-1):
            print('    '+
                  self.applyVJs[iapply].visEqType.name+
                  '['+str(self.applyVJs[iapply].VEindex)+'] '+
                  self.applyVJs[iapply].calTypeName+' '+
                  self.applyVJs[iapply].calFileName)

    def showCorrectVE(self):
        print('Correcting VE:  Vcorr = ',end='')
        for idown in range(len(self.applyVJs)-1,-1,-1):
            print(self.applyVJs[idown].visEqType.name+
                  '['+str(self.applyVJs[idown].VEindex)+']'+"'.",end='')
        print ('Vobs')
        print(' Applying: ')
        for iapply in range(len(self.applyVJs)):
            print('    '+
                  self.applyVJs[iapply].visEqType.name+
                  '['+str(self.applyVJs[iapply].VEindex)+'] '+
                  self.applyVJs[iapply].calTypeName+' '+
                  self.applyVJs[iapply].calFileName)


    # Contract the supplied list of VisJones into a single general JJones
    def contractVJ(self,VJlist,visData,invert=False):
        Jagg=JJones(verbose=False)
        Jagg.setParShapeFromVisData(visData,False)
        Jagg.initJones()
        for ivc in VJlist:
            #print(' Accumulating: '+
            #      ivc.visEqType.name+'['+str(ivc.VEindex)+'] '+
            #      ivc.calTypeName+' '+
            #      ivc.calFileName)
            ivc.resampleInFreq(visData)
            ivc.resampleInTime(visData)
            ivc.calcJones()
            Jagg.accumulate(ivc)
        if invert:
            Jagg.invertJones()
        return Jagg

            
    # corrupt specified visibilities with the upstreamVJs list (if any)
    def corrupt(self,visData,V):
        if len(self.upstreamVJs)>0:
            Jup=self.contractVJ(self.upstreamVJs,visData,False)  # no inverse
            Jup.applyRight(visData,V)
            Jup.applyLeft(visData,V)

            
    # correct specified visibilities with the inverted dosnstreamVJ list (if any)
    def correct(self,visData,V):
        if len(self.downstreamVJs)>0:
            Jdown=self.contractVJ(self.downstreamVJs,visData,True)  # invert!
            Jdown.applyRight(visData,V)
            Jdown.applyLeft(visData,V)

    # form corrected and corrupted visibilites for solving
    def contractVE(self,visData,Vup,Vdown):
        self.corrupt(visData,Vup)
        self.correct(visData,Vdown)


    # Traditional polarization refactor: B(G)J --> B'(G)D
    #  While the polarizer (general J) and backend filter (diagonal B) separately operate on the 
    #   incoming signal in a freq-dep way (i.e., both as a "bandpass" effect), practical calibration
    #   heuristics have effectively refactored these terms so that the effective B calibration
    #   contains both the backend filter's and polarizer's like-polarization (i.e., on-diagonal Jones)
    #   effects in a single diagonal Jones matrix (NB: the diagonal factorization of J commutes with G).
    #   Separately, the cross-polarization (i.e., off-diagonal) effects remain characterized by a
    #   "D-term" (and typically, only if polarimetry is relevant) with ones on its diagonal.  Note
    #   that the off-diagonal D-term elements are normalized by the on-diagonal elements of the
    #   original polarizer's J matrix.
    #
    #    BJ = [Bp 0][Jpp Jpq] = [BpJpp  0][1  Jpq/Jpp] = [Bp'  0][1  Dpq]  = B'D
    #         [0 Bq][Jqp Jqq]   [0  BqJqq][Jqp/Jqq  1]   [0  Bq'][Dqp  1]
    def refactorPolTraditional(self,B,J):
        Jagg=JJones(verbose=False)
        Jagg.setParShape(B.nTimePar,B.nAntPar,B.nChanPar)
        Jagg.initJones()
        for ivj in [B,J]:
            ivj.calcJones()
            Jagg.accumulate(ivj)

        # Extract Bout's pars from Jagg
        Bout=BJones()
        Bout.setParShape(B.nTimePar,B.nAntPar,B.nChanPar)
        Bout.initPar()
        Bout.pars=Jagg.Jones[:,:,:,[0,1],[0,1]].reshape((B.nTimePar,B.nAntPar,B.nChanPar,2,1))
        Bout.calcJones()
        Bout.invertJones()  # temporarily, to form Dout below

        # Extract Dout's pars from Jagg
        Dout=DJones()
        Dout.setParShape(B.nTimePar,B.nAntPar,B.nChanPar)
        Dout.initPar()
        Dout.Jones=Jagg.Jones.copy()
        Dout.accumulate(Bout,fromRight=False)
        Dout.pars=Dout.Jones[:,:,:,[0,1],[1,0]].reshape((B.nTimePar,B.nAntPar,B.nChanPar,2,1))

        # Clear Jones member; Users of Bout, Dout will have to redo calcJones (if needed)
        Dout.clearJones()
        Bout.clearJones()

        return Bout,Dout

    # Smirnov polarization refactor  B(G)J -> (G)J'
    #  Smirnov (2010) proposes an ~economical refactoring of the standard and traditional Vis equation
    #   into strictly time-dependent and strictly frequency-dependent parts:  Since backend filter, B,
    #   and polarizer, J, are both frequency-dependent, it may be most efficient just to combine these
    #   and solve for them as a single term (NB: B commutes with G)
    #
    #    BJ = [Bp 0][Jpp Jpq] = [BpJpp  BpJpq] = J'
    #         [0 Bq][Jqp Jqq]   [BqJqp  BqJqq]
    def refactorPolSmirnov(self,B,J):
        Jout=JJones(verbose=False)
        Jout.setParShape(B.nTimePar,B.nAntPar,B.nChanPar)
        Jout.initPar()
        Jout.initJones()
        for ivj in [B,J]:
            ivj.calcJones()
            Jout.accumulate(ivj)

            
        Jout.pars[:,:,:,[0,0,1,1],[0,1,0,1]] = Jout.Jones[:,:,:,[0,0,1,1],[0,1,1,0]]
        Jout.clearJones()

        return Jout






# ...

