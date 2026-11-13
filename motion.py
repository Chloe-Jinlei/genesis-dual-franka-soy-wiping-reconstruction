"""Continuous, slow bimanual Cartesian paths; IK is warm-started along one branch."""
import numpy as np

def ease(u):
    u=np.clip(u,0.,1.)
    return u*u*u*(10+u*(-15+6*u))

# Symmetric half-width, centre y, TCP z, finger-joint opening.
KEYS=[
 (0.,(.34,-.04,1.37,.025),'initial'),
 (1.,(.34,-.04,1.37,.025),'initial'),
 (5.,(.135,-.23,.86,.025),'approach'),
 (7.,(.135,-.23,.7582,.025),'lower_to_edges'),
 (8.5,(.135,-.23,.7582,.001),'close_grippers'),
 (11.,(.129,-.23,.93,.001),'lift_cloth'),
 (14.,(.129,.04,.93,.001),'transport'),
 (16.,(.129,.04,.763,.001),'lay_on_spill'),
 (17.,(.129,-.045,.763,.001),'begin_wipe'),
]
GRASP_TIME=8.5
WIPE_END=57.  # at most eight cycles; the controller ends on a clean cycle boundary
END_TIME=62.

def target(t):
    if t<=17:
        for (t0,v0,_),(t1,v1,phase) in zip(KEYS,KEYS[1:]):
            if t<=t1:
                return (np.asarray(v0)*(1-ease((t-t0)/(t1-t0)))+
                        np.asarray(v1)*ease((t-t0)/(t1-t0))),phase
    if t<=WIPE_END:
        # Complete reciprocating strokes, C2 joins via smooth half-strokes.
        u=(t-17)/2.5
        k=min(int(u),round((WIPE_END-17)/2.5)-1)
        a,b=(-.045,.14) if k%2==0 else (.14,-.045)
        return np.array([.129,a+(b-a)*ease(u-k),.763,.001]),'wipe'
    if t<=WIPE_END+2:
        u=ease((t-WIPE_END)/2)
        return np.array([.129,-.045+.085*u,.763+.187*u,.001]),'final_lift'
    return np.array([.129,.04,.95,.001]),'hold_wet_cloth'

def positions(t):
    p,phase=target(t)
    return [np.array([-p[0],p[1],p[2]]),np.array([p[0],p[1],p[2]])],p[3],phase
