from collections import deque
import numpy as np
from config import Config
from touch import TouchFusion
from hands import Fingertip

def test_fusion_debounces_on_and_off():
    cfg=Config(n_on=2,n_off=2,use_kinematics=False,grace_hand_lost=.05)
    f=TouchFusion(cfg); dot=np.array([[10.,10.]])
    tip=Fingertip('Left',1,np.array([10.,10.]),np.array([10.,20.]),20.,hist=deque([(0,1.),(.1,1.)]))
    ev,_=f.update(0,[.3],dot,[tip]); assert not ev
    ev,_=f.update(.03,[.3],dot,[tip]); assert [e.type for e in ev]==['on']
    ev,_=f.update(.06,[.9],dot,[]); assert not ev
    ev,_=f.update(.09,[.9],dot,[]); assert [e.type for e in ev]==['off']

def test_velocity_monotone_and_clipped():
    cfg=Config(use_kinematics=False)
    f=TouchFusion(cfg)
    slow=Fingertip('L',0,np.zeros(2),np.ones(2),10,hist=deque([(1,0.0)]))
    fast=Fingertip('L',0,np.zeros(2),np.ones(2),10,hist=deque([(1,5.0)]))
    assert f._velocity(fast,1)>=f._velocity(slow,1)
    assert 1<=f._velocity(fast,1)<=127

def test_wta_can_be_ablated_and_sweep_gate_blocks_fast_motion():
    dots=np.array([[10.,10.],[20.,10.]])
    finger=Fingertip('Left',1,np.array([15.,10.]),np.array([15.,20.]),40.,speed_norm=0.)
    results=[]
    for wta in (True,False):
        cfg=Config(n_on=1,use_wta=wta,use_kinematics=False)
        fusion=TouchFusion(cfg)
        events,_=fusion.update(0,[.2,.2],dots,[finger])
        results.append(len(events))
    assert results==[1,2]
    cfg=Config(n_on=1,use_kinematics=True,v_sweep=2.0)
    fusion=TouchFusion(cfg)
    fast=Fingertip('Left',1,finger.p,finger.dip,40.,speed_norm=3.0)
    events,_=fusion.update(0,[.2,.2],dots,[fast])
    assert events==[]

def test_held_key_is_released_when_tracker_is_lost():
    cfg=Config(n_on=1,use_kinematics=False)
    fusion=TouchFusion(cfg); dots=np.array([[10.,10.]])
    finger=Fingertip('Left',1,np.array([10.,10.]),np.array([10.,20.]),20.)
    on,_=fusion.update(0,[.2],dots,[finger]); assert on[0].type=='on'
    off,_=fusion.update(.03,[.2],dots,[finger],track_status='lost')
    assert [event.type for event in off]==['off']
