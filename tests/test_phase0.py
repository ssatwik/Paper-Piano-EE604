import numpy as np
from validation import TouchValidator
from piano_player import PianoPlayer

def test_validator_freeze_resets_and_edge_crops_match():
    frame=np.full((30,30,3),255,np.uint8); frame[2,2]=0
    v=TouchValidator(); v.initialize_points_images(frame,[(2.,2.)]); assert len(v._is_pressed)==1
    v._is_pressed[0]=True; v.initialize_points_images(frame,[(2.,2.)]); assert v._is_pressed==[False]

def test_middle_selection_clamps_and_handles_one_key():
    p=PianoPlayer.__new__(PianoPlayer); p._sound_filenames=[f'{i}.mp3' for i in range(4)]
    assert p._select_sounds_middle(1)==['2.mp3']
    assert len(p._select_sounds_middle(8))==4
