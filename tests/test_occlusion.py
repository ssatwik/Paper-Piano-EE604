import numpy as np
from config import Config
from occlusion import OcclusionScorer

def test_ncc_search_and_brightness_invariance():
    cfg=Config(search_margin=4,use_search_window=True)
    scorer=OcclusionScorer(cfg)
    image=np.full((80,100),220,np.uint8); image[35:45,45:55]=30
    scorer.initialize(image,[[50,40]],[(45,35,10,10)])
    clean,_=scorer.score(image)
    shifted=np.full_like(image,220); shifted[35:45,48:58]=30
    moved,_=scorer.score(shifted)
    bright=np.clip(image*.6+30,0,255).astype(np.uint8); lighting,_=scorer.score(bright)
    assert clean[0]>.9 and moved[0]>.8 and abs(clean[0]-lighting[0])<.1

def test_variance_guard():
    s=OcclusionScorer(Config()); s.initialize(np.full((40,40),100,np.uint8),[[20,20]],[(18,18,4,4)])
    assert s.unreliable==[True]
