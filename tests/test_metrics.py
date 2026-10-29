from bench.metrics import match_events, hold_errors

def test_metrics_matching_same_key_and_substitution():
    gt=[{"key":0,"t_on":1.0},{"key":1,"t_on":2.0}]
    pred=[{"type":"on","key":0,"t":1.1},{"type":"on","key":2,"t":2.05},{"type":"on","key":3,"t":4.0}]
    m=match_events(gt,pred,.15)
    assert (m.tp,m.fp,m.fn,m.substitutions)==(1,2,1,1)

def test_metrics_tolerance_edge_inclusive():
    m=match_events([{"key":0,"t_on":0.0}],[{"type":"on","key":0,"t":.15}],.15)
    assert m.tp==1 and m.fp==0 and m.fn==0

def test_hold_duration_error_matches_note_events():
    gt=[{"key":1,"t_on":1.0,"t_off":1.4}]
    pred=[{"type":"off","key":1,"t":1.6},{"type":"on","key":1,"t":1.1}]
    m=match_events(gt,pred,.2)
    assert abs(hold_errors(gt,pred,m)[0]-.1)<1e-9
