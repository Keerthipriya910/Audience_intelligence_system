from src.ewtc import apply_ewtc

def test_positive_agreement_raises_confidence():
    r=apply_ewtc("negative",0.58,["negative"]*4+["positive"],[0.1,0.2,0.3,0.4,0.9])
    assert r.final_confidence >= r.original_confidence
    assert abs(r.correction_strength - 0.42) < 1e-9

def test_no_evidence_is_identity():
    r=apply_ewtc("neutral",0.61,[],[])
    assert r.final_label=="neutral" and r.final_confidence==0.61
