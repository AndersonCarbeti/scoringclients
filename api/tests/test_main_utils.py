from app.main import _decision_from_proba, _extract_first


def test_decision_refused_when_above_threshold():
    decision = _decision_from_proba(0.9, 0.4)
    assert decision["predicted_class"] == 1
    assert decision["decision"] == "REFUSED"


def test_decision_approved_when_below_threshold():
    decision = _decision_from_proba(0.1, 0.4)
    assert decision["predicted_class"] == 0
    assert decision["decision"] == "APPROVED"


def test_extract_first_supports_nested_lists():
    value = _extract_first([[0.3, 0.7]])
    assert value == 0.7
