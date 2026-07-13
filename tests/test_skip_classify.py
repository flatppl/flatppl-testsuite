"""The importer's stderr classifier: which converter errors are clean
"not implemented" SKIPs vs hard failures. Invalid-document errors
(`unsupported HS3 construct:`) must NEVER be masked as skips."""

from flatppl_testsuite.formats.hs3.importer import classify_unimplemented


def test_unknown_dist_type_skips():
    s = classify_unimplemented(
        "flatppl: hs3: unsupported HS3 distribution type: fft_conv_pdf")
    assert s is not None
    assert s.hs3_type == "fft_conv_pdf"


def test_unknown_modifier_skips():
    s = classify_unimplemented(
        "flatppl: hs3: unsupported histfactory modifier: custom_mod")
    assert s is not None
    assert s.hs3_type == "custom_mod"


def test_unimplemented_construct_skips():
    s = classify_unimplemented(
        "flatppl: hs3: unimplemented HS3 construct: "
        "datum `d` carries per-event weights")
    assert s is not None
    assert s.detail  # full stderr preserved for the skip report


def test_invalid_document_hard_fails():
    assert classify_unimplemented(
        "flatppl: hs3: unsupported HS3 construct: duplicate binding name `x`"
    ) is None


def test_clean_stderr_is_none():
    assert classify_unimplemented("") is None
