"""What a fixed snapshot must render to, and the few rules worth stating twice.

The pages are the product, so the golden comparison carries most of the weight:
one fixed snapshot in, three committed pages out, byte for byte. A unit test
earns its place here only where the behaviour is a rule the pages do not show —
a template rejected, a package deciding where an op is published, an expression
the evaluator must refuse.
"""
import ast
import os
import subprocess
import sys
from collections import Counter

import gen_bench_pages as g
import pytest
import workload_shape as ws

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")
GOLDEN = os.path.join(REPO, "tests", "golden")

# The method note is prose written in the renderer, with nothing read out of the
# snapshot: holding it byte for byte would only mean refreshing a golden file
# whenever a sentence is edited, and the sentence is already in the diff.
PROSE = {"reading.md"}

# The default locale writes `<slug>.md`; every other one writes `<slug>.<loc>.md`
# beside it, which is the layout mkdocs-static-i18n reads. Only the default
# locale is held byte for byte — a translation is copy, and a golden for it would
# be a second copy of the locale table.
LOCALE_SUFFIXES = tuple(suf for lang, suf in g.LANG_SUFFIX.items()
                        if lang != g.DEFAULT_LANG)


def default_locale(pages: dict) -> dict:
    return {n: t for n, t in pages.items() if not n.endswith(LOCALE_SUFFIXES)}


def render(out_dir: str, manifest_dir: str = os.path.join(FIXTURES, "manifest"),
           bench: str = os.path.join(FIXTURES, "bench_results.xml"),
           test_xml: str | None = os.path.join(FIXTURES, "test_results.xml")):
    """Run the renderer as the deploy runs it, and return what it wrote.

    The roofline tool is pointed at a directory that holds nothing, so the SOL
    column is the degraded one and the pages depend on this repository alone: a
    contributor with a TileOPs checkout beside them renders what CI renders.
    """
    cmd = [sys.executable, os.path.join(REPO, "scripts", "gen_bench_pages.py"),
           "--tileops", os.path.join(FIXTURES, "no-tileops"),
           "--bench-xml", bench,
           *(("--test-xml", test_xml) if test_xml else ()),
           "--meta", os.path.join(FIXTURES, "meta.json"),
           "--manifest-dir", manifest_dir,
           "--commit", "0123456789abcdef0123456789abcdef01234567",
           "--date", "2026-01-01", "--gpu", "NVIDIA H200", "--out-dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {n: open(os.path.join(out_dir, n), encoding="utf-8").read()
            for n in sorted(os.listdir(out_dir))}, proc.stderr


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    return render(str(tmp_path_factory.mktemp("bench")))


# The Ascend fork's snapshot schema: one comparison group per workload, written
# under the `baseline` tag, carrying the whole D036 candidate pool. Five cases,
# one per shape the renderer has to have a defined answer for.
D036_BENCH = os.path.join(FIXTURES, "bench_results_d036.xml")


@pytest.fixture(scope="module")
def d036(tmp_path_factory):
    pages, err = render(str(tmp_path_factory.mktemp("d036")),
                        bench=D036_BENCH, test_xml=None)
    return "".join(t for n, t in pages.items() if not n.endswith(".zh.md")), pages, err


_ROW_START = '<tr><td class="colsep">'


def _rows(text: str) -> list[str]:
    """The data rows of every table, one string each.

    Split on the row's own opening, not on `<tr>`: the two header rows and the
    prose between tables would otherwise land inside a row.
    """
    return [r.split("</tr>")[0] for r in text.split(_ROW_START)[1:]]


def _row(text: str, needle: str) -> str:
    """The one data row that mentions `needle`."""
    rows = [r for r in _rows(text) if needle in r]
    assert len(rows) == 1, f"{needle} matched {len(rows)} rows"
    return rows[0]


# --- The pages --------------------------------------------------------------

def test_pages_match_the_committed_output(rendered):
    pages = {n: t for n, t in default_locale(rendered[0]).items()
             if n not in PROSE}
    assert sorted(pages) == sorted(os.listdir(GOLDEN))
    for name, text in pages.items():
        expected = open(os.path.join(GOLDEN, name), encoding="utf-8").read()
        assert text == expected, (
            f"{name} changed. Read the diff — it is the change, stated in the "
            f"product — then run: python tests/refresh_golden.py")


# --- The locales -----------------------------------------------------------

def test_every_page_is_written_in_every_locale(rendered):
    """A page the renderer stops writing in one locale does not fail the build:
    i18n serves the default-language page in its place. So assert it here.
    """
    pages = rendered[0]
    stems = {n.removesuffix(".md") for n in default_locale(pages)}
    for lang, suffix in g.LANG_SUFFIX.items():
        got = {n for n in pages if n.endswith(suffix)
               and not n.endswith(tuple(s for s in LOCALE_SUFFIXES
                                        if s != suffix))}
        assert got == {s + suffix for s in stems}, f"{lang} is missing a page"


def test_every_locale_declares_the_same_keys():
    """A key dropped while translating would fall back silently."""
    keys = set(g.STRINGS[g.DEFAULT_LANG])
    for lang, table in g.STRINGS.items():
        assert set(table) == keys, f"{lang} declares a different key set"


def test_the_locale_table_carries_no_dead_copy():
    """Every key is read by a page builder. A key nothing reads is copy someone
    would translate for a page that never shows it.
    """
    src = open(os.path.join(REPO, "scripts", "gen_bench_pages.py"),
               encoding="utf-8").read()
    # Count parsed literals, so single and double quoting are equivalent.
    uses = Counter(n.value for n in ast.walk(ast.parse(src))
                   if isinstance(n, ast.Constant) and isinstance(n.value, str))
    unused = sorted(k for k in g.STRINGS[g.DEFAULT_LANG]
                    if uses[k] < len(g.STRINGS) + 1)
    assert not unused, f"declared but never rendered: {unused}"


def test_an_untranslated_locale_renders_the_default_text(rendered):
    """While a locale's values are still the English placeholders, its pages are
    the English pages. This holds until the translation lands, and then it is
    the translation that breaks it — delete the assertion, not the translation.
    """
    pages = rendered[0]
    for lang, suffix in g.LANG_SUFFIX.items():
        if any(v != g.STRINGS[g.DEFAULT_LANG][k]
               for k, v in g.STRINGS[lang].items()):
            continue
        for name, text in default_locale(pages).items():
            assert pages[name.removesuffix(".md") + suffix] == text


# --- The rules the pages do not show ---------------------------------------

def test_rendering_is_deterministic(tmp_path):
    # Two runs over one snapshot, so an ordering that depends on a set or on
    # dict iteration shows up here rather than as a diff on the deployed site.
    assert render(str(tmp_path / "a"))[0] == render(str(tmp_path / "b"))[0]


def test_a_run_reports_what_it_could_not_describe(rendered):
    stderr = rendered[1]
    assert "MysteryFwdOp" in stderr              # no manifest entry
    assert "recorded ratio disagrees" in stderr  # ratio against the times
    assert "brand-new-lib" in stderr             # a baseline tag with no tier


def test_without_a_manifest_the_pages_still_render(tmp_path):
    pages, _ = render(str(tmp_path / "bare"), manifest_dir=str(tmp_path / "none"))
    assert pages, "a missing manifest must not stop the deploy"
    text = "".join(pages.values())
    # A workload the manifest described is now named by its benchmark id alone:
    # nothing declares its shapes any more.
    assert "decode-b1-h8-bfloat16" in text
    assert "[B, H, DK]" not in text
    # A workload whose shape the snapshot recorded itself does not depend on the
    # manifest and still states it. Only what the signature added is lost:
    # `is_causal` is no longer known to be a parameter, so it prints as one more
    # scalar rather than a dimmed one.
    assert "<span class=\"wl-k\">x</span>: [1, 4097, 128]" in text
    assert "probe-tail" in text


# --- Rules the pages do not show -------------------------------------------

def test_a_template_may_not_bind_one_symbol_to_two_values():
    # `[D, D]` says the two dimensions are equal. This shape says they are not,
    # so the template is rejected and the row prints its concrete shape.
    assert ws._bind("[D, D]", [64, 32]) is None
    assert ws._bind("[D, D]", [64, 64]) == (["D", "D"], {"D": 64})


def test_a_template_is_not_executed():
    # Templates are parsed, not run: only integer arithmetic over the names a
    # workload sets resolves, and anything else leaves the tensor undescribed.
    assert ws._eval_template("[max(a, b)]", {"a": 1, "b": 2}) is None
    assert ws._eval_template("[n * 2, k]", {"n": 4, "k": 3}) == [8, 3]


def test_the_package_decides_the_family_not_a_word_in_the_name():
    # `linear` matches `linear_attention` as a substring, so the package has to
    # win: otherwise a linear-attention op is published on the GEMM page.
    assert g.family_of("DeltaDecodeFwdOp",
                       "tileops.ops.linear_attention.delta") == "linear_attention"
    assert g.family_of("GemmOp", "tileops.ops.gemm.gemm") == "linear_algebra"
    # An op defined in a module rather than a package still falls through.
    assert g.family_of("RmsNormFwdOp", None) == "normalization"


# --- The D036 comparison group ---------------------------------------------
# Everything here is a rule about *what the reader is told they are compared
# against*. The page got this wrong for a whole run of nightly snapshots -- our
# own provider was printed in the Alternatives column and the baseline was
# dropped as a duplicate -- so each rule is stated once, on its own.

def test_our_own_provider_is_not_listed_as_an_alternative(d036):
    text = d036[0]
    # `tilelang` is the implementation under test. It reached the page as its
    # own opponent while `Device time` sat empty, which is the defect this
    # asserts against.
    assert "<code>tilelang</code>" not in text
    # Our own time, in the Device time column: the third cell of the row.
    assert _row(text, "catlass-r252:r252_gemm").split("<td>")[2] == "0.9406</td>"


def test_every_pool_member_is_shown_with_its_tier(d036):
    row = _row(d036[0], "catlass-r252:r252_gemm")
    for name in ("catlass-r252:r252_gemm", "ops-nn-gemm:aclnnMm",
                 "torch_npu eager", "catlass:catlass_gemm_fp16"):
        assert f"<code>{name}</code>" in row, name
    assert row.count('class="tier tier-hw"') == 2
    assert row.count('class="tier tier-vendor"') == 2


def test_the_winner_of_the_pool_is_the_one_marked(d036):
    # The pool's fastest member is what `Ratio` divided by, so it is the line
    # that carries the mark and the only line not dimmed.
    row = _row(d036[0], "catlass-r252:r252_gemm")
    head, mark = row.split("alt-pick")[0], row.split("alt-pick")[1]
    assert "catlass-r252:r252_gemm" in head
    assert "ops-nn-gemm:aclnnMm" in mark  # the runner-up comes after the mark
    assert row.count("alt-pick") == 1


def test_a_tier_badge_links_to_what_a_tier_means(d036):
    # A reader meeting `handwritten` beside a slower time than `vendor` has to
    # be able to reach the sentence saying a tier is not a speed ranking.
    row = _row(d036[0], "ops-nn:aclnnRmsNorm")
    assert f'href="../reading/#{g.PROV_ANCHOR}"' in row
    reading = d036[1]["reading.md"]
    assert "{#" + g.PROV_ANCHOR + "}" in reading


def test_a_losing_candidate_retains_its_time_without_an_extra_headline_ratio(d036):
    # The renderer now publishes one ratio in the headline measurement regime.
    # Pool times stay visible without substituting an eager ratio for it.
    row = _row(d036[0], "ops-nn-ew:aclnnForeachMulList")
    assert "1.01×" in row
    assert "0.0565" in row and "0.0517" in row
    assert "open-src 0.96×" not in row
    # And is absent where the open-source candidate won the pool: there it
    # would be the same number twice.
    assert "hw " not in _row(d036[0], "ops-nn:aclnnRmsNorm")
    assert "hw " not in _row(d036[0], "catlass-r252:r252_gemm")


def test_an_untimed_candidate_never_renders_as_zero(d036):
    # `not-timed` is the harness recording that it never measured this
    # candidate. Rendered as 0 it would read as an infinitely fast opponent.
    (row,) = [r for r in _rows(d036[0]) if "alt-untimed" in r]
    # The baseline latency the run did publish, marked as the stand-in it is.
    assert "0.0602*" in row
    assert ">0<" not in row and "0.0000" not in row
    # And the row is still rated: the comparison happened, only the pool line
    # is missing a time.
    assert "1.25×" in row


def test_a_template_instantiation_cannot_break_the_table(d036):
    text = d036[0]
    # `<`, `>` and `|` arrive unescaped from the XML parser. Escaped, or they
    # close the cell they are in.
    assert "&lt;BasicMatmul&lt;half|bfloat16_t" in text
    assert "<BasicMatmul<" not in text
    # The full binding stays reachable; only its head is printed.
    assert 'title="catlass-r252:r252_gemm -&gt; Catlass::Gemm' in text


def test_the_tier_subhead_appears_only_where_the_snapshot_has_tiers(d036, rendered):
    assert g.STRINGS["en"]["table.sub.alt_name_tiered"] in d036[0]
    # The upstream snapshot publishes no tiers, so promising one would be a
    # header the rows cannot honour.
    upstream = "".join(default_locale(rendered[0]).values())
    assert g.STRINGS["en"]["table.sub.alt_name_tiered"] not in upstream


def test_the_index_reports_both_coverage_readings_over_one_denominator(d036):
    index = d036[1]["index.md"]
    # Three ops, all three with an open-source candidate somewhere in a pool.
    assert "**3 of 3 ops** are rated against a real alternative" in index
    assert "**3 of 3 ops** have a tier-1 open-source baseline" in index
    assert "**3 ops**" in index and "5 workloads" in index  # explicit denominator


# --- The rules of the pool parser ------------------------------------------

def test_a_pool_entry_splits_on_its_last_equals_sign():
    pool = ("catlass:g -> DeviceGemm<BasicMatmul<half|bf16, R x R|C>>"
            "=398.375us (handwritten); torch_npu eager (vendor)"
            "=408.250us (vendor)")
    got = g.parse_pool(pool)
    assert [c["name"] for c in got] == [
        "catlass:g -> DeviceGemm<BasicMatmul<half|bf16, R x R|C>>",
        "torch_npu eager (vendor)"]
    assert [c["ms"] for c in got] == [0.398375, 0.40825]
    assert [c["prov"] for c in got] == ["handwritten", "vendor"]


def test_an_untimed_pool_entry_keeps_no_time_rather_than_zero():
    (c,) = g.parse_pool("torch_npu eager (vendor)=not-timed (vendor)")
    assert c["ms"] is None and c["raw_time"] == "not-timed"


def test_a_candidate_name_may_contain_the_separator():
    got = g.parse_pool("a=1.000us (vendor); ops:x; y=2.000us (handwritten)")
    assert [c["name"] for c in got] == ["a", "ops:x; y"]
    assert [c["ms"] for c in got] == [0.001, 0.002]


def test_an_unparseable_pool_loses_no_member():
    got = g.parse_pool("a=1.000us (vendor); something the parser cannot read")
    assert [c["name"] for c in got] == ["a", "something the parser cannot read"]
    assert got[1]["ms"] is None


def test_the_baseline_alias_is_dropped_only_when_it_duplicates_a_rival(tmp_path):
    # Upstream writes `baseline_*` as an alias for a baseline it also publishes
    # by name; the Ascend fork writes it as the only comparison group there is.
    both = tmp_path / "both.xml"
    both.write_text(
        '<testsuite><testcase name="X[a-float16]" classname="X">'
        '<property name="op" value="X" />'
        '<property name="tileops_device_busy_ms" value="1.0" />'
        '<property name="triton_device_busy_ms" value="2.0" />'
        '<property name="baseline_device_busy_ms" value="2.0" />'
        "</testcase></testsuite>", encoding="utf-8")
    (w,) = g.parse_bench_xml(str(both))[0]
    assert sorted(w["impls"]) == ["tileops", "triton"]
    alone = tmp_path / "alone.xml"
    alone.write_text(
        '<testsuite><testcase name="X[a-float16]" classname="X">'
        '<property name="op" value="X" />'
        '<property name="tilelang_latency_ms" value="1.0" />'
        '<property name="baseline_latency_ms" value="2.0" />'
        '<property name="baseline_name" value="torch_npu eager (vendor)" />'
        "</testcase></testsuite>", encoding="utf-8")
    (w,) = g.parse_bench_xml(str(alone))[0]
    assert sorted(w["impls"]) == ["baseline", "tilelang"]
    assert w["impls"]["baseline"]["name"] == "torch_npu eager (vendor)"


def test_the_two_d036_time_spellings_are_both_read(tmp_path):
    # The harness has written `_us` and `_latency_ms` for the runner-up and
    # open-source times at different points. A suffix table that let the bare
    # `latency_ms` claim them would file them under an invented tag.
    xml = tmp_path / "spellings.xml"
    xml.write_text(
        '<testsuite><testcase name="X[a-float16]" classname="X">'
        '<property name="op" value="X" />'
        '<property name="tilelang_latency_ms" value="1.0" />'
        '<property name="baseline_latency_ms" value="2.0" />'
        '<property name="baseline_handwritten_latency_ms" value="3.0" />'
        '<property name="baseline_runner_up_us" value="4000.0" />'
        "</testcase></testsuite>", encoding="utf-8")
    (w,) = g.parse_bench_xml(str(xml))[0]
    assert sorted(w["impls"]) == ["baseline", "tilelang"]
    base = w["impls"]["baseline"]
    assert base["handwritten_latency_ms"] == 3.0
    assert base["runner_up_us"] == 4000.0
    assert g.workload_metrics(w)["hw"]["ms"] == 3.0


def test_the_provider_under_test_is_found_whatever_it_is_called():
    # One run has one provider, and upstream's name for it wins where a
    # snapshot somehow carries both.
    assert g.ours_of({"tilelang": {"a": 1}}) == {"a": 1}
    assert g.ours_of({"tileops": {"a": 1}, "tilelang": {"a": 2}}) == {"a": 1}
    assert g.ours_of({"triton": {"a": 1}}) == {}


def test_every_locale_names_every_op_family():
    """A family missing from a locale renders its English name mid-page."""
    fams = set(g.FAMILY_TITLE["en"])
    for lang in g.LANG_SUFFIX:
        assert set(g.FAMILY_TITLE[lang]) == fams, f"{lang} is missing a family"
    assert fams == set(g.FAMILY_TITLE["en"]) >= {f for _, _, fs in g.DATA_PAGES
                                                 for f in fs}
