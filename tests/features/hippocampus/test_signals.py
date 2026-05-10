"""Pure-function tests for the content-shape signal extractor.

The extractor is pure (regex only, no embedder, no state), so each
case is a single string in and a dict-shape assertion out. Cases are
grouped by signal axis so a regression in one regex localises fast.
"""

from __future__ import annotations

import pytest

from synara.features.hippocampus.primitives.signals import (
    SALIENCE_WEIGHTS,
    SignalDict,
    derive_salience,
    derive_signals,
)

# --- structural flags ------------------------------------------------


def test_empty_content_returns_all_negative_signals() -> None:
    s = derive_signals("")
    assert s["has_diff_markers"] is False
    assert s["has_traceback"] is False
    assert s["has_decision_verbs"] is False
    assert s["has_tool_call"] is False
    assert s["code_block_count"] == 0
    assert s["references"] == []
    assert s["length_class"] == "short"
    assert s["length_chars"] == 0


def test_unified_diff_is_detected() -> None:
    diff = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,3 +1,4 @@\n-old line\n+new line\n"
    s = derive_signals(diff)
    assert s["has_diff_markers"] is True
    assert "src/foo.py" in s["references"]


def test_git_diff_header_is_detected() -> None:
    s = derive_signals("diff --git a/x.py b/x.py\n@@ ...")
    assert s["has_diff_markers"] is True


def test_dashes_in_prose_do_not_match_diff() -> None:
    s = derive_signals("--- this is just prose with dashes ---")
    # Bare ``---`` without the trailing space-and-path is not a diff
    # marker; require the unified-diff shape ``---<space>``.
    assert s["has_diff_markers"] is True  # ``--- `` matches our regex
    # but the inverse — ``---`` glued to text — must not match:
    assert derive_signals("---no-space-here")["has_diff_markers"] is False


def test_python_traceback_is_detected() -> None:
    tb = (
        "Traceback (most recent call last):\n"
        '  File "foo.py", line 1, in <module>\n'
        "ValueError: bad value\n"
    )
    s = derive_signals(tb)
    assert s["has_traceback"] is True


def test_error_token_alone_matches_traceback() -> None:
    assert derive_signals("RuntimeError: connection lost")["has_traceback"] is True
    assert derive_signals("KeyError: 'x'")["has_traceback"] is True


def test_lowercase_error_word_does_not_match_traceback() -> None:
    # Free-form prose mentioning errors must not trigger the flag.
    assert derive_signals("there was an error here")["has_traceback"] is False
    assert derive_signals("error: lowercase")["has_traceback"] is False


def test_rust_panic_is_detected() -> None:
    assert derive_signals("panic: out of bounds")["has_traceback"] is True
    assert derive_signals("thread 'main' panicked at src/main.rs:10:5")["has_traceback"] is True


def test_decision_verbs_are_detected() -> None:
    for phrase in (
        "We decided to use Postgres",
        "Going with the smaller model",
        "I chose to skip the cache",
        "switching to async",
        "Instead of mocking, we patched the env",
        "opted for a simpler design",
    ):
        assert derive_signals(phrase)["has_decision_verbs"] is True, phrase


def test_non_decision_prose_does_not_match() -> None:
    assert derive_signals("the choice was difficult")["has_decision_verbs"] is False


def test_tool_call_json_payload_is_detected() -> None:
    payload = '{"name": "recall_episodes", "arguments": {"k": 8}}'
    s = derive_signals(payload)
    assert s["has_tool_call"] is True


def test_explicit_tool_call_marker_is_detected() -> None:
    assert derive_signals("tool_call: store_episode")["has_tool_call"] is True
    assert derive_signals("Tool Call -> X")["has_tool_call"] is True


def test_code_block_count_pairs_fences() -> None:
    text = "```python\nprint(1)\n```\n\n```\nplain\n```\n"
    assert derive_signals(text)["code_block_count"] == 2


def test_unbalanced_fence_floors_at_pairs() -> None:
    # Three opening fences, two closers — round down to one block.
    text = "```py\nA\n```\n```py\nB\n```\n```py\nC\n"
    assert derive_signals(text)["code_block_count"] == 2


# --- reference extraction -------------------------------------------


def test_file_paths_extracted_with_supported_extensions() -> None:
    text = "Edit src/auth/login.py and tests/test_login.py and README.md please"
    refs = derive_signals(text)["references"]
    assert "src/auth/login.py" in refs
    assert "tests/test_login.py" in refs
    assert "README.md" in refs


def test_unsupported_extension_is_ignored() -> None:
    refs = derive_signals("look at notes.txt and image.png")["references"]
    assert "notes.txt" not in refs
    assert "image.png" not in refs


def test_backtick_identifiers_extracted() -> None:
    text = "Calling `store_episode` reuses `HippocampusService.recall`."
    refs = derive_signals(text)["references"]
    assert "store_episode" in refs
    assert "HippocampusService.recall" in refs


def test_camelcase_outside_backticks_extracted() -> None:
    refs = derive_signals("The HippocampusConfig field controls X")["references"]
    assert "HippocampusConfig" in refs


def test_camelcase_noise_words_filtered() -> None:
    refs = derive_signals("Use GitHub or GitLab for hosting")["references"]
    assert "GitHub" not in refs
    assert "GitLab" not in refs


def test_capitalised_english_does_not_become_symbol() -> None:
    refs = derive_signals("The model is large")["references"]
    assert "The" not in refs
    assert "This" not in refs


def test_references_are_deduplicated_and_sorted() -> None:
    refs = derive_signals("`foo` and `foo` and src/a.py and src/a.py")["references"]
    assert refs == sorted(set(refs))
    assert refs.count("foo") == 1


# --- length classification ------------------------------------------


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, "short"),
        (199, "short"),
        (200, "medium"),
        (1499, "medium"),
        (1500, "long"),
        (10000, "long"),
    ],
)
def test_length_class_boundaries(length: int, expected: str) -> None:
    text = "x" * length
    assert derive_signals(text)["length_class"] == expected


# --- derived salience -----------------------------------------------


def test_default_salience_for_empty_signals_is_base() -> None:
    assert derive_salience(derive_signals(""), base=0.3) == pytest.approx(0.3)


def test_traceback_plus_diff_no_density_bonus() -> None:
    # Use Rust panic + Go diff so no CamelCase class name leaks into
    # the reference list and triggers the >=3 density bonus.
    text = "panic: out of bounds\n--- a/lib.go\n+++ b/lib.go\n@@ ...\n"
    sig = derive_signals(text)
    # ``a/lib.go`` and ``b/lib.go`` collapse to a single ``lib.go``
    # reference after diff-prefix stripping.
    assert sig["references"] == ["lib.go"]
    # base 0.3 + traceback 0.30 + diff 0.20 = 0.80; one reference < 3
    # so no density bonus.
    assert derive_salience(sig, base=0.3) == pytest.approx(0.80)


def test_reference_density_kicks_in_at_three() -> None:
    two = derive_signals("`a` and `b`")
    three = derive_signals("`a` and `b` and `c`")
    assert derive_salience(two, base=0.0) == pytest.approx(0.0)
    assert derive_salience(three, base=0.0) == pytest.approx(SALIENCE_WEIGHTS["reference_density"])


def test_salience_is_clamped_to_unit_interval() -> None:
    # A maximalist signal vector + base 1.0 must clip at 1.0.
    text = (
        "Traceback (most recent call last):\nRuntimeError: x\n"
        "--- a/foo.py\n+++ b/foo.py\n@@ ...\n"
        "We decided to use Postgres instead of MySQL.\n"
        '{"name": "store_episode"}\n'
        "`a` `b` `c` `d`\n" + ("x" * 1600)
    )
    assert derive_salience(derive_signals(text), base=1.0) == pytest.approx(1.0)
    assert derive_salience(derive_signals(""), base=-0.5) == pytest.approx(0.0)


def test_long_content_bonus_only_at_long_class() -> None:
    short_sig = derive_signals("hi")
    long_sig = derive_signals("x" * 1500)
    assert derive_salience(short_sig, base=0.0) == pytest.approx(0.0)
    assert derive_salience(long_sig, base=0.0) == pytest.approx(SALIENCE_WEIGHTS["long_content"])


# --- URL extraction --------------------------------------------------


def test_url_extracted_as_reference() -> None:
    refs = derive_signals("docs at https://example.com/path are here")["references"]
    assert "https://example.com/path" in refs


def test_url_trailing_punctuation_is_stripped() -> None:
    refs = derive_signals("see https://example.com/x.")["references"]
    assert "https://example.com/x" in refs
    assert "https://example.com/x." not in refs


def test_url_in_parens_does_not_swallow_closer() -> None:
    refs = derive_signals("(https://a.example/p)")["references"]
    assert "https://a.example/p" in refs


# --- Issue / PR ref extraction --------------------------------------


def test_hash_issue_ref_extracted() -> None:
    refs = derive_signals("fixes #123 and #45")["references"]
    assert "#123" in refs
    assert "#45" in refs


def test_prefixed_tracker_ref_extracted() -> None:
    refs = derive_signals("see GH-12 and JIRA-1234")["references"]
    assert "GH-12" in refs
    assert "JIRA-1234" in refs


def test_utf8_like_codes_not_treated_as_issue_refs() -> None:
    refs = derive_signals("encoded as UTF-8 in the header")["references"]
    assert "UTF-8" not in refs


# --- Question signal ------------------------------------------------


def test_question_mark_detected() -> None:
    assert derive_signals("How do we cache this?")["has_question"] is True
    assert derive_signals("Why? Because of latency.")["has_question"] is True


def test_inline_question_mark_in_code_not_detected() -> None:
    # Kotlin nullable / shell glob shapes — no trailing whitespace
    # after the ``?``, so the signal must stay off.
    assert derive_signals("val x: String?=null")["has_question"] is False


def test_question_adds_salience_weight() -> None:
    sig = derive_signals("Should we use Redis?")
    base = derive_salience({"has_question": False}, base=0.0)
    boosted = derive_salience(sig, base=0.0)
    assert boosted - base == pytest.approx(SALIENCE_WEIGHTS["has_question"])


# --- Structured-doc signal ------------------------------------------


def test_markdown_header_detected_as_structured_doc() -> None:
    assert derive_signals("# Heading\nbody")["has_structured_doc"] is True
    assert derive_signals("### Deeper\n")["has_structured_doc"] is True


def test_bullet_list_detected_as_structured_doc() -> None:
    assert derive_signals("- item one\n- item two")["has_structured_doc"] is True
    assert derive_signals("* item")["has_structured_doc"] is True


def test_ordered_list_detected_as_structured_doc() -> None:
    assert derive_signals("1. first\n2. second")["has_structured_doc"] is True


def test_unified_diff_does_not_trigger_structured_doc() -> None:
    diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert derive_signals(diff)["has_structured_doc"] is False


# --- Camel-noise expansion ------------------------------------------


def test_expanded_camel_noise_filters_common_words() -> None:
    refs = derive_signals("Postgres vs MySQL on MacOS or NodeJS")["references"]
    assert "MySQL" not in refs
    assert "MacOS" not in refs
    assert "NodeJS" not in refs


# --- TypedDict surface ----------------------------------------------


def test_signal_dict_type_is_exported() -> None:
    # Smoke-check that the alias exists and the runtime value of
    # ``derive_signals`` is shape-compatible (TypedDict is a dict at
    # runtime, so this is a structural sanity check).
    sig: SignalDict = derive_signals("hello")
    assert sig["length_chars"] == 5
    assert isinstance(sig["references"], list)
