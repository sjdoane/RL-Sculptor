"""tests/test_kg_research.py — offline tests for `sculptor.kg.research`.

These tests stub `client.messages.parse` so no Anthropic API calls
happen. They pin the kwarg name the research module uses against
`anthropic.Anthropic.messages.parse`, which is `output_format` as of
SDK 0.96.x. A regression to `response_format` (Openai-style) is what
bug #6.4 / T5 was.
"""

from __future__ import annotations

from sculptor.kg import research as research_mod
from sculptor.kg.research import ResearchPaper, ResearchResponse, research_topic


class _StubResponse:
    """Mimics `anthropic.types.ParsedMessage`. The real SDK exposes the
    parsed payload under `.parsed_output` (a property that scans content
    blocks). If research.py reads `.output` instead, tests on a real SDK
    raise `AttributeError: 'ParsedMessage[TypeVar]' object has no
    attribute 'output'` — which is exactly what Sam hit on the Cartpole
    KG Research-a-topic button.
    """
    def __init__(self, parsed):
        self.parsed_output = parsed

    @property
    def output(self):  # pragma: no cover
        raise AttributeError(
            "'_StubResponse' does not expose `.output` — research.py must "
            "read `.parsed_output` (matches anthropic SDK 0.96.x + the "
            "extract.py / diagnose.py siblings)."
        )


class _StubMessages:
    def __init__(self, parsed):
        self._parsed = parsed
        self.captured_kwargs: list[dict] = []

    def parse(self, **kwargs):
        self.captured_kwargs.append(kwargs)
        return _StubResponse(self._parsed)


class _StubClient:
    def __init__(self, parsed):
        self.messages = _StubMessages(parsed)


def test_research_response_records_pipeline_counters():
    """Issue D (Test 1 2026-04-22): ResearchResponse must surface
    `papers_returned_by_claude` and `papers_deduped_against_kg` so the
    UI can distinguish 'Claude returned 0' from 'all already in KG'.
    """
    # No-KG path: Claude returns 2 papers, no dedupe.
    parsed = ResearchResponse(
        papers=[
            ResearchPaper(
                arxiv_id="1234.5678", title="A", relevance_score=0.9,
                justification="j",
            ),
            ResearchPaper(
                arxiv_id="2345.6789", title="B", relevance_score=0.8,
                justification="j",
            ),
        ],
        coverage_note="",
    )
    client = _StubClient(parsed)
    res = research_topic(
        "balance the pole", client=client, dedupe_against_kg=False,
    )
    assert res.papers_returned_by_claude == 2
    assert res.papers_deduped_against_kg == 0
    assert [p.arxiv_id for p in res.papers] == ["1234.5678", "2345.6789"]


def test_research_response_counts_dedupe_against_preexisting_kg(tmp_path):
    """When dedupe_against_kg=True and some IDs are already in the KG,
    `papers_deduped_against_kg` counts the drops."""
    from sculptor.kg.schema import Paper, make_paper_id
    from sculptor.kg.store import SculptorKG

    parsed = ResearchResponse(
        papers=[
            ResearchPaper(
                arxiv_id="9999.1111", title="existing", relevance_score=0.9,
                justification="j",
            ),
            ResearchPaper(
                arxiv_id="9999.2222", title="new", relevance_score=0.8,
                justification="j",
            ),
        ],
        coverage_note="",
    )
    client = _StubClient(parsed)

    # Seed the first paper into a real KG.
    db = tmp_path / "kg.db"
    with SculptorKG(db) as store:
        store.add_node(Paper(
            id=make_paper_id("9999.1111"),
            arxiv_id="9999.1111", title="existing",
        ))
    with SculptorKG(db) as store:
        res = research_topic(
            "topic", client=client, store=store, dedupe_against_kg=True,
        )
    assert res.papers_returned_by_claude == 2
    assert res.papers_deduped_against_kg == 1
    # Only the new one survives.
    assert [p.arxiv_id for p in res.papers] == ["9999.2222"]


def test_research_topic_uses_output_format_kwarg():
    """research_topic must pass `output_format`, not `response_format`."""
    parsed = ResearchResponse(papers=[], coverage_note="")
    client = _StubClient(parsed)

    research_topic(
        "series-elastic actuator dynamics",
        client=client,
        dedupe_against_kg=False,
    )

    assert len(client.messages.captured_kwargs) == 1
    kwargs = client.messages.captured_kwargs[0]
    assert "output_format" in kwargs, (
        "research_topic must pass `output_format=` to messages.parse "
        "(matching anthropic SDK 0.96.x). bug #6.4 regression guard."
    )
    assert "response_format" not in kwargs, (
        "`response_format=` is OpenAI-SDK syntax; anthropic's "
        "messages.parse signature does not accept it. bug #6.4."
    )
    assert kwargs["output_format"] is ResearchResponse


def test_research_module_call_sites_agree_with_extract_and_diagnose():
    """All three messages.parse call sites in sculptor.kg + sculptor.diagnose
    must use the same kwarg name AND the same response-unpacking
    attribute. Cross-module consistency guard."""
    import inspect

    from sculptor import diagnose as diagnose_mod
    from sculptor.kg import extract as extract_mod

    for mod in (research_mod, extract_mod, diagnose_mod):
        src = inspect.getsource(mod)
        # One or more messages.parse call sites per module.
        assert "messages.parse(" in src, f"{mod.__name__} has no messages.parse call"
        assert "response_format=" not in src, (
            f"{mod.__name__} uses `response_format=` — that's OpenAI-SDK syntax "
            "and will fail with anthropic SDK 0.96.x. Use `output_format=`."
        )
        assert "output_format=" in src, (
            f"{mod.__name__} is missing `output_format=` on messages.parse."
        )
        # Response unpacking: ParsedMessage exposes `.parsed_output`, not
        # `.output`. A bare `.output` is an AttributeError on real SDK
        # responses — matches what Sam hit on the KG Research-a-topic
        # flow during the 2026-04-22 Cartpole test. Regex-style check so
        # a variable literally named `.output` (e.g. block.parsed_output)
        # doesn't false-trip.
        assert "resp.parsed_output" in src or ".parsed_output" in src, (
            f"{mod.__name__} should unpack via `.parsed_output`; a bare "
            "`resp.output` raises AttributeError on anthropic SDK 0.96.x."
        )
        # Specifically forbid the broken form.
        assert "resp.output" not in src, (
            f"{mod.__name__} uses `resp.output` — broken; swap to "
            "`resp.parsed_output`."
        )
