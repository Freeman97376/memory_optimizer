from memory_optimizer.storage.evidence_search import assemble


def test_exact_source_is_deduplicated_and_not_cut_at_output_limit():
    source = dict(
        batch_id="b", message_ordinal=0, start_offset=0, end_offset=12, content="only v9. 🌱\r\n"
    )
    body = dict(members=[], relations=[])
    value = assemble(body, [source, source], "v9")
    assert value.count(source["content"]) == 1
    assert assemble(body, [source], "v9", limit=len(value)) == value
    assert assemble(body, [source], "v9", limit=len(value) - 1) is None


def test_raw_fallback_never_claims_a_confirmed_episode():
    source = dict(batch_id="b", message_ordinal=0, start_offset=0, end_offset=7, content="source!")
    value = assemble(None, [source], "source")
    assert value.startswith("原文证据片段") and "source!" in value
