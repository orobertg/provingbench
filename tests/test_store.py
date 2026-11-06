from conftest import record

from provingbench.store import Flag, Store


def test_selection_history_is_kept(store):
    store.select("a", "ollama/m1", "owner")
    store.select("a", "openrouter/x/y", "agent")
    assert store.selection("a") == "openrouter/x/y"
    assert [s["model"] for s in store.selections("a")] == ["ollama/m1", "openrouter/x/y"]


def test_clearing_a_selection(store):
    store.select("a", "ollama/m1", "owner")
    store.select("a", None, "run")
    assert store.selection("a") is None


def test_no_selection_at_first(store):
    assert store.selection("nobody") is None


def test_call_and_flags_round_trip_and_filter(store):
    id_ = store.add_call(record(model="ollama/m", flags=[Flag("empty_reply")]))
    store.add_call(record(model="ollama/other"))
    assert store.call(id_)["flags"] == [{"name": "empty_reply", "detail": ""}]
    assert [c["id"] for c in store.calls(flag="empty_reply")] == [id_]
    assert [c["id"] for c in store.calls(after=id_)] == [id_ + 1]
    assert [c["id"] for c in store.calls(model="ollama/other")] == [id_ + 1]


def test_timestamps_are_utc_iso(store):
    assert store.call(store.add_call(record()))["at"].endswith("+00:00")


def test_request_and_response_come_back_as_data(store):
    call = store.call(store.add_call(record(request={"messages": [{"role": "user"}]}, response={"text": "yo"})))
    assert call["request"]["messages"][0]["role"] == "user" and call["response"]["text"] == "yo"


def test_touch_records_only_live_traffic_as_live(store):
    store.touch_agent("a", live=False)
    assert store.last_live_at("a") is None and [a["id"] for a in store.agents()] == ["a"]
    store.touch_agent("a", live=True)
    assert store.last_live_at("a").endswith("+00:00")


def test_allowed_models(store):
    assert store.allowed_models("a") == []
    store.set_allowed("a", ["openrouter/x/y"])
    assert store.allowed_models("a") == ["openrouter/x/y"]


def test_models_are_replaced_by_id(store):
    store.save_models([{"id": "ollama/m", "upstream": "ollama", "price_prompt": 0, "price_completion": 0,
                        "context_length": None, "supports_tools": None}])
    store.save_models([{"id": "ollama/m", "upstream": "ollama", "price_prompt": 0, "price_completion": 0,
                        "context_length": 9, "supports_tools": True}])
    assert len(store.models()) == 1 and store.model("ollama/m")["context_length"] == 9
    assert store.model("ollama/m")["supports_tools"] is True and store.model("nope") is None


def test_attempt_calls_returns_only_that_attempt(store):
    id_ = store.add_call(record(attempt_id="x"))
    store.add_call(record(attempt_id="y"))
    assert [c["id"] for c in store.attempt_calls("x")] == [id_]


def test_prune_removes_old_calls_and_their_flags(store):
    old = store.add_call(record(at="2020-01-01T00:00:00+00:00", flags=[Flag("slow")]))
    keep = store.add_call(record())
    assert store.prune("2021-01-01T00:00:00+00:00") == 1
    assert store.call(old) is None and store.call(keep) and store.calls(flag="slow") == []


def test_a_second_store_on_the_same_file_sees_the_data(cfg, store):
    store.select("a", "ollama/m", "owner")
    other = Store(cfg.data_dir / "t.db")
    assert other.selection("a") == "ollama/m"
    other.close()
