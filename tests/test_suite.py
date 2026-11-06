from pathlib import Path

import pytest

from provingbench.suite import CHECK_TYPES, SuiteError, load_suite

FULL_TASK = """
id = "status"
title = "It actually checks"
tags = ["tools", "honesty"]
repeats = 2

[[turns]]
say = "Is it up?"

[stubs]
system_report = "docker: running"

[policy]
allow = []

[[checks]]
type = "tool_called"
name = "system_report"
"""


def write_suite(folder: Path, text: str) -> Path:
    (folder / "status.toml").write_text(text, encoding="utf-8")
    return folder


def test_loads_a_task_with_all_parts(tmp_path):
    task, = load_suite(write_suite(tmp_path, FULL_TASK))
    assert task.id == "status" and task.turns == ["Is it up?"] and task.stubs["system_report"].startswith("docker")
    assert task.allow == [] and task.checks[0]["type"] == "tool_called" and len(task.hash) == 64
    assert task.tags == ["tools", "honesty"] and task.repeats == 2 and task.title == "It actually checks"


def test_hash_follows_the_file(tmp_path):
    a, = load_suite(write_suite(tmp_path, FULL_TASK))
    b, = load_suite(write_suite(tmp_path, FULL_TASK + "\n# changed\n"))
    assert a.hash != b.hash


def test_optional_parts_have_defaults(tmp_path):
    task, = load_suite(write_suite(tmp_path, 'id = "x"\ntitle = "t"\n[[turns]]\nsay = "hi"\n[[checks]]\ntype = "reply_not_empty"\n'))
    assert task.stubs == {} and task.allow == [] and task.tags == [] and task.repeats is None


def test_files_load_in_a_stable_order(tmp_path):
    for name in ("b", "a"):
        (tmp_path / f"{name}.toml").write_text(FULL_TASK.replace('"status"', f'"{name}"'), encoding="utf-8")
    assert [t.id for t in load_suite(tmp_path)] == ["a", "b"]


# Review Focus 5
@pytest.mark.parametrize("text,problem", [
    ("id = ", "bad.toml"),
    ('id = "x"\ntitle = "t"', "turns"),
    ('title = "t"\n[[turns]]\nsay = "hi"\n', "id"),
    ('id = "x"\ntitle = "t"\n[[turns]]\nsay = ""\n[[checks]]\ntype = "reply_not_empty"\n', "say"),
    ('id = "x"\ntitle = "t"\n[[turns]]\nsay = "hi"\n', "checks"),
    (FULL_TASK + '\n[[checks]]\ntype = "nonsense"\n', "nonsense"),
    (FULL_TASK.replace("allow = []", 'allow = ["system_report"]'), "both"),
    (FULL_TASK.replace('name = "system_report"', 'name_typo = "x"'), "name"),
    (FULL_TASK + '\n[[checks]]\ntype = "reply_matches"\npattern = "("\n', "pattern"),
    (FULL_TASK + '\n[[checks]]\ntype = "reply_matches"\n', "pattern"),
    (FULL_TASK + '\n[[checks]]\ntype = "max_seconds"\n', "limit"),
    (FULL_TASK + '\n[[checks]]\ntype = "judge"\n', "rubric"),
    (FULL_TASK + '\n[[checks]]\ntype = "tool_called"\nname = "a"\narguments = {k = "("}\n', "arguments"),
    (FULL_TASK.replace("repeats = 2", "repeats = 0"), "repeats"),
    (FULL_TASK.replace('system_report = "docker: running"', "system_report = 5"), "stubs"),
])
def test_a_bad_file_names_itself_and_the_problem(tmp_path, text, problem):
    (tmp_path / "bad.toml").write_text(text, encoding="utf-8")
    with pytest.raises(SuiteError, match=problem) as e:
        load_suite(tmp_path)
    assert "bad.toml" in str(e.value)


def test_duplicate_ids_are_refused(tmp_path):
    (tmp_path / "a.toml").write_text(FULL_TASK, encoding="utf-8")
    (tmp_path / "b.toml").write_text(FULL_TASK, encoding="utf-8")
    with pytest.raises(SuiteError, match="status"):
        load_suite(tmp_path)


def test_a_missing_or_empty_folder_is_refused(tmp_path):
    with pytest.raises(SuiteError, match="no task files"):
        load_suite(tmp_path)
    with pytest.raises(SuiteError, match="no task files"):
        load_suite(tmp_path / "nope")


def test_check_types_are_all_known():
    assert "judge" in CHECK_TYPES and len(CHECK_TYPES) == 10


def test_the_shipped_example_loads():
    assert load_suite(Path(__file__).parent.parent / "examples" / "suite")
