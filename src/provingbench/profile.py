"""What the results add up to: the models-by-tasks scoreboard, and each model's strengths and weaknesses."""
import statistics


def _effective(results: list[dict], overrides: dict[int, dict]) -> list[dict]:
    return [{**r, "effective": overrides[r["id"]]["outcome"] if r["id"] in overrides else r["outcome"]} for r in results]


def _latest_cells(store) -> tuple[list[dict], dict[str, str], dict[tuple[str, str], list[dict]]]:
    """(results, latest hash per task, and for each model and task the results of its most recent run).

    Only the most recent run counts, so a fix in the agent or a re-run shows up at once. A cell from an older version
    of a task is kept, and marked stale."""
    results = _effective(store.all_results(), store.overrides())
    latest_hash = {r["task_id"]: r["task_hash"] for r in results}
    last_run: dict[tuple[str, str], int] = {}
    for r in results:
        key = (r["model"], r["task_id"])
        last_run[key] = r["run_id"]
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        key = (r["model"], r["task_id"])
        if r["run_id"] == last_run[key]:
            cells.setdefault(key, []).append(r)
    return results, latest_hash, cells


def _counts(rows: list[dict]) -> dict:
    outcomes = [r["effective"] for r in rows]
    passed, failed = outcomes.count("pass"), outcomes.count("fail")
    return {"passed": passed, "failed": failed, "errors": outcomes.count("error"),
            "rate": passed / (passed + failed) if passed + failed else None}


def scoreboard(store) -> dict:
    results, latest_hash, cells = _latest_cells(store)
    tags = {r["task_id"]: r["tags"] for r in results}
    board_cells = []
    for (model, task_id), rows in cells.items():
        task_hash = rows[-1]["task_hash"]
        board_cells.append({"model": model, "task_id": task_id, "task_hash": task_hash,
                            "stale": task_hash != latest_hash[task_id], "run_id": rows[-1]["run_id"], **_counts(rows)})
    return {"tasks": [{"id": task_id, "tags": tags[task_id]} for task_id in sorted(tags)],
            "models": sorted({r["model"] for r in results}), "cells": board_cells}


def profile(store, model_id: str) -> dict:
    _, latest_hash, cells = _latest_cells(store)
    mine = {task_id: rows for (model, task_id), rows in cells.items() if model == model_id}
    tagged: dict[str, list[dict]] = {}
    for rows in mine.values():
        for tag in rows[-1]["tags"]:
            tagged.setdefault(tag, []).extend(rows)
    everything = [r for rows in mine.values() for r in rows]
    ttfts = store.model_ttfts(model_id)
    return {"model": model_id,
            "by_tag": {tag: _counts(rows)["rate"] for tag, rows in tagged.items()},
            "flags": store.model_flags(model_id),
            "failed_tasks": sorted(task_id for task_id, rows in mine.items() if _counts(rows)["failed"]),
            "stale_tasks": sorted(t for t, rows in mine.items() if rows[-1]["task_hash"] != latest_hash[t]),
            "median_ttft_s": statistics.median(ttfts) if ttfts else None,
            "cost_per_task": sum(r["cost_usd"] or 0 for r in everything) / len(everything) if everything else None,
            "last_run": max((r["at"] for r in everything), default=None),
            "notes": store.notes(model_id)}
