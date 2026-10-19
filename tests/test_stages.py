import pytest

from enkaz.stages import StageRunner


def test_stage_runs_once_and_resumes(tmp_path):
    calls = []
    runner = StageRunner(tmp_path, log=lambda _: None)

    def work():
        calls.append(1)
        return {"n": 3}

    assert runner.run("crawl", work) == {"n": 3}
    assert runner.run("crawl", work) == {"n": 3}
    assert len(calls) == 1

    # a fresh runner (new Colab session) still sees the marker
    assert StageRunner(tmp_path, log=lambda _: None).run("crawl", work) == {"n": 3}
    assert len(calls) == 1


def test_version_bump_and_force_rerun(tmp_path):
    calls = []
    runner = StageRunner(tmp_path, log=lambda _: None)
    runner.run("s", lambda: calls.append(1))
    runner.run("s", lambda: calls.append(1), version="2")
    runner.run("s", lambda: calls.append(1), version="2", force=True)
    assert len(calls) == 3


def test_failed_stage_leaves_no_marker(tmp_path):
    runner = StageRunner(tmp_path, log=lambda _: None)

    def boom():
        raise RuntimeError("disconnected")

    with pytest.raises(RuntimeError):
        runner.run("s", boom)
    assert not runner.done("s")
