"""enable_keyboard must fail closed: no operator Enter -> never publish enable=true."""
import io
import types

import pytest

from rebel_mocap_follower import enable_keyboard as ek


class FakeTty(io.StringIO):
    def isatty(self):
        return True


class RaisingTty(FakeTty):
    def readline(self, *args, **kwargs):
        raise OSError("terminal went away")


def test_enter_on_a_tty_is_consent():
    assert ek.wait_for_enter(FakeTty("\n")) is True
    assert ek.wait_for_enter(FakeTty("go\n")) is True


def test_eof_is_not_consent():
    assert ek.wait_for_enter(FakeTty("")) is False


def test_partial_line_without_newline_is_not_consent():
    assert ek.wait_for_enter(FakeTty("y")) is False


def test_oserror_is_not_consent():
    assert ek.wait_for_enter(RaisingTty()) is False


def test_non_tty_is_not_consent():
    assert ek.wait_for_enter(io.StringIO("\n")) is False
    assert ek.wait_for_enter(None) is False


class FakeRclpy:
    def __init__(self, published):
        self.published = published

    def init(self, args=None):
        pass

    def create_node(self, name):
        published = self.published

        class Pub:
            def publish(self, msg):
                published.append(msg.data)

        class Node:
            def create_publisher(self, *a, **k):
                return Pub()

            def destroy_node(self):
                pass

        return Node()

    def ok(self):
        # One loop iteration at most, then "shutdown".
        self.published.append("loop")
        return len([p for p in self.published if p == "loop"]) <= 1

    def shutdown(self):
        pass


def run_main(monkeypatch, stdin, args=()):
    published = []
    monkeypatch.setattr(ek, "rclpy", FakeRclpy(published))
    monkeypatch.setattr(ek, "Bool", lambda data: types.SimpleNamespace(data=data))
    monkeypatch.setattr(ek.time, "sleep", lambda s: None)
    monkeypatch.setattr(ek.signal, "signal", lambda *a: None)
    monkeypatch.setattr(ek.sys, "stdin", stdin)
    monkeypatch.setattr(ek, "_stop", False)
    ek.main(list(args))
    return [p for p in published if p != "loop"]


@pytest.mark.parametrize("stdin", [FakeTty(""), RaisingTty(), io.StringIO("\n")],
                         ids=["eof", "oserror", "not-a-tty"])
def test_main_never_publishes_true_without_an_operator(monkeypatch, stdin):
    published = run_main(monkeypatch, stdin)
    assert True not in published
    assert published and all(p is False for p in published)  # released: false only


def test_main_publishes_true_after_enter_and_false_on_exit(monkeypatch):
    published = run_main(monkeypatch, FakeTty("\n"))
    assert published[0] is True
    assert published[-1] is False


def test_main_no_prompt_is_the_explicit_opt_in(monkeypatch):
    published = run_main(monkeypatch, FakeTty(""), ["--no-prompt"])
    assert published[0] is True
    assert published[-1] is False
