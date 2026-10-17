from __future__ import annotations

from infernux.engine.interaction import ModalService


class _Presenter:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.active = False

    def render(self, _ctx) -> None:
        self.events.append(f"render:{self.name}")

    def cancel(self) -> None:
        self.events.append(f"cancel:{self.name}")
        self.active = False


def _register(
    service: ModalService,
    presenter: _Presenter,
    *,
    allowed_parent_ids=(),
) -> None:
    service.register(
        presenter.name,
        is_active=lambda: presenter.active,
        render=presenter.render,
        cancel=presenter.cancel,
        allowed_parent_ids=allowed_parent_ids,
    )


def test_modal_service_serializes_roots_and_allows_explicit_children():
    events: list[str] = []
    service = ModalService()
    parent = _Presenter("parent", events)
    child = _Presenter("child", events)
    blocked = _Presenter("blocked", events)
    _register(service, parent)
    _register(service, child, allowed_parent_ids=("parent",))
    _register(service, blocked)
    for presenter in (parent, child, blocked):
        presenter.active = True

    assert service.activate("parent", owner_id="particle_graph_editor")
    assert not service.activate("blocked", owner_id="project")
    assert service.activate("child", owner_id="particle_graph_editor")
    assert [entry.modal_id for entry in service.active_stack] == ["parent", "child"]

    service.render(object())
    assert events == ["render:child"]

    child.active = False
    service.render(object())
    assert events == ["render:child", "render:parent"]
    assert service.active_modal_id == "parent"


def test_modal_service_cancels_owned_modal_stack_before_view_destruction():
    events: list[str] = []
    service = ModalService()
    parent = _Presenter("parent", events)
    child = _Presenter("child", events)
    _register(service, parent)
    _register(service, child, allowed_parent_ids=("parent",))
    for presenter in (parent, child):
        presenter.active = True
    assert service.activate("parent", owner_id="timeline")
    assert service.activate("child", owner_id="timeline")

    assert service.cancel_owner("timeline")
    assert events == ["cancel:child", "cancel:parent"]
    assert service.active_stack == ()


def test_lost_popup_releases_input_until_the_presenter_reopens_it():
    events: list[str] = []
    service = ModalService()
    attempts = [False, True]

    def render(_ctx):
        visible = attempts.pop(0)
        events.append(f"render:{visible}")
        return visible

    active = [True]
    service.register(
        "recoverable",
        is_active=lambda: active[0],
        render=render,
        cancel=lambda: active.__setitem__(0, False),
    )
    assert service.activate("recoverable")
    # Activation before the first draw still owns input.
    assert service.active_modal_id == "recoverable"

    service.render(object())
    assert service.is_presented("recoverable") is False
    assert service.active_modal_id == ""

    service.render(object())
    assert service.is_presented("recoverable") is True
    assert service.active_modal_id == "recoverable"
    assert events == ["render:False", "render:True"]
