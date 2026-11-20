"""Inherited declarations are reconstructed from the current source together."""
import sys
import pytest
from test_reload_coordinator import reload_project as reload_project


@pytest.mark.parametrize("changed_base", [False, True])
def test_current_base_fields_and_super_use_current_classes(reload_project, changed_base):
    scene, coordinator, write = reload_project
    base = write("ReloadBase", """
        from infernux.components import InxComponent
        class Base(InxComponent):
            value: int = 7
            def marker(self): return 'old-base'
    """)
    child = write("ReloadChild", """
        from ReloadBase import Base
        class Child(Base):
            def marker(self): return super().marker() + ':child'
    """)
    coordinator.ensure_ready((base, child))
    obj = scene.create_game_object("继承组件")
    old = sys.modules["ReloadChild"].Child()
    old._script_path = child
    obj.add_py_component(old)
    identity = old.component_id
    old.value = 31
    write("ReloadBase", """
        from infernux.components import InxComponent
        class Base(InxComponent):
            value: int = 7
            extra: int = 9
            def marker(self): return 'new-base'
    """)
    if changed_base:
        write("ReloadChild", """
            from ReloadBase import Base
            class NewBase(Base):
                def marker(self): return super().marker() + ':new-base'
            class Child(NewBase):
                def marker(self): return super().marker() + ':child'
        """)
    coordinator.ensure_ready((base, child))
    current = obj.get_py_components()[0]
    assert current is not old and current.component_id == identity
    assert current.value == 31 and current.extra == 9
    assert isinstance(current, sys.modules["ReloadBase"].Base)
    assert current.marker() == ('new-base:new-base:child' if changed_base else 'new-base:child')
