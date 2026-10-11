from __future__ import annotations

import textwrap

import pytest

from infernux.engine.script_candidate_policy import analyze_script_candidate


def _report(source: str):
    return analyze_script_candidate(textwrap.dedent(source).encode("utf-8"), filename="candidate.py")


@pytest.mark.parametrize(
    "source",
    (
        "from infernux import *\n",
        "from infernux.components import *\n",
        "import numpy as np\nDEFAULT = np.array([1, 2, 3])\n",
        "import numpy as np\nDEFAULT = np.zeros((2, 3), dtype=np.float32)\n",
        "from numpy import dtype, asarray\nDEFAULT = asarray([1, 2], dtype=dtype('f4'))\n",
        "from infernux import Vector3, serialized_field\nVALUE = Vector3(1, 2, 3)\nfield = serialized_field(default=1.0)\n",
        "import infernux as inx\nACCENT = inx.color(0.2, 0.8, 1.0, 1.0)\nCONFIG = inx.DataAssetRef()\n",
        "from dataclasses import dataclass\n@dataclass\nclass Config:\n    value: int = 1\n",
        "from pathlib import Path\nROOT = Path('Assets')\n",
        "def update(self, delta_time):\n    open('out.txt', 'w')\n    import subprocess\n    subprocess.run(['tool'])\n",
        "class Component:\n    def update(self):\n        import os\n        os.environ['X'] = '1'\n",
    ),
)
def test_common_imports_declarations_and_function_bodies_are_not_blocked(source):
    report = _report(source)

    assert report.blocked == ()


@pytest.mark.parametrize("imports, decorator", [
    ("import infernux as inx", "inx.jit.compile"),
    ("import infernux as inx", "inx.jit.compile"),
    ("from infernux import jit as cpu", "cpu.compile"),
    ("import infernux.jit as cpu", "cpu.compile"),
    ("from infernux.jit import compile as optimize", "optimize"),
])
def test_public_jit_compile_is_a_controlled_declaration(imports, decorator):
    report = _report(
        f"{imports}\n@{decorator}(cache=True)\ndef advance(values):\n    return values\n"
    )
    assert report.blocked == report.runtime_guard_required == ()


def test_builtin_compile_call_is_not_confused_with_jit_declaration():
    report = _report("factory = compile('value = 1', '<generated>', 'exec')\n")
    assert len(report.blocked) == 1
    assert report.blocked[0].operation == "compile"


@pytest.mark.parametrize("source", [
    "import infernux as inx\nfactory = inx.jit.compile(cache=True)\n",
    "import unrelated as inx\n@inx.jit.compile()\ndef run(x):\n    return x\n",
])
def test_decorator_factories_do_not_require_named_capabilities(source):
    report = _report(source)
    assert not report.is_rejected


def test_decorator_factories_may_be_held_as_module_values():
    report = _report(
        "from infernux.renderstack import render_effect_feature\n"
        "@render_effect_feature('tests.post.effect')\n"
        "class Effect:\n"
        "    pass\n"
    )

    assert report.blocked == ()
    assert report.runtime_guard_required == ()

    eager = _report(
        "from infernux.renderstack import render_effect_feature\n"
        "factory = render_effect_feature('tests.post.effect')\n"
    )
    assert not eager.is_rejected


@pytest.mark.parametrize("imports, decorator", [
    ("import infernux as inx", "inx.renderstack.geometry_buffer"),
    ("import infernux as inx", "inx.renderstack.geometry_buffer"),
    ("from infernux import renderstack as rendering", "rendering.geometry_buffer"),
    ("import infernux.renderstack as rendering", "rendering.geometry_buffer"),
    ("from infernux.renderstack import geometry_buffer as provider", "provider"),
    ("from infernux.renderstack.geometry_buffers import geometry_buffer", "geometry_buffer"),
])
def test_geometry_buffer_provider_is_a_controlled_declaration(imports, decorator):
    report = _report(
        f"{imports}\nclass Pipeline:\n"
        f"    @{decorator}('preview_color', dependencies={{'depth'}})\n"
        "    def provide(self, context):\n"
        "        return context.graph.create_texture('preview_color')\n"
    )
    assert report.blocked == report.runtime_guard_required == ()


@pytest.mark.parametrize("source", [
    "import infernux as inx\nfactory = inx.renderstack.geometry_buffer('preview_color')\n",
    "import unrelated as inx\nclass Pipeline:\n"
    "    @inx.renderstack.geometry_buffer('preview_color')\n"
    "    def provide(self, context): pass\n",
    "import infernux as inx\nclass Pipeline:\n"
    "    @inx.renderstack.geometry_buffer(read_semantic())\n"
    "    def provide(self, context): pass\n",
])
def test_custom_decorators_and_factory_arguments_are_ordinary_python(source):
    assert not _report(source).is_rejected


def test_lowercase_public_namespace_supports_declaration_only_component_scripts():
    report = _report(
        """
        import infernux as inx

        @inx.require_component(inx.Rigidbody)
        @inx.disallow_multiple()
        class Player(inx.InxComponent):
            direction = inx.serialized_field(
                default=inx.Vector3(0.0, 0.0, 1.0)
            )
        """
    )

    assert report.blocked == ()
    assert report.runtime_guard_required == ()


def test_annotated_field_markers_are_allowed_inside_component_declarations():
    report = _report(
        """
        from typing import Annotated
        import infernux as inx

        @inx.disallow_multiple
        @inx.require_component(inx.Rigidbody)
        class TargetReporter(inx.InxComponent):
            speed: Annotated[
                float,
                inx.components.Header("Movement"),
                inx.components.Range(0.0, 12.0),
                inx.components.Tooltip("Maximum movement speed."),
            ] = 3.0
            target: Annotated[
                inx.GameObject,
                inx.components.Header("References"),
                inx.components.RequiredComponent("MeshRenderer"),
            ]
        """
    )

    assert report.blocked == ()
    assert report.runtime_guard_required == ()


def test_field_metadata_may_be_shared_at_module_scope():
    report = _report(
        "import infernux as inx\n"
        "marker = inx.components.Header('not a declaration')\n"
    )

    assert report.blocked == ()
    assert not report.is_rejected


def test_lowercase_public_namespace_supports_render_declaration_decorator():
    report = _report(
        "import infernux as inx\n"
        "@inx.renderstack.render_effect_feature('tests.post.effect')\n"
        "class Effect:\n    pass\n"
    )

    assert report.blocked == ()
    assert report.runtime_guard_required == ()


@pytest.mark.parametrize(
    ("source", "code", "operation"),
    (
        ("import helper\nhelper.VALUE = 2\n", "NX-R1-STATIC-MODULE-WRITE", "helper.VALUE"),
        ("from helper import VALUE as current\ncurrent += 1\n", "NX-R1-STATIC-MODULE-WRITE", "current"),
        ("import helper\ndel helper.VALUE\n", "NX-R1-STATIC-MODULE-WRITE", "helper.VALUE"),
        ("import os\nos.environ['MODE'] = 'x'\n", "NX-R1-STATIC-ENVIRONMENT-WRITE", "os.environ"),
        ("from os import environ as env\nenv['MODE'] = 'x'\n", "NX-R1-STATIC-ENVIRONMENT-WRITE", "env"),
        ("import sys\nsys.path.append('Assets')\n", "NX-R1-STATIC-MODULE-WRITE", "sys.path.append"),
        ("import sys\nsys.modules['helper'] = None\n", "NX-R1-STATIC-MODULE-WRITE", "sys.modules"),
        ("import sys\nsys.meta_path.clear()\n", "NX-R1-STATIC-MODULE-WRITE", "sys.meta_path.clear"),
        ("open('out.txt', 'wb')\n", "NX-R1-STATIC-FILE-WRITE", "open"),
        ("from io import open as write\nwrite('out.txt', mode='a')\n", "NX-R1-STATIC-FILE-WRITE", "open"),
        ("from pathlib import Path\nPath('out.txt').write_text('x')\n", "NX-R1-STATIC-FILE-WRITE", "Path.write_text"),
        ("from pathlib import Path\nPath('out.txt').write_custom('x')\n", "NX-R1-STATIC-FILE-WRITE", "Path.write_custom"),
        ("from pathlib import Path\nPath('out.txt').touch()\n", "NX-R1-STATIC-FILE-WRITE", "Path.touch"),
        ("from pathlib import Path\np = Path('out.txt')\np.unlink()\n", "NX-R1-STATIC-FILE-WRITE", "p.unlink"),
        ("import os\nos.remove('out.txt')\n", "NX-R1-STATIC-FILE-WRITE", "os.remove"),
        ("import os\nos.system('tool')\n", "NX-R1-STATIC-PROCESS", "os.system"),
        ("import os\nos.spawnv(os.P_WAIT, 'tool', ())\n", "NX-R1-STATIC-PROCESS", "os.spawnv"),
        ("import subprocess\nsubprocess.run(['tool'])\n", "NX-R1-STATIC-PROCESS", "subprocess.run"),
        ("from threading import Thread as Worker\nWorker(target=lambda: None)\n", "NX-R1-STATIC-PROCESS", "threading.Thread"),
        ("import socket\nsocket.socket()\n", "NX-R1-STATIC-PROCESS", "socket.socket"),
        ("import atexit\natexit.register(lambda: None)\n", "NX-R1-STATIC-PROCESS", "atexit.register"),
        ("from importlib import reload as reload_module\nreload_module(helper)\n", "NX-R1-STATIC-DYNAMIC-CODE", "importlib.reload"),
        ("exec('value = 1')\n", "NX-R1-STATIC-DYNAMIC-CODE", "exec"),
    ),
)
def test_obvious_top_level_side_effects_are_blocked(source, code, operation):
    report = _report(source)

    assert report.is_blocked
    assert report.blocked[0].code == code
    assert report.blocked[0].operation == operation
    assert report.blocked[0].line >= 1
    assert report.blocked[0].column >= 0


@pytest.mark.parametrize("source", [
    "result = user_factory(1)\n",
    "from library import metadata\n@metadata('speed')\nclass Probe: pass\n",
    "class Probe:\n    field = field_factory(default=make_value())\n",
    "import importlib\nmodule = importlib.import_module('helper')\n",
    "target = object()\nsetattr(target, 'value', 1)\n",
    "target = object()\ndelattr(target, 'value')\n",
    "import numpy as np\nDEFAULT = np.hstack(([1], [2]))\n",
])
def test_ecosystem_declarations_need_no_class_or_call_allowlist(source):
    report = _report(source)
    assert not report.is_rejected
    assert report.runtime_guard_required == ()


@pytest.mark.parametrize("source", [
    "import asyncio\nVALUE = asyncio.iscoroutine(None)\n",
    "from asyncio import iscoroutinefunction\nVALUE = iscoroutinefunction(lambda: None)\n",
    "import asyncio\nQUEUE = asyncio.Queue()\n",
    "import subprocess\nVALUE = subprocess.list2cmdline(['hello world'])\n",
    "from subprocess import CompletedProcess\nRESULT = CompletedProcess([], 0)\n",
    "import threading\nLOCK = threading.Lock()\n",
    "from threading import RLock\nLOCK = RLock()\n",
    "import threading\nEVENT = threading.Event()\n",
    "import socket\nVALUE = socket.inet_aton('127.0.0.1')\n",
    "from socket import ntohs\nVALUE = ntohs(80)\n",
])
def test_stdlib_declarations_are_not_rejected_by_module_name(source):
    assert not _report(source).is_rejected


@pytest.mark.parametrize("source", [
    "import asyncio\nasyncio.run(main())\n",
    "from asyncio import run_coroutine_threadsafe\nrun_coroutine_threadsafe(main(), loop)\n",
    "import socket\nsocket.create_connection(('localhost', 80))\n",
    "from socket import socketpair\nsocketpair()\n",
    "import multiprocessing\nmultiprocessing.Manager()\n",
    "import atexit\natexit.unregister(callback)\n",
])
def test_explicit_external_stdlib_operations_remain_rejected(source):
    report = _report(source)
    assert report.is_rejected
    assert report.blocked[0].code == "NX-R1-STATIC-PROCESS"


@pytest.mark.parametrize("source", [
    "import numpy as np\nnp.save('state.npy', np.zeros(1))\n",
    "import numpy as np\nnp.seterr(all='ignore')\n",
    "from numpy import save\nsave('state.npy', [1])\n",
])
def test_explicit_external_mutations_still_have_actionable_diagnostics(source):
    report = _report(source)
    assert report.is_blocked
    assert report.is_rejected
    assert report.runtime_guard_required == ()


def test_report_is_immutable_and_preserves_source_locations():
    report = _report("import helper\nhelper.VALUE = 2\n")

    with pytest.raises(AttributeError):
        report.blocked = ()
    assert report.blocked[0].line == 2
    assert report.blocked[0].column == 0
