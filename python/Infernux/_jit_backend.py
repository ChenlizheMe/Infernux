"""CPU code ownership: published specializations own their LLVM engines.

Numba's shared MCJIT never removes modules. Keep its typing registry and NRT,
but retire authored machine code with its last dispatcher/linking consumer.
Do not call remove_module on a live MCJIT engine.
"""

import dis
import inspect
import weakref
from types import CodeType, FunctionType

from numba.core import compiler, sigutils, types, utils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.cpu import CPUContext
from numba.core.registry import CPUDispatcher, CPUTarget, cpu_target
from numba.core.runtime import rtsys


_MAX_CPU_SPECIALIZATIONS = 64


class _OwnedContext(CPUContext):
    _compiling_owner = None

    def call_unresolved(self, builder, name, signature, args):
        # A recursive call can promote int32 to an already-published int64
        # specialization. Its definition now lives in a different engine;
        # link that specific library instead of assuming a global symbol pool.
        owner = self._compiling_owner() if self._compiling_owner is not None else None
        result = owner.overloads.get(tuple(signature.args)) if owner is not None else None
        if result is not None and result.fndesc.mangled_name == name:
            self.active_code_library.add_linking_library(result.library)
            return self.call_internal(builder, result.fndesc, signature, args)
        return super().call_unresolved(builder, name, signature, args)

    def get_function(self, function, signature, _firstcall=True):
        if isinstance(function, types.Dispatcher):
            result = function.dispatcher.get_compile_result(signature)
            if result.entry_point not in self._defns:
                self.insert_user_function(result.entry_point, result.fndesc, (result.library,))
        elif isinstance(function, types.Function):
            # Registered overloads are resolved through Numba's shared target.
            # Borrow the definition, retaining its library through normal linking.
            key = function.get_impl_key(signature.as_function())
            shared = cpu_target.target_context._defns
            if key in shared and key not in self._defns:
                self._defns[key] = shared[key]
        return super().get_function(function, signature, _firstcall)


class _OwnedTarget(CPUTarget):
    def __init__(self):
        super().__init__("cpu")
        # Process-wide NRT must not retain the first authored private engine.
        rtsys.initialize(cpu_target.target_context)
        # Outside compilation only target metadata is needed. Do not create
        # an unused private MCJIT engine for every dispatcher; authored code
        # is always lowered in compile()'s private context below.
        self._owned_context = cpu_target.target_context

    @property
    def typing_context(self):
        return cpu_target.typing_context

    @property
    def target_context(self):
        return self._owned_context


class _OwnedDispatcher(CPUDispatcher):
    def __init__(self, py_func, locals=None, targetoptions=None, pipeline_class=compiler.Compiler):
        # Keep CPUDispatcher's constructor contract: its serializer rebuilds
        # this class with positional locals/targetoptions in another process.
        self.targetdescr = _OwnedTarget()
        options = {**(targetoptions or {}), "nopython": True}
        super().__init__(py_func, locals, options, pipeline_class)

    def _make_finalizer(self):
        # Numba's default finalizer captures one target context. Each of our
        # specializations instead owns the context that actually compiled it.
        overloads = self.overloads

        def finalize():
            if utils.shutting_down():
                return
            for result in overloads.values():
                context = result.target_context
                if result.entry_point in context._defns:
                    context.remove_user_function(result.entry_point)

        return finalize

    @global_compiler_lock
    def compile(self, sig):
        args, _ = sigutils.normalize_signature(sig)
        args = tuple(args)
        if args in self.overloads:
            return super().compile(sig)
        if len(self.overloads) >= _MAX_CPU_SPECIALIZATIONS:
            raise RuntimeError(
                f"CPU JIT specialization limit ({_MAX_CPU_SPECIALIZATIONS}) exceeded "
                f"for '{self.py_func.__qualname__}'; the new signature was not compiled: {tuple(args)}. "
                "Existing signatures remain valid."
            )
        # Do not add provisional code to an engine containing callable code.
        # This also isolates object files loaded from the on-disk cache.
        previous = self.targetctx
        stage = _OwnedContext(self.typingctx, "cpu")
        stage._compiling_owner = weakref.ref(self)
        self.targetctx = self.targetdescr._owned_context = stage
        try:
            return super().compile(sig)
        except BaseException:
            # A cache-write error can occur *after* add_overload published the
            # entry point. It is then live and must not be closed here.
            if args not in self.overloads:
                stage.codegen()._engine._ee.close()
            raise
        finally:
            # Restore the enclosing context, including for recursive typing.
            # Successful results retain their own context/library/engine.
            self.targetctx = self.targetdescr._owned_context = previous


def _global_names(code):
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            yield instruction.argval
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _global_names(constant)


def compile_cpu(function=None, **options):
    locals_map = options.pop("locals", None)
    pipeline = options.pop("pipeline_class", compiler.Compiler)

    def compile_function(fn):
        # Authored helpers belong to this publication, not Numba's permanent
        # register_jitable registry. Clone only the compiler namespace; never
        # replace functions or globals in the author's live Python module.
        functions = [fn]
        seen = {fn}
        names = {}
        for source in functions:
            names[source] = tuple(dict.fromkeys(_global_names(source.__code__)))
            for name in names[source]:
                dependency = source.__globals__.get(name)
                if (inspect.isfunction(dependency)
                        and dependency.__module__ == fn.__module__
                        and dependency not in seen):
                    seen.add(dependency)
                    functions.append(dependency)
        if len(functions) == 1:
            return _OwnedDispatcher(fn, locals_map, options, pipeline)
        clones = {}
        for source in functions:
            namespace = {name: source.__globals__[name] for name in names[source]
                         if name in source.__globals__}
            namespace["__builtins__"] = source.__builtins__
            namespace["__name__"] = source.__module__
            clone = FunctionType(source.__code__, namespace,
                                 source.__name__, source.__defaults__, source.__closure__)
            clone.__module__ = source.__module__
            clone.__qualname__ = source.__qualname__
            clone.__kwdefaults__ = source.__kwdefaults__
            clones[source] = clone
        dispatchers = {
            source: _OwnedDispatcher(clone, locals_map if source is fn else None,
                                     options if source is fn else {}, pipeline)
            for source, clone in clones.items()
        }
        for source, clone in clones.items():
            for name in names[source]:
                dependency = source.__globals__.get(name)
                if inspect.isfunction(dependency) and dependency in dispatchers:
                    clone.__globals__[name] = dispatchers[dependency]
        return dispatchers[fn]

    return compile_function if function is None else compile_function(function)
