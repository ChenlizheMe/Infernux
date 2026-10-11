"""One-time compatibility for llvmlite's new-pass-manager ownership bug."""


def prepare_cpu_backend() -> None:
    from llvmlite.binding import ffi, newpassmanagers

    # Some released llvmlite versions put ObjectRef before NewPassManager in
    # the MRO, hiding the mixin's disposer behind ObjectRef's empty method.
    # Correct only that exact defect; fixed upstream classes remain untouched.
    # This runs at backend import, never per compilation or per frame.
    for manager in (newpassmanagers.ModulePassManager, newpassmanagers.FunctionPassManager):
        if manager._dispose is ffi.ObjectRef._dispose:
            manager._dispose = newpassmanagers.NewPassManager._dispose
