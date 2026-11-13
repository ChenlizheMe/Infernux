# Debug

<div class="class-info">
class in <b>infernux.debug</b>
</div>

## Description

Utility class for logging messages to the console.

<!-- USER CONTENT START --> description

The Console retains ordinary log history and tracks current problems by operation. Shader compilation, asset import, script compilation, and script callbacks update one owned diagnostic when an operation fails repeatedly. A successful result resolves that problem and updates the error count. Ordinary `log_error` history remains independent.

An `owner` identifies the reporting subsystem and a `key` identifies its operation or object. Source location supports navigation; `details` contains the full compiler output or traceback. Different objects and callback phases remain independent. Successful script compilation resolves compilation diagnostics; a runtime failure resolves only after the corresponding callback actually succeeds.

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `static Debug.log(message: Any, context: Any = ...) → None` | Log a message to the console. |
| `static Debug.log_warning(message: Any, context: Any = ...) → None` | Log a warning message to the console. |
| `static Debug.log_error(message: Any, context: Any = ..., source_file: str = ..., source_line: int = ...) → None` | Log an error message to the console. |
| `static Debug.log_exception(exception: Exception, context: Any = ...) → None` | Log an exception to the console. |
| `static Debug.log_suppressed(where: str, exc: BaseException, context: Any = ...) → None` | Log an exception that was deliberately swallowed by the caller. |
| `static Debug.log_assert(condition: bool, message: Any = ..., context: Any = ...) → None` | Assert a condition and log if it fails. |
| `static Debug.clear_console() → None` | Clear all messages in the debug console. |
| `static Debug.log_internal(message: Any, context: Any = ...) → None` | Log an internal engine message (hidden from user by default). |
| `static Debug.set_diagnostic(owner, key, message, context=None, *, source_file="", source_line=0, details="", log_type=LogType.ERROR) → None` | Publish or update a recoverable problem. |
| `static Debug.resolve_diagnostic(owner: str, key: str) → bool` | Resolve the problem and update the Console. |
| `static Debug.resolve_script_error(site: str, context=None) → bool` | Confirm recovery of one object's callback. Built-in lifecycle dispatch calls this automatically. |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example

Custom tools can publish and resolve their own problems:

```python
from infernux.debug import Debug

Debug.set_diagnostic(
    "navigation", "level_main",
    "Navigation data has no entrance region",
    source_file="Assets/Levels/main.py",
    source_line=12,
    details="Full build diagnostics go here.",
)

# Reporting the same owner/key updates that record. Resolve it after success.
Debug.resolve_diagnostic("navigation", "level_main")
```

Read current problems with `DebugConsole.instance().get_diagnostics()`, optionally supplying an `owner`. This queries active diagnostics without scanning log history or inferring recovery from message text.

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
