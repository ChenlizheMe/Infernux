# Debug

<div class="class-info">
类位于 <b>infernux.debug</b>
</div>

## 描述

调试工具类。

<!-- USER CONTENT START --> description

Console 同时保存普通日志和有明确归属的当前问题。Shader 编译、资产导入、脚本编译和脚本回调使用跟踪诊断：同一操作重复失败时更新原记录，成功后解决该问题，错误计数随之减少。普通 `log_error` 保留历史，不会因为某个资产修好而被顺带清除。

诊断用 `owner` 和 `key` 标识。`owner` 区分功能，`key` 区分这个功能中的操作或对象；文件路径和行号负责定位，`details` 保存完整编译信息或调用栈。不同对象、不同回调及不同编译任务各自跟踪。脚本重新编译成功只解决编译诊断；运行错误要等对应回调真正成功执行后才会解决。

<!-- USER CONTENT END -->

## 静态方法

| 方法 | 描述 |
|------|------|
| `static Debug.log(message: Any, context: Any = ...) → None` | 输出日志消息到控制台。 |
| `static Debug.log_warning(message: Any, context: Any = ...) → None` | 输出警告消息到控制台。 |
| `static Debug.log_error(message: Any, context: Any = ..., source_file: str = ..., source_line: int = ...) → None` | 输出错误消息到控制台。 |
| `static Debug.log_exception(exception: Exception, context: Any = ...) → None` | Log an exception to the console. |
| `static Debug.log_suppressed(where: str, exc: BaseException, context: Any = ...) → None` | Log an exception that was deliberately swallowed by the caller. |
| `static Debug.log_assert(condition: bool, message: Any = ..., context: Any = ...) → None` | Assert a condition and log if it fails. |
| `static Debug.clear_console() → None` | Clear all messages in the debug console. |
| `static Debug.log_internal(message: Any, context: Any = ...) → None` | Log an internal engine message (hidden from user by default). |
| `static Debug.set_diagnostic(owner, key, message, context=None, *, source_file="", source_line=0, details="", log_type=LogType.ERROR) → None` | 发布或更新一个可恢复的问题。 |
| `static Debug.resolve_diagnostic(owner: str, key: str) → bool` | 解决对应问题，并同步更新 Console。 |
| `static Debug.resolve_script_error(site: str, context=None) → bool` | 确认特定对象的脚本回调已经恢复。内置生命周期调度会自动调用。 |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

自定义工具也可以发布和解决自己的问题：

```python
from infernux.debug import Debug

Debug.set_diagnostic(
    "navigation", "level_main",
    "导航数据缺少入口区域",
    source_file="Assets/Levels/main.py",
    source_line=12,
    details="完整构建诊断放在这里。",
)

# 同一个 owner/key 的再次报告更新同一项；构建成功后明确解决。
Debug.resolve_diagnostic("navigation", "level_main")
```

需要读取当前问题时，使用 `DebugConsole.instance().get_diagnostics()`；传入 `owner` 可只读取一个功能的问题。无需扫描全部日志，也不需要按错误文本猜测哪个问题已经解决。

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
