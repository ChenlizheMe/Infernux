# Infernux Automation SDK

A standard-library-only client. Importing it never imports Infernux, native bindings, a renderer or an MCP server.
The engine distribution includes this module. An independent harness/service environment can install only this subproject with `pip install ./python` or its pure-Python wheel.

```python
from infernux_automation import AutomationClient
engine = AutomationClient("/path/to/project")
print(engine.operations("camera"))
job = engine.submit("infernux.player.build", {"target": "windows"})
print(engine.job(job["job_id"]))
```

The project must have an active engine. Discovery binds the exact instance. Failed writes are never replayed. `submit` returns the engine's job handle; closing the client releases observations without cancelling an already running job or undoing its effects.

Use `observe` and `observation(after=...)` for signals, `stop_observing` to release them. A client is an observation owner; use it as a context manager or call `close()` when its service/session ends. Observation requests renew the lease. An otherwise idle harness can call `keepalive()` at least every 25 seconds; after 90 seconds without activity the engine releases a dead owner's subscriptions. This does not query the world or add frame polling. MCP handles this lifetime automatically.

`report(query_id, consumer, reports=3, mode="turns")` schedules delivery for the consumer. `reports(consumer, cycle_id=real_user_turn_id)` reads it. Changes and actual user turns have separate counting rules; A → B → A does not count the repeated A again. A turn can report a pending/failed query honestly. Exhausting the quota or stopping its report closes the managed query; use `close_query=False` if a separate owner still needs it. Closing a query also removes its report deliveries.

The current Host exposes plugin install/update/enable/uninstall/reload, compatible releases and theme list/select/refresh. These call the same `PluginManager` and `EditorThemes` as the GUI. For example, `call("infernux.plugin.update", {"reference": "vendor/plugin"})` selects the highest compatible published release. An explicit `release_tag` or local package `source` can be supplied. Publisher files are overwritten, removed publisher members are deleted, and independent user files survive. Activation errors and restart requirements remain in the returned actual state.

The thin CLI uses that same directory and does not import the engine:

```powershell
python -m infernux_automation --project E:/Games/MyGame operations plugin
python -m infernux_automation --project E:/Games/MyGame call infernux.plugin.list
python -m infernux_automation --project E:/Games/MyGame call infernux.plugin.update --arguments-file update.json
```

`update.json` contains the operation's arguments, for example `{"reference":"vendor/plugin"}`. Use `submit` instead of `call` for work that should return a job handle immediately. No separate installer, permission mode or task registration is required.

An optional external harness can use this wheel from its own Python environment. Its ordinary engine plugin can register domain operations through `infernux.host.OperationRegistry` during `InxPreload.preload`; the normal source owner retires those registrations on unload/failure. The harness invokes the published IDs through this SDK or MCP. SDK entry points are an optional integration detail of that harness, not a requirement for `.inxpkg` packages, and the engine does not import a specific agent/model SDK.
