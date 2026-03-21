#include "PyComponentProxy.h"
#include "Collider.h"
#include "GameObject.h"
#include "physics/PhysicsContactListener.h"
#include <core/log/InfLog.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace infengine
{

namespace
{
void BindPythonMirrorHelpers(const py::object &pyComponent, Component *nativeComponent, GameObject *gameObject)
{
    if (pyComponent.is_none())
        return;

    try {
        if (py::hasattr(pyComponent, "_bind_native_component")) {
            pyComponent.attr("_bind_native_component")(
                py::cast(nativeComponent, py::return_value_policy::reference),
                gameObject ? py::cast(gameObject, py::return_value_policy::reference) : py::none());
            return;
        }

        if (gameObject && py::hasattr(pyComponent, "_set_game_object")) {
            pyComponent.attr("_set_game_object")(py::cast(gameObject, py::return_value_policy::reference));
        }
        pyComponent.attr("_cpp_component") = py::cast(nativeComponent, py::return_value_policy::reference);
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Failed to bind Python mirror: ", e.what());
    }
}

void SyncPythonMirrorState(const py::object &pyComponent, const Component *nativeComponent)
{
    if (pyComponent.is_none() || !nativeComponent)
        return;

    try {
        if (py::hasattr(pyComponent, "_sync_native_state")) {
            pyComponent.attr("_sync_native_state")(nativeComponent->IsEnabled(), nativeComponent->HasAwake(),
                                                   nativeComponent->HasStarted(), nativeComponent->IsDestroyed(),
                                                   nativeComponent->GetExecutionOrder());
            return;
        }

        pyComponent.attr("_component_id") = py::int_(nativeComponent->GetComponentID());
        pyComponent.attr("_execution_order") = py::int_(nativeComponent->GetExecutionOrder());
        pyComponent.attr("_enabled") = py::bool_(nativeComponent->IsEnabled());
        pyComponent.attr("_awake_called") = py::bool_(nativeComponent->HasAwake());
        pyComponent.attr("_has_started") = py::bool_(nativeComponent->HasStarted());
        pyComponent.attr("_is_destroyed") = py::bool_(nativeComponent->IsDestroyed());
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Failed to sync Python mirror state: ", e.what());
    }
}

void SyncEnabledFromPython(const py::object &pyComponent, bool &enabled, const char *phase)
{
    if (!py::hasattr(pyComponent, "enabled")) {
        return;
    }

    try {
        enabled = pyComponent.attr("enabled").cast<bool>();
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Failed to get enabled state",
                     (phase && phase[0] != '\0') ? std::string(" in ") + phase : std::string(),
                     ": ", e.what());
    }

    pyComponent.attr("enabled") = py::bool_(enabled);
}

void CallPythonLifecycleNoArg(const py::object &pyComponent, const std::string &typeName, const char *entryPoint,
                              const char *displayName)
{
    try {
        pyComponent.attr(entryPoint)();
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Error in ", typeName, ".", displayName, "(): ", e.what());
    }
}

void CallPythonLifecycleFloatArg(const py::object &pyComponent, const std::string &typeName, const char *entryPoint,
                                 const char *displayName, float value)
{
    try {
        pyComponent.attr(entryPoint)(value);
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Error in ", typeName, ".", displayName, "(): ", e.what());
    }
}

void CallPythonLifecycleOneArg(const py::object &pyComponent, const std::string &typeName, const char *entryPoint,
                               const char *displayName, py::object arg)
{
    try {
        pyComponent.attr(entryPoint)(std::move(arg));
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Error in ", typeName, ".", displayName, "(): ", e.what());
    }
}
} // namespace

PyComponentProxy::PyComponentProxy(py::object pyComponent)
    : m_pyComponent(std::move(pyComponent)), m_typeName("PyComponent")
{
    if (!m_pyComponent.is_none()) {
        try {
            // Get the Python class name for type identification
            py::object pyType = m_pyComponent.attr("__class__");
            m_typeName = pyType.attr("__name__").cast<std::string>();

            if (py::hasattr(pyType, "_execute_in_edit_mode_")) {
                try {
                    m_executeInEditMode = pyType.attr("_execute_in_edit_mode_").cast<bool>();
                } catch (const py::error_already_set &e) {
                    INFLOG_WARN("[PyComponentProxy] Failed to read _execute_in_edit_mode_ for '", m_typeName, "': ", e.what());
                    m_executeInEditMode = false;
                }
            }

            // Get stable type GUID from Python class (module.classname hash)
            if (py::hasattr(m_pyComponent, "_get_type_guid")) {
                py::object typeGuid = pyType.attr("_get_type_guid")();
                if (!typeGuid.is_none()) {
                    m_typeGuid = typeGuid.cast<std::string>();
                }
            }

            // Inject component ID (generated in Component constructor)
            m_pyComponent.attr("_component_id") = py::int_(m_componentId);

            SyncEnabledFromPython(m_pyComponent, m_enabled, "constructor");

            if (py::hasattr(m_pyComponent, "_script_guid")) {
                py::object guidAttr = m_pyComponent.attr("_script_guid");
                if (!guidAttr.is_none()) {
                    m_scriptGuid = guidAttr.cast<std::string>();
                }
            }
        } catch (const py::error_already_set &e) {
            INFLOG_ERROR("[PyComponentProxy] Failed to get type name: ", e.what());
        }
    }
}

PyComponentProxy::~PyComponentProxy()
{
    // Note: OnDestroy is called explicitly before destruction by GameObject
    // Clear reference to allow Python GC
    m_pyComponent = py::none();
}

PyComponentProxy::PyComponentProxy(PyComponentProxy &&other) noexcept
    : Component(std::move(other)), m_pyComponent(std::move(other.m_pyComponent)),
      m_typeName(std::move(other.m_typeName)), m_typeGuid(std::move(other.m_typeGuid))
{
    other.m_pyComponent = py::none();
}

PyComponentProxy &PyComponentProxy::operator=(PyComponentProxy &&other) noexcept
{
    if (this != &other) {
        Component::operator=(std::move(other));
        m_pyComponent = std::move(other.m_pyComponent);
        m_typeName = std::move(other.m_typeName);
        m_typeGuid = std::move(other.m_typeGuid);
        other.m_pyComponent = py::none();
    }
    return *this;
}

void PyComponentProxy::BindPythonMirror()
{
    if (m_pyComponent.is_none())
        return;

    BindPythonMirrorHelpers(m_pyComponent, static_cast<Component *>(this), m_gameObject);
    try {
        m_pyComponent.attr("_execute_in_edit_mode") = py::bool_(m_executeInEditMode);
    } catch (const py::error_already_set &e) {
        INFLOG_WARN("[PyComponentProxy] Failed to set _execute_in_edit_mode on '", m_typeName, "': ", e.what());
    }
}

void PyComponentProxy::SyncPythonMirror() const
{
    if (m_pyComponent.is_none())
        return;

    SyncPythonMirrorState(m_pyComponent, this);
}

void PyComponentProxy::Awake()
{
    if (m_pyComponent.is_none())
        return;

    try {
        BindPythonMirror();
        SyncEnabledFromPython(m_pyComponent, m_enabled, "Awake");

        // Call Python awake
        CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_awake", "awake");
        SyncPythonMirror();
    } catch (const py::error_already_set &e) {
        INFLOG_ERROR("[PyComponentProxy] Error in ", m_typeName, ".awake setup: ", e.what());
    }
}

void PyComponentProxy::OnEnable()
{
    if (m_pyComponent.is_none())
        return;

    SyncPythonMirror();
    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_on_enable", "on_enable");
}

void PyComponentProxy::Start()
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_start", "start");
    SyncPythonMirror();
}

void PyComponentProxy::Update(float deltaTime)
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleFloatArg(m_pyComponent, m_typeName, "_call_update", "update", deltaTime);
}

void PyComponentProxy::FixedUpdate(float fixedDeltaTime)
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleFloatArg(m_pyComponent, m_typeName, "_call_fixed_update", "fixed_update", fixedDeltaTime);
}

void PyComponentProxy::LateUpdate(float deltaTime)
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleFloatArg(m_pyComponent, m_typeName, "_call_late_update", "late_update", deltaTime);
}

void PyComponentProxy::OnDisable()
{
    if (m_pyComponent.is_none())
        return;

    SyncPythonMirror();
    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_on_disable", "on_disable");
}

void PyComponentProxy::OnDestroy()
{
    if (m_pyComponent.is_none())
        return;

    SyncPythonMirror();
    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_on_destroy", "on_destroy");
}

void PyComponentProxy::OnValidate()
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_on_validate", "on_validate");
}

void PyComponentProxy::Reset()
{
    if (m_pyComponent.is_none())
        return;

    CallPythonLifecycleNoArg(m_pyComponent, m_typeName, "_call_reset", "reset");
}

// ========================================================================
// Physics callbacks (Unity-style) — forwarded to Python
// ========================================================================

void PyComponentProxy::OnCollisionEnter(const CollisionInfo &collision)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_collision_enter", "on_collision_enter",
                              py::cast(collision));
}

void PyComponentProxy::OnCollisionStay(const CollisionInfo &collision)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_collision_stay", "on_collision_stay",
                              py::cast(collision));
}

void PyComponentProxy::OnCollisionExit(const CollisionInfo &collision)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_collision_exit", "on_collision_exit",
                              py::cast(collision));
}

void PyComponentProxy::OnTriggerEnter(Collider *other)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_trigger_enter", "on_trigger_enter",
                              py::cast(other, py::return_value_policy::reference));
}

void PyComponentProxy::OnTriggerStay(Collider *other)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_trigger_stay", "on_trigger_stay",
                              py::cast(other, py::return_value_policy::reference));
}

void PyComponentProxy::OnTriggerExit(Collider *other)
{
    if (m_pyComponent.is_none())
        return;
    CallPythonLifecycleOneArg(m_pyComponent, m_typeName, "_call_on_trigger_exit", "on_trigger_exit",
                              py::cast(other, py::return_value_policy::reference));
}

const char *PyComponentProxy::GetTypeName() const
{
    return m_typeName.c_str();
}

std::vector<std::string> PyComponentProxy::GetRequiredComponentTypes() const
{
    std::vector<std::string> result;
    if (m_pyComponent.is_none())
        return result;

    try {
        py::object pyType = m_pyComponent.attr("__class__");
        if (py::hasattr(pyType, "_require_components_")) {
            py::list reqList = pyType.attr("_require_components_").cast<py::list>();
            for (auto item : reqList) {
                // Each entry is either a string or a Python type with __name__
                if (py::isinstance<py::str>(item)) {
                    result.push_back(item.cast<std::string>());
                } else if (py::hasattr(item, "__name__")) {
                    result.push_back(item.attr("__name__").cast<std::string>());
                }
            }
        }
    } catch (const py::error_already_set &e) {
        INFLOG_WARN("[PyComponentProxy] Failed to get required components for '", m_typeName, "': ", e.what());
    }
    return result;
}

std::string PyComponentProxy::Serialize() const
{
    json j;
    j["schema_version"] = 1;
    j["type"] = "PyComponentProxy";
    j["py_type_name"] = m_typeName;
    j["type_guid"] = m_typeGuid; // Stable type GUID for deserialization
    j["execution_order"] = GetExecutionOrder();
    bool enabled = m_enabled;
    if (!m_pyComponent.is_none()) {
        SyncEnabledFromPython(m_pyComponent, enabled, "Serialize");
    }
    j["enabled"] = enabled;
    j["component_id"] = m_componentId;
    j["script_guid"] = m_scriptGuid;

    // Serialize Python component's serializable fields
    if (!m_pyComponent.is_none()) {
        try {
            // Call Python side serialization if available
            if (py::hasattr(m_pyComponent, "_serialize_fields")) {
                py::object fieldsJson = m_pyComponent.attr("_serialize_fields")();
                if (!fieldsJson.is_none()) {
                    std::string fieldsStr = fieldsJson.cast<std::string>();
                    j["py_fields"] = json::parse(fieldsStr);
                }
            }
        } catch (const py::error_already_set &e) {
            INFLOG_ERROR("[PyComponentProxy] Error serializing fields: ", e.what());
        }
    }

    return j.dump(2);
}

bool PyComponentProxy::Deserialize(const std::string &jsonStr)
{
    try {
        json j = json::parse(jsonStr);

        // Base class deserialize
        Component::Deserialize(jsonStr);

        if (j.contains("py_type_name")) {
            m_typeName = j["py_type_name"].get<std::string>();
        }
        if (j.contains("script_guid")) {
            m_scriptGuid = j["script_guid"].get<std::string>();
        }

        // Python component and fields will be restored by Python side
        // after the C++ scene structure is rebuilt

        return true;
    } catch (const std::exception &e) {
        INFLOG_ERROR("[PyComponentProxy] Error deserializing: ", e.what());
        return false;
    }
}

void PyComponentProxy::SetScriptGuid(const std::string &guid)
{
    m_scriptGuid = guid;
    if (!m_pyComponent.is_none()) {
        try {
            m_pyComponent.attr("_script_guid") = py::str(guid);
        } catch (const py::error_already_set &e) {
            INFLOG_ERROR("[PyComponentProxy] Failed to set script guid: ", e.what());
        }
    }
}

} // namespace infengine
