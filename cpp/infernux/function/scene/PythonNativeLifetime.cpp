#include "PythonNativeLifetime.h"

namespace py = pybind11;

namespace infernux
{
namespace
{
class PythonNativeObserver final : public NativeLifetimeObserver
{
  public:
    explicit PythonNativeObserver(NativeLifetimeOwner &owner, const py::detail::type_info *type)
        : m_owner(&owner), m_type(type)
    {
    }

    void Attach(py::detail::instance *instance, const std::shared_ptr<PythonNativeObserver> &self)
    {
        m_reference = py::weakref(py::handle(reinterpret_cast<PyObject *>(instance)),
                                  py::cpp_function([weak = std::weak_ptr<PythonNativeObserver>(self)](py::handle) {
                                      if (auto observer = weak.lock()) {
                                          if (NativeLifetimeOwner *owner = std::exchange(observer->m_owner, nullptr))
                                              owner->ForgetNativeLifetimeObserver(observer.get());
                                      }
                                  }));
    }

    ~PythonNativeObserver() override
    {
        // Normal shutdown releases Scenes before CPython. Do not call CPython
        // if an embedding host instead finalizes it before its native objects.
        if (!Py_IsInitialized())
            m_reference.release();
        else if (m_reference) {
            py::gil_scoped_acquire acquire;
            m_reference = py::weakref{};
        }
    }

    void RetireNativeObject() noexcept override
    {
        m_owner = nullptr;
        if (!Py_IsInitialized())
            return;
        py::gil_scoped_acquire acquire;
        // The native owner may destroy this observer after releasing the GIL.
        // Clear its Python state within this scope, including early returns.
        py::weakref reference = std::move(m_reference);
        // A Python-owned holder invokes the native destructor during tp_dealloc,
        // after pybind has deregistered it. Never resurrect that zero-ref object
        // or mutate the holder while its own destructor is executing.
        PyObject *value = PyWeakref_GetObject(reference.ptr());
        if (value == Py_None || Py_REFCNT(value) == 0)
            return;
        auto *instance = reinterpret_cast<py::detail::instance *>(value);
        auto entry = instance->get_value_and_holder(m_type);
        if (entry.instance_registered()) {
            if (!py::detail::deregister_instance(instance, entry.value_ptr(), entry.type))
                std::terminate(); // Broken binding ownership, never continue with an aliased cache entry.
            entry.set_instance_registered(false);
        }
        auto &holder = entry.holder<py::smart_holder>();
        if (holder.has_pointee()) {
            if (holder.vptr_is_using_std_default_delete)
                holder.disown(entry.type->get_memory_guarded_delete);
            holder.release_disowned();
        }
        // smart_holder remains constructed, so generic pybind callers reject it
        // before the null value_ptr can enter pybind's lazy allocation path.
        entry.value_ptr() = nullptr;
    }

  private:
    NativeLifetimeOwner *m_owner;
    const py::detail::type_info *m_type;
    py::weakref m_reference;
};
} // namespace

void ObservePythonNativeLifetime(NativeLifetimeOwner &owner, py::detail::instance *instance,
                                 const py::detail::type_info *type)
{
    auto observer = std::make_shared<PythonNativeObserver>(owner, type);
    observer->Attach(instance, observer);
    owner.ObserveNativeLifetime(std::move(observer));
}
} // namespace infernux
