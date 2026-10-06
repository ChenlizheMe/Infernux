#pragma once

#include "NativeLifetime.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <type_traits>

namespace infernux
{
class InvalidNativeObjectError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

void ObservePythonNativeLifetime(NativeLifetimeOwner &owner, pybind11::detail::instance *instance,
                                 const pybind11::detail::type_info *type);

inline bool PythonNativeObjectIsAlive(pybind11::handle value)
{
    auto *instance = reinterpret_cast<pybind11::detail::instance *>(value.ptr());
    bool found = false;
    for (auto &entry : pybind11::detail::values_and_holders(instance)) {
        if (entry.type->holder_enum_v != pybind11::detail::holder_enum_t::smart_holder)
            continue;
        found = true;
        if (!entry.holder_constructed() || !entry.holder<pybind11::smart_holder>().has_pointee())
            return false;
    }
    return found;
}
} // namespace infernux

namespace pybind11::detail
{
// Retain the original wrapper until invocation: converting a later argument
// can run Python and retire an already loaded native argument.
template <typename T>
class type_caster<T, enable_if_t<std::is_base_of_v<infernux::NativeLifetimeOwner, T>>> : public type_caster_base<T>
{
    object m_source;

    void RequireAlive() const
    {
        if (m_source && !infernux::PythonNativeObjectIsAlive(m_source))
            throw infernux::InvalidNativeObjectError("Native object has been destroyed or is not initialized");
    }

  public:
    bool load(handle src, bool convert)
    {
        m_source = object();
        if (!src || !this->typeinfo)
            return false;
        if (!src.is_none()) {
            if (PyObject_TypeCheck(src.ptr(), this->typeinfo->type))
                m_source = reinterpret_borrow<object>(src);
            else {
                // Public Python components expose this explicit binding
                // protocol. Capture the actual wrapper before loading a pointer;
                // a conduit returning only an address loses lifetime identity.
                object resolve = getattr(src, "_get_bound_native_component", none());
                if (!resolve || !PyCallable_Check(resolve.ptr()))
                    return false;
                m_source = resolve();
                if (m_source.is_none())
                    throw infernux::InvalidNativeObjectError("Python component is not bound to a live native object");
                if (!PyObject_TypeCheck(m_source.ptr(), this->typeinfo->type))
                    return false;
            }
            RequireAlive();
            return type_caster_base<T>::load(m_source, convert);
        }
        return type_caster_base<T>::load(src, convert);
    }

    operator T *()
    {
        RequireAlive();
        return type_caster_base<T>::operator T *();
    }

    operator T &()
    {
        RequireAlive();
        return type_caster_base<T>::operator T &();
    }
};

// Keep element casters alive through all argument conversions. The standard
// list caster discards them during load, leaving only unchecked raw pointers.
template <typename T, typename Alloc> class NativePointerListCaster
{
    using Vector = std::vector<T *, Alloc>;
    using ListCaster = list_caster<Vector, T *>;
    std::vector<type_caster<T>> m_elements;
    Vector m_value;

    void Resolve()
    {
        m_value.clear();
        m_value.reserve(m_elements.size());
        for (auto &element : m_elements)
            m_value.push_back(static_cast<T *>(element));
    }

  public:
    static constexpr auto name = ListCaster::name;
    template <typename U> using cast_op_type = movable_cast_op_type<U>;

    bool load(handle src, bool convert)
    {
        if (!object_is_convertible_to_std_vector(src))
            return false;
        object sequence;
        if (isinstance<pybind11::sequence>(src))
            sequence = reinterpret_borrow<object>(src);
        else if (convert)
            sequence = tuple(reinterpret_borrow<iterable>(src));
        else
            return false;
        m_elements.clear();
        m_elements.reserve(len(sequence));
        for (handle item : sequence) {
            m_elements.emplace_back();
            if (!m_elements.back().load(item, convert))
                return false;
        }
        return true;
    }

    operator Vector *()
    {
        Resolve();
        return &m_value;
    }
    operator Vector &()
    {
        Resolve();
        return m_value;
    }
    operator Vector &&() &&
    {
        Resolve();
        return std::move(m_value);
    }

    template <typename U> static handle cast(U &&src, return_value_policy policy, handle parent)
    {
        return ListCaster::cast(std::forward<U>(src), policy, parent);
    }
};

template <typename T, typename Alloc>
class type_caster<std::vector<T *, Alloc>>
    : public conditional_t<std::is_base_of_v<infernux::NativeLifetimeOwner, T>, NativePointerListCaster<T, Alloc>,
                           list_caster<std::vector<T *, Alloc>, T *>>
{
};
} // namespace pybind11::detail

namespace infernux
{
/// Registers one native entity type with pybind's explicit empty-holder state.
/// New wrappers are observed at initialization, before any user code sees them.
template <typename T, typename... Bases>
class NativeClass : public pybind11::class_<T, pybind11::smart_holder, Bases...>
{
    using Base = pybind11::class_<T, pybind11::smart_holder, Bases...>;
    using Initializer = void (*)(pybind11::detail::instance *, const void *);
    inline static Initializer s_initialize = nullptr;

    static void Initialize(pybind11::detail::instance *instance, const void *holder)
    {
        s_initialize(instance, holder);
        auto *type = pybind11::detail::get_type_info(typeid(T));
        auto entry = instance->get_value_and_holder(type);
        ObservePythonNativeLifetime(*entry.template value_ptr<T>(), instance, type);
    }

  public:
    template <typename... Extras>
    NativeClass(pybind11::handle scope, const char *name, const Extras &...extras) : Base(scope, name, extras...)
    {
        auto *type = pybind11::detail::get_type_info(typeid(T));
        s_initialize = type->init_instance;
        type->init_instance = &Initialize;
        this->def("__bool__", [type](pybind11::handle self) {
            if (!PyObject_TypeCheck(self.ptr(), type->type))
                throw pybind11::type_error("Native liveness requires an instance of the bound type");
            return PythonNativeObjectIsAlive(self);
        });
    }
};
} // namespace infernux
