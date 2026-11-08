#pragma once

#include "core/FoundationApi.h"

#include "TweenNumerics.h"
#include <memory>
#include <string>

namespace infernux::tween
{
struct StringData;

struct StringSlices
{
    size_t shown, startBegin, startCount;
};

class StringAlphabet
{
  public:
    INFERNUX_FOUNDATION_API explicit StringAlphabet(std::u16string text, bool shuffle = true);
    const std::u16string &Text() const
    {
        return text_;
    }

  private:
    std::u16string text_;
};

class StringSampler
{
  public:
    StringSampler(std::shared_ptr<const StringData> data, EaseKernel ease, double duration, bool rich, bool relative,
                  int mode, std::shared_ptr<const StringAlphabet> alphabet, int startLength = -1,
                  int changeLength = -1);
    INFERNUX_FOUNDATION_API std::u16string Sample(double progress, int cycles) const;
    INFERNUX_FOUNDATION_API std::u16string Interpolate(double factor, int cycles) const;
    double Factor(double progress) const
    {
        return ease_.Sample(progress, duration_);
    }
    INFERNUX_FOUNDATION_API StringSlices PlainSlices(double factor) const;
    INFERNUX_FOUNDATION_API size_t WorkSize(int cycles) const;

  private:
    std::shared_ptr<const StringData> data_;
    EaseKernel ease_;
    double duration_;
    bool rich_, relative_;
    int mode_;
    size_t startLength_, changeLength_;
    std::shared_ptr<const StringAlphabet> alphabet_;
};

class StringSnapshot
{
  public:
    INFERNUX_FOUNDATION_API StringSnapshot(std::u16string start, std::u16string end);
    INFERNUX_FOUNDATION_API size_t Length(bool rich) const;
    INFERNUX_FOUNDATION_API size_t StartLength(bool rich) const;
    INFERNUX_FOUNDATION_API StringSampler Bind(EaseKernel ease, double duration, bool rich, bool relative, int mode,
                                               std::shared_ptr<const StringAlphabet> alphabet, int startLength = -1,
                                               int changeLength = -1) const;

  private:
    std::shared_ptr<const StringData> data_;
};
} // namespace infernux::tween
