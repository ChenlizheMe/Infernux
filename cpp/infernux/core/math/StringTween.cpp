#include "StringTween.h"
#include "Random.h"

namespace infernux::tween
{
namespace
{
struct Tag
{
    size_t begin, end;
    char16_t key;
};

struct Text
{
    explicit Text(std::u16string value) : value(std::move(value))
    {
        stripped = this->value.size();
        size_t newline = this->value.find(u'\n');
        for (size_t i = 0; i < this->value.size(); ++i) {
            if (this->value[i] != u'<')
                continue;
            const auto end = this->value.find(u'>', i + 1);
            if (end == std::u16string::npos)
                break;
            stripped -= end + 1 - i;
            while (newline < i)
                newline = this->value.find(u'\n', newline + 1);
            if (newline > end) {
                const bool close = i + 1 < end && this->value[i + 1] == u'/';
                const char16_t key = i + (close ? 2 : 1) < this->value.size() ? this->value[i + (close ? 2 : 1)] : 0;
                tags.push_back({i, end + 1, key == u'#' ? u'c' : key});
                if (close)
                    closing.push_back(tags.back());
            }
            i = end;
        }
        if (this->value.size() > 3 && this->value.back() == u'>') {
            const auto last = this->value.rfind(u'<', this->value.size() - 3);
            if (last != std::u16string::npos && this->value[last + 1] != u'/')
                ++stripped;
        }
    }

    size_t Length(bool rich) const
    {
        return rich ? stripped : value.size();
    }

    void Append(std::u16string &output, size_t begin, size_t count, bool rich) const
    {
        if (!rich) {
            output.append(value, begin, count);
            return;
        }
        std::vector<char16_t> open;
        size_t cursor = 0, limit = count, first = begin, tagIndex = 0;
        while (cursor < limit && cursor < value.size()) {
            if (value[cursor] != u'<') {
                const auto end = std::min({value.find(u'<', cursor), limit, value.size()});
                const auto start = std::max(cursor, first);
                if (start < end)
                    output.append(value, start, end - start);
                cursor = end;
                continue;
            }
            if (cursor + 1 >= value.size())
                throw std::out_of_range("unfinished rich-text tag");
            const bool closes = value[cursor + 1] == u'/';
            if (closes) {
                if (open.empty())
                    throw std::out_of_range("rich-text closing tag has no opening tag");
                open.pop_back();
            } else
                open.push_back(value[cursor + 1] == u'#' ? u'c' : value[cursor + 1]);
            while (tagIndex < tags.size() && tags[tagIndex].begin < cursor)
                ++tagIndex;
            if (tagIndex == tags.size()) {
                ++cursor;
                continue;
            }
            const auto &tag = tags[tagIndex++];
            output.append(value, tag.begin, tag.end - tag.begin);
            const auto advance = tag.end - cursor;
            limit += advance;
            first += advance;
            cursor = tag.end;
        }
        while (!open.empty() && cursor + 1 < value.size()) {
            const auto tag =
                std::lower_bound(closing.begin(), closing.end(), cursor,
                                 [](const Tag &candidate, size_t position) { return candidate.begin < position; });
            if (tag == closing.end())
                break;
            if (tag->key == open.back()) {
                output.append(value, tag->begin, tag->end - tag->begin);
                open.pop_back();
            }
            cursor += tag->end - tag->begin;
        }
    }

    std::u16string value;
    size_t stripped;
    std::vector<Tag> tags, closing;
};

struct Tables
{
    std::array<std::u16string, 4> text{u"ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvxyz1234567890",
                                       u"ABCDEFGHIJKLMNOPQRSTUVXYZ", u"abcdefghijklmnopqrstuvxyz", u"1234567890"};
    bool initialized = false;
    int last = 0;
    void Initialize(Random::Scope &random)
    {
        if (initialized)
            return;
        for (auto &alphabet : text)
            random.Shuffle(alphabet.data(), alphabet.size());
        initialized = true;
    }
};

Tables &SharedTables()
{
    static Tables tables;
    return tables;
}

size_t ShownCount(size_t length, double factor)
{
    const double rounded = std::nearbyint(double(length) * static_cast<float>(factor));
    return rounded > 0 ? static_cast<size_t>(std::min(rounded, double(length))) : 0;
}

size_t RemainingCount(size_t startLength, size_t length, size_t shown)
{
    if (shown >= length || shown >= startLength)
        return 0;
    if (startLength > length) {
        const float percentage = static_cast<float>(shown) / static_cast<float>(length);
        return startLength - static_cast<size_t>(double(startLength) * percentage);
    }
    return startLength - shown;
}
} // namespace

struct StringData
{
    StringData(std::u16string start, std::u16string end) : start(std::move(start)), end(std::move(end))
    {
    }
    Text start, end;
};

StringAlphabet::StringAlphabet(std::u16string text, bool shuffle) : text_(std::move(text))
{
    if (!shuffle)
        return;
    if (text_.size() == 1)
        text_ += text_;
    Random::Scope random;
    SharedTables().Initialize(random);
    random.Shuffle(text_.data(), text_.size());
}

StringSnapshot::StringSnapshot(std::u16string start, std::u16string end)
    : data_(std::make_shared<const StringData>(std::move(start), std::move(end)))
{
}
size_t StringSnapshot::Length(bool rich) const
{
    return data_->end.Length(rich);
}
size_t StringSnapshot::StartLength(bool rich) const
{
    return data_->start.Length(rich);
}
StringSampler StringSnapshot::Bind(EaseKernel ease, double duration, bool rich, bool relative, int mode,
                                   std::shared_ptr<const StringAlphabet> alphabet, int startLength,
                                   int changeLength) const
{
    return StringSampler(data_, std::move(ease), duration, rich, relative, mode, std::move(alphabet), startLength,
                         changeLength);
}

StringSampler::StringSampler(std::shared_ptr<const StringData> data, EaseKernel ease, double duration, bool rich,
                             bool relative, int mode, std::shared_ptr<const StringAlphabet> alphabet, int startLength,
                             int changeLength)
    : data_(std::move(data)), ease_(std::move(ease)), duration_(duration), rich_(rich), relative_(relative),
      mode_(mode), startLength_(rich && startLength >= 0 ? size_t(startLength) : data_->start.Length(rich)),
      changeLength_(rich && changeLength >= 0 ? size_t(changeLength) : data_->end.Length(rich)),
      alphabet_(std::move(alphabet))
{
}

size_t StringSampler::WorkSize(int cycles) const
{
    return data_->start.value.size() + data_->end.value.size() * (relative_ && cycles > 0 ? size_t(cycles) + 1 : 1);
}

std::u16string StringSampler::Sample(double progress, int cycles) const
{
    return Interpolate(Factor(progress), cycles);
}

StringSlices StringSampler::PlainSlices(double factor) const
{
    const auto length = data_->end.value.size();
    const auto shown = ShownCount(length, factor);
    const auto remaining = RemainingCount(data_->start.value.size(), length, shown);
    return {shown, shown, std::min(remaining, data_->start.value.size() - std::min(shown, data_->start.value.size()))};
}

std::u16string StringSampler::Interpolate(double factor, int cycles) const
{
    const auto length = changeLength_;
    const auto shown = ShownCount(length, factor);
    std::u16string result;
    if (relative_) {
        result += data_->start.value;
        for (int i = 0; i < cycles; ++i)
            result += data_->end.value;
    }
    data_->end.Append(result, 0, shown, rich_);
    if (mode_ != 0) {
        const std::u16string *letters = nullptr;
        int offset = 0;
        {
            Random::Scope random;
            auto &tables = SharedTables();
            tables.Initialize(random);
            if (shown < length) {
                if (mode_ == 5) {
                    if (!alphabet_)
                        throw std::invalid_argument("custom scrambling has no alphabet");
                    letters = &alphabet_->Text();
                } else
                    letters = &tables.text[mode_ >= 1 && mode_ <= 4 ? mode_ - 1 : 0];
                if (letters->size() < 2)
                    throw std::invalid_argument("scrambling alphabet is empty");
                do {
                    offset = random.Range(0, static_cast<int>(letters->size()));
                } while (offset == tables.last);
                tables.last = offset;
            }
        }
        size_t remaining = length - shown;
        while (remaining) {
            const auto count = std::min(remaining, letters->size() - offset);
            result.append(*letters, size_t(offset), count);
            remaining -= count;
            offset = 0;
        }
    } else if (!relative_) {
        const auto startLength = startLength_;
        if (shown < length && shown < startLength) {
            const auto remaining = RemainingCount(startLength, length, shown);
            data_->start.Append(result, shown, rich_ ? shown + remaining : remaining, rich_);
        }
    }
    return result;
}
} // namespace infernux::tween
