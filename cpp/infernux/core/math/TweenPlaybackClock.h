#pragma once

#include "TweenNumerics.h"

namespace infernux::tween
{
// Per-instance playback state. Immutable timing recipes remain shareable;
// current/previous phases never retain Python owners or callbacks.
class PlaybackClock
{
  public:
    PlaybackClock(float duration, float delay, int64_t loops, double position) : recipe_(duration, delay, loops)
    {
        Seek(position);
        previous_ = current_;
    }

    void Seek(double position)
    {
        previous_ = current_;
        State state;
        state.time = position;
        state.elapsedDelay = static_cast<float>(std::min<double>(position, recipe_.delay));
        state.delayComplete = position >= recipe_.delay || position < 0;
        const float elapsed =
            static_cast<float>(position < 0 ? position - recipe_.delay : std::max(0.0, position - recipe_.delay));
        if (recipe_.IsTerminal(position)) {
            state.completed = recipe_.loops;
            state.position = recipe_.duration;
        } else {
            const double count = std::floor(static_cast<float>(elapsed / recipe_.duration));
            if (!(count >= -9223372036854775808.0 && count < 9223372036854775808.0))
                throw NumericOverflow("Tween loop count exceeds its native integer range");
            state.completed = static_cast<int64_t>(count);
            state.position = std::fmod(elapsed, recipe_.duration);
            if (recipe_.loops != -1 && state.completed >= recipe_.loops) {
                state.completed = recipe_.loops;
                state.position = recipe_.duration;
            } else if (state.position <= 0) {
                state.position = state.completed > 0 ? recipe_.duration : 0;
            }
        }
        current_ = state;
    }

    bool Advance(float delta, float scale, bool backwards)
    {
        float step;
        if (!BeginStep(delta, scale, step))
            return false;
        AdvanceStep(step, backwards);
        return true;
    }

    bool BeginStep(float delta, float scale, float &step)
    {
        step = delta * scale;
        if (step > -1e-6f && step < 1e-6f)
            return false;
        if (!std::isfinite(step))
            throw NumericOverflow("Tween frame step is not finite");
        if (!current_.delayComplete) {
            current_.elapsedDelay += step;
            if (current_.elapsedDelay <= recipe_.delay) {
                current_.time = current_.elapsedDelay;
                return false;
            }
            step = current_.elapsedDelay - recipe_.delay;
            current_.elapsedDelay = recipe_.delay;
            current_.delayComplete = true;
            current_.time = static_cast<double>(recipe_.delay) + ElapsedDouble(current_);
        }
        return true;
    }

    void AdvanceStep(float step, bool backwards)
    {
        previous_ = current_;
        const bool wasEnd = current_.position >= recipe_.duration;
        if (backwards) {
            current_.position -= step;
            while (current_.position < 0 && current_.completed > -1) {
                current_.position += recipe_.duration;
                --current_.completed;
            }
            if (current_.completed < 0 || (wasEnd && current_.completed < 1)) {
                current_.position = 0;
                current_.completed = wasEnd ? 1 : 0;
            }
        } else {
            current_.position += step;
            while (current_.position >= recipe_.duration &&
                   (recipe_.loops == -1 || current_.completed - static_cast<int64_t>(wasEnd) < recipe_.loops)) {
                current_.position -= recipe_.duration;
                ++current_.completed;
            }
        }
        if (wasEnd)
            --current_.completed;
        if (Complete(current_) || current_.position > recipe_.duration) {
            current_.position = recipe_.duration;
        } else if (current_.position <= 0) {
            current_.position = current_.completed > 0 ? recipe_.duration : 0;
        }
        current_.time =
            Complete(current_) ? recipe_.fullDuration : static_cast<double>(recipe_.delay) + ElapsedDouble(current_);
    }

    double BeginCallbacks()
    {
        callbackStep_ = current_;
        current_ = previous_;
        return current_.time;
    }

    double FinishCallbacks(PlaybackClock &destination)
    {
        destination.previous_ = destination.current_;
        State &state = destination.current_;
        state.completed = callbackStep_.completed;
        state.position = std::min(callbackStep_.position, destination.recipe_.duration);
        state.time = destination.Complete(state)
                         ? destination.recipe_.fullDuration
                         : static_cast<double>(destination.recipe_.delay) + destination.ElapsedDouble(state);
        return state.time;
    }

    double Time() const
    {
        return current_.time;
    }
    double Position() const
    {
        return current_.position;
    }
    int64_t CompletedLoops() const
    {
        return current_.completed;
    }
    bool IsComplete() const
    {
        return Complete(current_);
    }
    bool WasComplete() const
    {
        return Complete(previous_);
    }
    double ElapsedDelay() const
    {
        return current_.elapsedDelay;
    }
    double Elapsed() const
    {
        const float cycles =
            recipe_.duration * static_cast<float>(current_.completed - (current_.position >= recipe_.duration ? 1 : 0));
        return static_cast<float>(cycles + current_.position);
    }

    int64_t CompletedAt(double position) const
    {
        if (position == current_.time)
            return current_.completed;
        if (position == previous_.time)
            return previous_.completed;
        const double elapsed = position < 0 ? position - recipe_.delay : std::max(0.0, position - recipe_.delay);
        return static_cast<int64_t>(std::floor(elapsed / recipe_.duration));
    }

    bool CompleteAt(double position) const
    {
        if (position == current_.time)
            return Complete(current_);
        if (position == previous_.time)
            return Complete(previous_);
        return recipe_.loops != -1 && position >= recipe_.fullDuration;
    }

    ProgressSample Sample(double position, bool yoyo, bool inverted) const
    {
        const State *state = position == current_.time ? &current_ : position == previous_.time ? &previous_ : nullptr;
        if (!state)
            return recipe_.IsTerminal(position) ? ProgressSample{static_cast<double>(recipe_.loops - 1),
                                                                 recipe_.TerminalProgress(yoyo, inverted), false}
                                                : recipe_.SampleInterior(position, yoyo, inverted);
        const bool end = state->position >= recipe_.duration;
        const int64_t cycle = state->completed - (end ? 1 : 0);
        double progress = static_cast<double>(state->position) / recipe_.duration;
        if (yoyo && recipe_.hasLoops && cycle % 2)
            progress = 1 - progress;
        if (inverted)
            progress = 1 - progress;
        return {static_cast<double>(cycle), progress, false};
    }

  private:
    struct State
    {
        double time = 0;
        float position = 0, elapsedDelay = 0;
        int64_t completed = 0;
        bool delayComplete = false;
    };

    bool Complete(const State &state) const
    {
        return recipe_.loops != -1 && state.completed >= recipe_.loops;
    }
    double ElapsedDouble(const State &state) const
    {
        const int64_t cycle = state.completed - (state.position >= recipe_.duration ? 1 : 0);
        return static_cast<double>(recipe_.duration) * static_cast<double>(cycle) + state.position;
    }

    const PreparedTweenClock recipe_;
    State current_, previous_, callbackStep_;
};
} // namespace infernux::tween
