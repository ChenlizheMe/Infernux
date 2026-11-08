# String tween compute and display boundaries

`StringTween.cpp` is a Foundation kernel with no Python, renderer or device
dependency. Its immutable snapshots compile UTF-16 text and tag positions once;
bound samplers own easing, duration, options and custom alphabet references.
The same kernel runs on Windows, Linux, Android and WebAssembly. Web uses the
Player's Wasm exception ABI, including recoverable invalid-alphabet evaluations.

The Python binding chooses a literal ASCII specialization at bind time when both
strings are ASCII, rich/relative/scrambling are disabled and the plan is below the
large-job threshold. It uses the same native rounding/range calculation but writes
the final Python Unicode object directly through CPython's C API. This avoids a
temporary UTF-16 result and a second copy/decode. It is an implementation of the
same plan contract, with no runtime backend retry or Python character loop.

Binding owners retain immutable Python strings. Their Unicode accesses and
refcounts require the GIL. General native snapshots contain no Python objects;
large native construction/evaluation releases the GIL once, before reacquiring it
to create the Python result. Ordinary text stays on the calling thread. The
1,048,576 estimated UTF-16-unit threshold differs from heavy numerical sampling:
512-unit strings showed millisecond GIL reacquisition outliers under contention.
This is a measured policy, not a claim of optimal thresholds on every machine.

Scrambling shares the gameplay Random state and previous offset. Its alphabet
initialization/shuffle/selection uses one `Random::Scope`, with no Python access
under that lock. Output copying follows after unlocking. Sampling order remains
observable; reordering random jobs or resetting their statics changes behavior.

UTF-16 reveal can produce a lone surrogate, including halfway through an emoji.
The tween and Python `UIText.text` property preserve that value. `TextForDisplay`
converts only the glyph input to valid UTF-8: unpaired units become U+FFFD, valid
surrogate pairs join, and ordinary strings use cached UTF-8. Layout source byte
offsets refer to `TextLayout.text`, the normalized presentation input. Do not use
those offsets as UTF-16 indices into the original Python property.

Validation covers UTF-16 arrays, shared RNG state, clock and lifetime transitions,
getter timing and recoverable errors. The standalone `string_tween_probe`
accepts a numeric corpus as its first argument. Local sampling tools and
provenance records are kept outside the published source.
Run the frontend worker in a fresh process to retain global scrambling order.
The engine's text-layout tests independently cover partial-surrogate presentation.
Malformed markup outside that matrix and full mobile/Web Player integration
remain separate acceptance work.
