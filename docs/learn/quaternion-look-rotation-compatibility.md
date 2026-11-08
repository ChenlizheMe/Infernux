# Public LookRotation components

`quatf.look_rotation` delegates to QuaternionLookRotation.cpp in Foundation.
The implementation uses native arrays and owns no Python objects, locks or
renderer resources. A small scalar call retains the caller's GIL.

Construction from an orthonormal basis uses the measured trace/diagonal branch,
including tie ordering. A generic largest-component quaternion conversion can
return the opposite public components even when its orientation is equivalent.
Global positive-w canonicalization would also be incorrect: valid reference
outputs include negative w.

Forward length below 1e-5 produces identity. A collinear/zero up direction uses
the shortest rotation from +Z, with the measured fixed X axis near -Z. atan2
keeps the nearly opposite case stable before basis/component construction.
Camera/view basis conventions are a separate API contract.

The plugin's own QuaternionLookRotationReference.cs records 1,640 actual Unity
2022.3.62f3c1 observations, including 1,024 independent random direction pairs,
axes, small/scaled directions, near-collinearity and opposite-direction edges.
Compare public x/y/z/w components directly at absolute 0.000002. The standalone
quaternion_look_rotation_probe consumes this fixture as its first argument.
Windows, Linux, Android arm64 device and WebAssembly results pass; maximum
observed error is 2.384185791015625e-7. Actual Linux binding checks pass separately.

The subsequent hierarchy conversion, planar endpoint stores and public quaternion
product corrections close the measured live path failures: all 872 configurations
/ 9,592 states pass their original component tolerances. The SDK and four native
targets also pass 23,760 exact forward/direction components and 8,192 exact public
product components. Inverse scale rounding remains a separate open prerequisite;
these results do not establish every Transform operation or possible path input.
