# Quaternion to Euler conversion

The shared implementation is `cpp/infernux/core/math/QuaternionEuler.cpp`, owned
by InfernuxFoundation. The Python quaternion property, Transform conversion and
TransformECSStore use this implementation instead of three independent formulas.
Its small header exposes the conversion and public angle-range adapter. Numeric
implementation edits rebuild the Foundation translation unit rather than every
scene and binding consumer. No extra plugin binary or runtime compiler is needed.

The rotation order is intrinsic YXZ / extrinsic ZXY. Actual UnityEngine reference
measurements pin these details:

- Float normalization stores and addition order matter near singular poses.
- Input magnitudes below `1e-5` produce zero Euler angles. This is the measured
  native Euler conversion boundary; Unity's managed Quaternion.Normalize has a
  different small-input boundary and must not substitute for this behavior.
- The singular branch begins at `abs(x*w-y*z) >= 0.499999` after normalization.
  Pitch retains the clamped asin result, yaw uses twice atan2(y,w), and roll is
  zero. Forcing pitch to 90 degrees too early changes observable output.
- The ordinary branch preserves the measured denominator evaluation order.
  Algebraically equivalent rearrangements can amplify float rounding near the
  singularity. Keep intermediate stores separate; do not enable fast-math or
  reassociation for this contract.
- The public adapter retains tiny signed angles around zero and wraps once at
  the measured `-0.005729578` / `359.99427` boundary. It does not snap them to zero.

Transform's authoring continuity and multiple-turn caches remain separate. Their
public adapter reduces complete turns before applying the signed range rule.
Sharing the conversion does not establish that every authoring/cache convention
matches Unity's public Transform properties.

## Reference and verification

Our own `QuaternionEulerReference.cs` exporter lives in the Tween checkout under
`tests/tools`. Its fixture `tests/fixtures/quaternion_euler_unity_2022.json` records
Unity 2022.3.62f3c1 public inputs and outputs. Vendor implementation and binaries
are not redistributed. There are 2,362 cases: 1,320 grid configurations, 1,024
independent random rotations and 18 boundary inputs. Inputs include scaled and
negative quaternions, tiny signed angles, tiny/zero magnitude, large magnitude
and dense samples on either side of the singular branch.

`tests/native/quaternion_euler_probe.cpp` accepts that fixture as its first CLI
argument. Build the `infernux_quaternion_euler_tests` CMake target and run it with
the fixture path. Its default CTest entry checks analytic boundaries. The complete
fixture passes on Windows, Linux, an Android arm64 device and WebAssembly/Node.
The actual Windows/Linux Python binding also passes all public property values;
Android/Web binding translation units compile. Maximum observed absolute public
angle error is 0.000030518 degrees, below the unchanged 0.00005-degree tolerance.
The checks do not replace mismatches with modulo-angle or orientation equivalence.

The Tween checkout's `test_quaternion_euler_reference.py` is an automated SDK
property gate; `tests/tools/audit_quaternion_euler.py` emits a detailed report and
exits nonzero on differences. The live-options matrix's 2,502 configurations also
pass, including two speed-duration cases that originally exposed this prerequisite.
155 engine rotation, Transform batch, vector, camera and scene regressions pass.

The current physics comparisons pass 183 configurations / 2,121 fixed-step poses.
All 136 pose-order cases pass both paused stepping and real headless frames. A
separate installed-package check exercises moving kinematic contacts, simulation
pause/resume, independent clocks, settling and owner destruction. These checks
do not establish identical solvers or every contact/callback combination.
Full Android/Web Player behavior and universal equality for every Unity backend
or float input are outside what these measurements establish.

## Public normalization

`quatf.normalized` uses `rotation::NormalizePublic` in the same Foundation
translation unit. It is deliberately separate from Euler's internal normalization
and from Transform rotation storage. The Unity 2022 Mono reference keeps wider
intermediate precision in the dot product, stores the norm and magnitude as float,
then divides components individually. GLM's reciprocal-multiply implementation
produces different public components for some inputs. Zero magnitude returns
identity; there is no additional per-component validation or recovery path.

The existing 2,362-case fixture also records the public normalized copy. The
native probe and SDK test compare those four components exactly, independently of
the Euler tolerance. Windows, Linux, Android arm64 device and WebAssembly/Node
pass the same reference. The actual Linux Python binding also passes. This pins
the measured Mono contract and does not establish bit identity with every Unity
JIT/IL2CPP backend. Transform's native local/world setters have separately
observed rounding and must not be replaced with this public-property formula.

## Transform normalization

`rotation::NormalizeTransform` implements the separately measured native storage
contract. First round the four component squares to float. The x/z denominator
uses sqrt((xx+yy)+(zz+ww)); the y/w denominator uses sqrt((yy+zz)+(ww+xx)). Each
addition, sqrt and final division follows float rounding. Zero magnitude returns
identity. A single common denominator or reciprocal multiplication does not
reproduce these outputs. The pattern is consistent with cyclic lane reduction;
the source of Unity's closed native implementation is not inferred to be identical.

All 2,362 recorded Transform rotations match exactly on Windows, Linux (Clang
and GCC), Android arm64 device and WebAssembly. Actual SDK world/local setters
also match every component exactly. Euler authoring setters and batch/local-frame
Euler preparation use the same normalization once. Raw ECS storage and physics
publication remain separate. The local setter bodies live in Transform.cpp to
avoid rebuilding every consumer for an implementation edit.

The Tween checkout's `TransformPrecisionReference.BeginHierarchy` additionally
captures 20 actual Play-mode parent setups. Stored rotations and parent TRS
matrices pass exact comparisons for local/world setters and root/identity-parent
hierarchies. This does not prove inverse-matrix or point-conversion parity. The
remaining path orientation differences still expose those separate prerequisites.

## Hierarchy rotation storage

`rotation::MultiplyTransform` preserves the measured native float product/addition
order separately from the public quaternion operator. Native world assignment
applies each ancestor's conjugate from root to leaf, then normalizes the final
local value once. World readback multiplies local rotations from leaf to root.
Quaternion multiplication is not associative at float precision: pre-collapsing
the parent chain, or using a general inverse divided by squared magnitude, changes
observable local/world values.

`TransformRotationReference.Run` records 14,172 inputs across six parent setups,
including nested ancestors, nonunit inputs and repeated local/world assignments.
Every stored/readback component matches in the actual SDK and native probes on
Windows, Linux, Android arm64 device and WebAssembly. The reference is stored in
the Tween checkout as `transform_rotation_unity_2022.json.gz`.

Authored rotations now use the same local storage path during cached frames.
World batches preserve ancestor order in both modes. Physics can still publish
its explicit world pose, and a later authored local write supersedes that channel.
All 136 measured queue/edit/sync/step cases pass in paused fixed stepping and real
headless frames; the separate 118-case Transform/vector/rigidbody regression
passes. This does not establish complete path orientation or contact parity.

## Public quaternion-vector rotation and native conversion diagnostics

`RotatePublic` implements the separately measured managed Quaternion * Vector3
expression. The nine doubled-quaternion products have float stores; coefficient
addition and vector evaluation retain wider intermediates until the output float
store. It does not normalize the quaternion or vector. This is distinct from
native TransformDirection and from building a float rotation matrix first.

`TransformPointReference.Begin` records 21,258 raw quaternion/vector samples plus
1,980 stored-world-rotation samples over 60 parent/scale setups. All 23,238 public
operator outputs pass exact component comparisons in the actual Windows binding
and native Windows/Linux/Android-device/WebAssembly probes. Python now accepts
`quatf * Vector3`; Transform's world forward/up/right readback uses this same
public expression without an extra normalization.

Native TransformPoint, TransformVector and both direction conversions apply local
rotations in hierarchy order. `TransformVectorMath.h` keeps the rotation's delta
matrix separate from the original vector and preserves the paired float sums.
Using a flattened world matrix or a composed world quaternion changes public
components. Directions retain zero and input magnitude without normalization.

All four conversions pass 1,980 hierarchy inputs / 23,760 exact components in
the actual Windows SDK and native Windows/Linux/Android-device/WebAssembly probes.
All 60 forward matrices also compare exactly. This does not close inverse scale
rounding: the strict audit still records 147 inverse-matrix, 1,282 inverse-point
and 993 inverse-vector component differences. Keep those differences visible;
do not substitute mathematical equivalence for a measured float contract.

## Public quaternion products

`MultiplyPublic` stores only the final x/y/z/w components as float and retains
wider intermediate Hamilton products. It neither normalizes inputs nor changes
the quaternion sign. `MultiplyTransform` remains separate because native pose
composition has its own measured reduction order.

The Tween checkout supplies public cancellation/non-unit tests and a native probe
for 2,048 independently measured products. All 8,192 components match exactly on
Windows, Linux, an Android arm64 device and WebAssembly/Node. The SDK binding
passes the same inputs. The strict live path audit also passes all 872 measured
configurations / 9,592 states at its original tolerances, including closed paths
that feed the previous orientation back into the next sample. These results do
not establish universal equality for every numerical input or execution backend.

## Quaternion interpolation

`quatf.slerp` clamps progress to [0, 1] and chooses the shorter quaternion arc.
An absolute dot product of 0.95 or greater selects normalized linear
interpolation; smaller values select spherical interpolation without changing
the input magnitudes. Zero quaternion storage remains zero on that spherical
path. Normalizing every input or using the math library's default branch
threshold changes observable results.

Direct quaternion Tween inputs are copied when From or endpoint changes are
configured. A custom easing callback receives the current sample after its
quaternion inputs have been captured as native values, so component edits affect
the next invocation. This boundary does not allocate temporary Python quaternion
copies. Calling Python easing and setters requires the GIL; the scalar native
math itself does not add a per-call GIL release/reacquire cycle.

Assigning the start, end or change field copies a supplied quaternion; component
edits through the stored field remain live. An implicit From captures its old
endpoint before invoking the getter. Direct plugin evaluation uses its explicit
elapsed/duration arguments, including the seconds received by custom easing,
instead of substituting the Tween's own duration. The seconds sampler is cached
on first use and invalidated when easing is reconfigured. Normal playback does
not allocate this extra delegate.
