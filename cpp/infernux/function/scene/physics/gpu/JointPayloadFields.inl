// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: MIT
// One compiled schema shared by C++ payloads and Slang typed accessors.
#define JPH_FIXED_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2)
#define JPH_HINGE_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(ownerAxis) \
    F(limits) \
    F(motor) \
    F(passive)
#define JPH_SLIDER_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(ownerAxis) \
    F(limits)
#define JPH_DISTANCE_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(connectedAnchor) \
    F(connectedLocalAnchor) \
    F(limits) \
    F(linearSpring)
#define JPH_SWING_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(angularFrame) \
    F(angularLimits) \
    U(angularModes) \
    F(angularSpring) \
    F(angularTorque) \
    F(angularTarget) \
    F(angularVelocity)
#define JPH_FIXED_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(frame1) \
    F(frame2) \
    F(pointImpulse) \
    F(angularImpulse)
#define JPH_HINGE_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(frame1) \
    F(frame2) \
    F(pointImpulse) \
    F(angularImpulse) \
    F(axisImpulse) \
    F(rotor) \
    F(setup) \
    F(poseDriveRelative)

#define JPH_SLIDER_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(frame1) \
    F(frame2) \
    F(pointImpulse) \
    F(angularImpulse) \
    F(axisImpulse) \
    F(setup)
#define JPH_DISTANCE_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(axisImpulse) \
    F(setup) \
    F(distanceReference)
#define JPH_SWING_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(frame1) \
    F(frame2) \
    F(pointImpulse) \
    F(angularLimitX) \
    F(angularLimitY) \
    F(angularLimitZ) \
    F(angularLimitImpulse) \
    F(angularMotorImpulse) \
    F(angularMotorSetupX) \
    F(angularMotorSetupY) \
    F(angularMotorSetupZ) \
    F(poseDriveRelative)

#define JPH_ALL_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(ownerAxis) \
    F(limits) \
    F(motor) \
    F(passive) \
    F(connectedAnchor) \
    F(connectedLocalAnchor) \
    F(linearSpring) \
    F(angularFrame) \
    F(angularLimits) \
    U(angularModes) \
    F(angularSpring) \
    F(angularTorque) \
    F(angularTarget) \
    F(angularVelocity) \
    U(sixModes) \
    F(sixLinearMin) \
    F(sixLinearMax) \
    F(sixAngularMin) \
    F(sixAngularMax) \
    F(sixLimitStiffness) \
    F(sixLimitDamping) \
    F(sixLinearFriction) \
    F(sixAngularFriction) \
    F(sixLinearMotorK) \
    F(sixLinearMotorC) \
    F(sixLinearMotorForce) \
    F(sixAngularMotorK) \
    F(sixAngularMotorC) \
    F(sixAngularMotorTorque) \
    F(sixTargetPosition) \
    F(sixTargetVelocity)


#define JPH_SIX_AUTHOR_FIELDS(F,U) \
    F(ownerAnchor) \
    F(ownerLocalAnchor) \
    F(rootCOM1) \
    F(rootCOM2) \
    F(scale1) \
    F(scale2) \
    F(angularFrame) \
    U(sixModes) \
    F(sixLinearMin) \
    F(sixLinearMax) \
    F(sixAngularMin) \
    F(sixAngularMax) \
    F(sixLimitStiffness) \
    F(sixLimitDamping) \
    F(sixLinearFriction) \
    F(sixAngularFriction) \
    F(sixLinearMotorK) \
    F(sixLinearMotorC) \
    F(sixLinearMotorForce) \
    F(sixAngularMotorK) \
    F(sixAngularMotorC) \
    F(sixAngularMotorTorque) \
    F(sixTargetPosition) \
    F(sixTargetVelocity) \
    F(angularTarget) \
    F(angularVelocity)
#define JPH_SIX_STATE_FIELDS(F,U) \
    U(flags) \
    F(anchor1) \
    F(anchor2) \
    F(collapsed1) \
    F(collapsed2) \
    F(frame1) \
    F(frame2) \
    F(pointImpulse) \
    F(angularImpulse) \
    F(angularLimitX) \
    F(angularLimitY) \
    F(angularLimitZ) \
    F(angularLimitImpulse) \
    F(angularMotorImpulse) \
    F(angularMotorSetupX) \
    F(angularMotorSetupY) \
    F(angularMotorSetupZ) \
    F(sixTranslationAxisX) \
    F(sixTranslationAxisY) \
    F(sixTranslationAxisZ) \
    F(sixLimitSetupX) \
    F(sixLimitSetupY) \
    F(sixLimitSetupZ) \
    F(sixMotorSetupX) \
    F(sixMotorSetupY) \
    F(sixMotorSetupZ) \
    F(sixLimitImpulse) \
    F(sixMotorImpulse) \
    F(poseDriveRelative)
