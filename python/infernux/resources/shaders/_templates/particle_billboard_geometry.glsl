vec3 inxSafeBillboardAxis(vec3 value, vec3 fallbackValue) {
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-10 ? value * inversesqrt(lengthSquared) : fallbackValue;
}

void inxParticleBillboardBasis(
    vec3 position, vec3 customAxis, vec3 viewRight, vec3 viewUp, vec4 alignmentReference,
    out vec3 rightAxis, out vec3 upAxis) {
    vec3 cameraRight = inxSafeBillboardAxis(viewRight, vec3(1.0, 0.0, 0.0));
    vec3 cameraUp = inxSafeBillboardAxis(viewUp, vec3(0.0, 1.0, 0.0));
    vec3 cameraNormal = inxSafeBillboardAxis(cross(cameraRight, cameraUp), vec3(0.0, 0.0, 1.0));
    int alignment = int(round(alignmentReference.w));
    if (alignment == 0) {
        rightAxis = cameraRight;
        upAxis = cameraUp;
        return;
    }
    if (alignment == 1) {
        vec3 toCamera = inxSafeBillboardAxis(alignmentReference.xyz - position, cameraNormal);
        rightAxis = inxSafeBillboardAxis(cross(cameraUp, toCamera), cameraRight);
        upAxis = inxSafeBillboardAxis(cross(toCamera, rightAxis), cameraUp);
        return;
    }
    vec3 requestedUp = alignment == 2 ? alignmentReference.xyz : customAxis;
    upAxis = inxSafeBillboardAxis(requestedUp, cameraUp);
    vec3 projectedRight = cross(upAxis, cameraNormal);
    if (dot(projectedRight, projectedRight) <= 1.0e-10)
        projectedRight = cameraRight - upAxis * dot(cameraRight, upAxis);
    rightAxis = inxSafeBillboardAxis(projectedRight, cameraRight);
    upAxis = inxSafeBillboardAxis(cross(cameraNormal, rightAxis), upAxis);
}

void inxParticleBillboardVertex(
    vec2 corner, vec4 positionSize, float roll, vec2 scale, vec3 customAxis,
    vec3 viewRight, vec3 viewUp, vec4 alignmentReference,
    out vec3 position, out vec3 normal, out vec4 tangent) {
    vec3 rightAxis, upAxis;
    inxParticleBillboardBasis(positionSize.xyz, customAxis, viewRight, viewUp, alignmentReference,
                             rightAxis, upAxis);
    float cosine = cos(roll);
    float sine = sin(roll);
    vec2 rotatedCorner = mat2(cosine, -sine, sine, cosine) * corner;
    position = positionSize.xyz +
        (rightAxis * rotatedCorner.x * scale.x + upAxis * rotatedCorner.y * scale.y) * positionSize.w;
    normal = inxSafeBillboardAxis(cross(rightAxis, upAxis), vec3(0.0, 0.0, 1.0));

    // Geometry applies scale after roll. Its corner-x derivative supplies T;
    // size changes its direction only by sign, avoiding tiny-size underflow.
    float sizeSign = positionSize.w < 0.0 ? -1.0 : 1.0;
    vec3 rolledRight = (rightAxis * cosine - upAxis * sine) * sizeSign;
    vec3 tangentAxis = (rightAxis * (cosine * scale.x) - upAxis * (sine * scale.y)) * sizeSign;
    float tangentMagnitude = max(max(abs(tangentAxis.x), abs(tangentAxis.y)), abs(tangentAxis.z));
    tangent.xyz = tangentMagnitude > 0.0
        ? normalize(tangentAxis / tangentMagnitude)
        : inxSafeBillboardAxis(rolledRight, rightAxis);
    // SixWay uses physical up (corner +y), opposite to the sprite's UV-v.
    // A zero-size/scale card has no surface; keep its fallback frame finite.
    tangent.w = (scale.x < 0.0) != (scale.y < 0.0) ? -1.0 : 1.0;
}
