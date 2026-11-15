[[vk::binding(2, 1)]] RWTexture2D<float4> sourceImage;
[[vk::binding(3, 1)]] RWTexture2D<float4> destinationImage;
[[vk::binding(0, 2)]] Texture2D<float4> sampledTexture;
[[vk::binding(1, 2)]] SamplerState sampledSampler;
[[vk::binding(0, 3)]] StructuredBuffer<float4> inputData;
[[vk::binding(1, 3)]] RWStructuredBuffer<float4> outputData;
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    float4 value = sourceImage[int2(0)] + sampledTexture.SampleLevel(sampledSampler, float2(0.5), 0.0);
    destinationImage[int2(0)] = value;
    outputData[0] = inputData[0] + value;
}
