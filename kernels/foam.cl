/*
MIT License

Copyright (c) 2025 Marcin Hajder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
constant sampler_t sampler = CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;
constant sampler_t sampler_point = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
// patch_info.x - ocean patch size
// patch_info.y - ocean texture unified resolution
// params.x - z min
// params.y - z max
// params.z - foam delimiter
kernel void update_foam( int2 patch_info, float3 params, read_only image2d_t noise,
                         read_only image2d_t displ, read_only image2d_t src, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 fuv = convert_float2(uv) / patch_info.y;

    float texel = 1.f / patch_info.y;
    float4 ndata = read_imagef(src, sampler_point, uv);

    float3 n0 = read_imagef(src, sampler, (float2)(fuv.x + 4.0 * texel, fuv.y)).xyz;
    float3 n1 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y + 4.0 * texel)).xyz;
    float3 n2 = read_imagef(src, sampler, (float2)(fuv.x - 4.0 * texel, fuv.y)).xyz;
    float3 n3 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y - 4.0 * texel)).xyz;

    float f0 = clamp(fabs(dot(n0, n2) * (-0.5f) + 0.5f), 0.0f, 1.0f);
    float f1 = clamp(fabs(dot(n1, n3) * (-0.5f) + 0.5f), 0.0f, 1.0f);

    f0 = pow(f0 * 8.0f, 2.0f);
    f1 = pow(f1 * 8.0f, 2.0f);

    float dz_c = read_imagef(displ, sampler, uv).y;
    float z_bias = fabs((dz_c - params.x) / (params.y - params.x));

    float4 n = read_imagef(noise, sampler_point, uv);
    float foam_fac = n.x * clamp(max(f0, f1), 0.0f, 1.0f) * pow(z_bias, params.z);

    foam_fac = max(foam_fac, ndata.w);
    foam_fac -= 0.001; // should be replaced with time-dependent factor

    write_imagef(dst, uv, (float4)(ndata.xyz, foam_fac));
}
