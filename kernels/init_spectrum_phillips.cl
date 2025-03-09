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
constant float PI = 3.14159265359f;
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
constant float GRAVITY = 9.81f;

float4 gaussRND(float4 rnd)
{
    float u0 = 2.0*PI*rnd.x;
    float v0 = sqrt(-2.0 * log(rnd.y));
    float u1 = 2.0*PI*rnd.z;
    float v1 = sqrt(-2.0 * log(rnd.w));

    float4 ret = (float4)(v0 * cos(u0), v0 * sin(u0), v1 * cos(u1), v1 * sin(u1));
    return ret;
}

// patch_info.x - ocean patch size
// patch_info.y - ocean texture unified resolution
// params.x - wind x
// params.y - wind.y
// params.z - amplitude
// params.w - capillar supress factor

kernel void init_spectrum( int2 patch_info, float4 params, read_only image2d_t noise, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));

    float2 fuv = convert_float2(uv) - (float2)((float)(patch_info.y-1)/2.f);
    float2 k = (2.f * PI * fuv) / patch_info.x;
    float k_mag = length(k);

    float wind_speed = length((float2)(params.x, params.y));
    float4 params_n = params;
    params_n.xy = (float2)(params.x/wind_speed, params.y/wind_speed);
    float l_phl = (wind_speed * wind_speed) / GRAVITY;

    float magSq = k_mag * k_mag;

    float phillips = exp(-(1.0/(magSq * l_phl * l_phl)));
    float amplitude = (params.z/(magSq*magSq));
    float f0 = sqrt(amplitude * phillips * exp(-magSq*pow(params.w, 2.f))) / sqrt(2.0);

    // directional distribution
    float dp = pow(dot(normalize(k), params_n.xy), 2.f);
    float dm = pow(dot(normalize(-k), params_n.xy), 2.f);

    float h0kp = f0 * dp;
    float h0km = f0 * dm;

    float4 rnd = clamp(read_imagef(noise, sampler, uv), 0.001f, 1.f);
    float4 gauss_random = gaussRND(rnd);
    write_imagef(dst, uv, (float4)(gauss_random.xy*h0kp, gauss_random.zw*h0km));
}
