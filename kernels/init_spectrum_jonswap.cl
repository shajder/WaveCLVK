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
    int res = patch_info.y;
    float2 fuv = convert_float2(uv) - (float2)((float)(res-1)/2.f);

    float2 k = (2.f * PI * fuv) / (patch_info.x/2.f);
    float2 kinv = (float2)(1.f)/k;

    float wind_speed = length((float2)(params.x, params.y));
    float2 wind_norm = (float2)(params.x/wind_speed, params.y/wind_speed);

    // below JONSWAP params comes from struggle for the best visual result
    // also good results:
    // fp = 0.03, gamma = 32.0, alpha = 0.00076, beta=1.25, spreading = 16.0
    // fp = 0.025, gamma = 32.0, alpha = 0.0004, beta=1.25, spreading = 16.0


    // fp describes peak wave-frequency which refers to wave length
    // original factors varies between 0.05-0.5
    float fp = 0.08f;

    // gamma refers to choppines of the wave (higher->more peaky wave)
    // origunal factor was 3.3
    float gamma = 8.f;

    // alpha originally between 0.0075-0.8 but in practice
    // both alpha and beta may be used as scale factors for the remaining params setup
    float alpha = 0.06f;
    float beta = 1.2f;

    // additional usefull factor to scale spread of the wave
    float spreading = 16.f;

    // JONSWAP spectrum
    float sigmax = (k.x <= fp) ? 0.07 : 0.09;
    float fdifx = k.x - fp;
    float rx = exp(-(fdifx*fdifx) / (2 * pow(sigmax * fp, 2)));
    float f0 = alpha * pow(kinv.x, 5) * exp(-beta * pow(fp * kinv.x, 4)) * pow(gamma, rx);

    float sigmay = (k.y <= fp) ? 0.07 : 0.09;
    float fdify = k.y - fp;
    float ry = exp(-(fdify*fdify) / (2 * pow(sigmay * fp, 2)));
    float f1 = alpha * pow(kinv.y, 5) * exp(-beta * pow(fp * kinv.y, 4)) * pow(gamma, ry);

    // directional distribution
    float dp = pow(dot(normalize(k), wind_norm.xy), spreading);
    float dm = pow(dot(normalize(-k), wind_norm.xy), spreading);

    float h0pk = f0 * dp;
    float h0mk = f0 * dm;

    float h1pk = f1 * dp;
    float h1mk = f1 * dm;

    float4 rnd = clamp(read_imagef(noise, sampler, uv), 0.001f, 1.f);
    float4 gauss_random = gaussRND(rnd);

#if 1
    write_imagef(dst, uv, (float4)(gauss_random.xy*(float2)(h0pk,h1pk), gauss_random.zw*(float2)(h0mk,h1mk)));
#else
    write_imagef(dst, uv, (float4)((float2)(h0pk,h1pk), (float2)(h0mk,h1mk)));
#endif
}
