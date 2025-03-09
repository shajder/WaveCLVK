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
constant float PI = 3.14159265359;
constant float G = 9.81;
constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

typedef float2 complex;

complex mul(complex c0, complex c1)
{
    return (complex)(c0.x * c1.x - c0.y * c1.y, c0.x * c1.y + c0.y * c1.x);
}

complex add(complex c0, complex c1)
{
    return (complex)(c0.x + c1.x, c0.y + c1.y);
}

complex conj(complex c)
{
    return (complex)(c.x, -c.y);
}

kernel void spectrum( float dt, int2 patch_info,
    read_only image2d_t src, write_only image2d_t dst_x,
    write_only image2d_t dst_y, write_only image2d_t dst_z )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 wave_vec = convert_float2(uv) - (float2)((float)(patch_info.y-1)/2.f);
    float2 k = (2.f * PI * wave_vec) / patch_info.x;
    float k_mag = length(k);

    float w = sqrt(G * k_mag);

    float4 h0k = read_imagef(src, sampler, uv);
    complex fourier_amp = (complex)(h0k.x, h0k.y);
    complex fourier_amp_conj = conj((complex)(h0k.z, h0k.w));

    float cos_wt = cos(w*dt);
    float sin_wt = sin(w*dt);

    // euler formula
    complex exp_iwt = (complex)(cos_wt, sin_wt);
    complex exp_iwt_inv = (complex)(cos_wt, -sin_wt);

    // dy
    complex h_k_t_dy = add(mul(fourier_amp, exp_iwt), (mul(fourier_amp_conj, exp_iwt_inv)));

    // dx
    complex dx = (complex)(0.0,-k.x/k_mag);
    complex h_k_t_dx = mul(dx, h_k_t_dy);

    // dz
    complex dz = (complex)(0.0,-k.y/k_mag);
    complex h_k_t_dz = mul(dz, h_k_t_dy);

    // amplitude
    write_imagef(dst_y, uv, (float4)(h_k_t_dy.x, h_k_t_dy.y, 0, 1));

    // choppiness
    write_imagef(dst_x, uv, (float4)(h_k_t_dx.x, h_k_t_dx.y, 0, 1));
    write_imagef(dst_z, uv, (float4)(h_k_t_dz.x, h_k_t_dz.y, 0, 1));
}
