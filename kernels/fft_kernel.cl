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

// mode.x - 0-horizontal, 1-vertical
// mode.y - subsequent count

__kernel void fft_1D( int2 mode, int2 patch_info,
    read_only image2d_t twiddle, read_only image2d_t src, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));

    int2 data_coords = (int2)(mode.y, uv.x * (1-mode.x) + uv.y * mode.x);
    float4 data = read_imagef(twiddle, sampler, data_coords);

    int2 pp_coords0 = (int2)(data.z, uv.y) * (1-mode.x) + (int2)(uv.x, data.z) * mode.x;
    float2 p = read_imagef(src, sampler, pp_coords0).rg;

    int2 pp_coords1 = (int2)(data.w, uv.y) * (1-mode.x) + (int2)(uv.x, data.w) * mode.x;
    float2 q = read_imagef(src, sampler, pp_coords1).rg;

    float2 w = (float2)(data.x, data.y);

    //Butterfly operation
    complex H = add(p,mul(w,q));

    write_imagef(dst, uv, (float4)(H.x, H.y, 0, 1));
}
