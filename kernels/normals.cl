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
constant sampler_t sampler_point = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
constant float normal_scale_fac = 3.f;
// patch_info.x - ocean patch size
// patch_info.y - ocean texture unified resolution
// scale_fac.x - choppines
// scale_fac.y - altitude scale
kernel void normals( int2 patch_info, read_only image2d_t src,
                     read_only image2d_t rdst, write_only image2d_t wdst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 fuv = convert_float2(uv) / patch_info.y;

    float texel = 1.f / patch_info.y;

    float dz_c = read_imagef(src, sampler, fuv).y;
    float dz_cr = read_imagef(src, sampler, (float2)(fuv.x + texel, fuv.y)).y;
    float dz_ct = read_imagef(src, sampler, (float2)(fuv.x, fuv.y + texel)).y;
    float dz_cl = read_imagef(src, sampler, (float2)(fuv.x - texel, fuv.y)).y;
    float dz_cb = read_imagef(src, sampler, (float2)(fuv.x, fuv.y - texel)).y;
    float dz_tr = read_imagef(src, sampler, (float2)(fuv.x + texel, fuv.y + texel)).y;
    float dz_br = read_imagef(src, sampler, (float2)(fuv.x + texel, fuv.y - texel)).y;
    float dz_tl = read_imagef(src, sampler, (float2)(fuv.x - texel, fuv.y + texel)).y;
    float dz_bl = read_imagef(src, sampler, (float2)(fuv.x - texel, fuv.y - texel)).y;

    float3 normal = (float3)(0.f, 0.f, 1.f / normal_scale_fac);
    normal.y = dz_c + 2.f * dz_cb + dz_br - dz_tl - 2.f * dz_ct - dz_tr;
    normal.x = dz_c + 2.f * dz_cl + dz_tl - dz_br - 2.f * dz_cr - dz_tr;

    float4 prev = read_imagef(rdst, sampler_point, uv);
    prev = (float4)(normalize(normal), prev.w);

    write_imagef(wdst, uv, prev);
}
