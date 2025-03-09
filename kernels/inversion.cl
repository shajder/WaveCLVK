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

kernel void inversion( int2 patch_info, read_only image2d_t src0,
    read_only image2d_t src1, read_only image2d_t src2, write_only image2d_t dst, write_only image2d_t ranges )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    int res2 = patch_info.y * patch_info.y;

    float x = read_imagef(src0, sampler, uv).x;
    float y = read_imagef(src1, sampler, uv).x;
    float z = read_imagef(src2, sampler, uv).x;

    write_imagef(dst, uv, (float4)(x/res2, y/res2, z/res2, 1));
    write_imagef(ranges, uv, (float4)(y/res2, y/res2, 0, 0));
}
