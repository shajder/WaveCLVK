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

kernel void reduce(
    int2 info,
    read_only image2d_t src,
    write_only image2d_t dst)
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 v0 = read_imagef(src, sampler, uv).xy;
    float2 v1 = read_imagef(src, sampler, (int2)(uv.x + info.x, uv.y)).xy;
    float2 v2 = read_imagef(src, sampler, (int2)(uv.x, uv.y + info.y)).xy;
    float2 v3 = read_imagef(src, sampler, (int2)(uv.x + info.x, uv.y + info.y)).xy;
    float2 mxv = max(max(max(v0, v1), v2), v3);
    write_imagef(dst, uv, (float4)(mxv.x, mxv.y, 0, 0));
}
