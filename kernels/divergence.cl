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

// 02 ----- 12 ----- 22
// |        |        |
// |        |        |
// 01 ----- 11 ----- 21
// |        |        |
// |        |        |
// 00 ----- 10 ----- 20
// info.x - simulation width
// info.y - simulation height
// info.z - dt
// info.w - unused
constant sampler_t sampler_repeat = CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;
float2 px2tx(float2 fuv, float4 info)
{
    return fuv / (float2)(info.x, info.y);
}
kernel void divergence( float4 info,
                        read_only image2d_t src,
                        write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 pxc = convert_float2(uv) + (float2)(0.5f);

    float2 field01 = read_imagef(src, sampler_repeat, px2tx(pxc + (float2)(-1.f, 0.f), info)).xy;
    float2 field21 = read_imagef(src, sampler_repeat, px2tx(pxc + (float2)( 1.f, 0.f), info)).xy;
    float2 field10 = read_imagef(src, sampler_repeat, px2tx(pxc + (float2)( 0.f,-1.f), info)).xy;
    float2 field12 = read_imagef(src, sampler_repeat, px2tx(pxc + (float2)( 0.f, 1.f), info)).xy;

    // to be verified
    float r = 0.25f * (field21.x - field01.x + field12.y - field10.y);

    write_imagef(dst, uv, (float4)(r, 0.f, 0.f, 0.f));
}
