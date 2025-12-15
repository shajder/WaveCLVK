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
constant sampler_t sampler_repeat = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;
kernel void pressure( float4 info,
                    read_only image2d_t press,
                    read_only image2d_t vels_src,
                    write_only image2d_t vels_dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));

    float3 field = read_imagef(vels_src, sampler_repeat, uv).xyz;
    float2 vc = field.xy;

    float pl = read_imagef(press, sampler_repeat, (uv - (int2)( 1, 0))).x;
    float pb = read_imagef(press, sampler_repeat, (uv - (int2)( 0, 1))).x;

    float pr = read_imagef(press, sampler_repeat, (uv + (int2)( 1, 0))).x;
    float pt = read_imagef(press, sampler_repeat, (uv + (int2)( 0, 1))).x;

    float dt = info.z;
    float2 grad = (float2)(dt * 0.5) * (float2)(pr - pl, pt - pb);

    write_imagef(vels_dst, uv, (float4)((vc.x-grad.x), (vc.y-grad.y), field.z, 0.f));
}
