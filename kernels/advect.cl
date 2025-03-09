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
constant sampler_t sampler_repeat = CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_TRUE;
float2 px2tx(float2 fuv, float4 info)
{
    return fuv / (float2)(info.x, info.y);
}
// info.x - simulation width
// info.y - simulation height
// info.z - dt
// info.w - dumping factor
kernel void advect( float4 info, read_only image2d_t vels,
                          read_only image2d_t field, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 pxc = convert_float2(uv) + (float2)(0.5f);

    float dt = info.z*info.w;
    float2 vel = read_imagef(vels, sampler_repeat, px2tx(pxc, info)).xy;

    float2 pos = pxc - (float2)(dt) * vel;
    float4 val = read_imagef(field, sampler_repeat, px2tx(pos, info));
    write_imagef(dst, uv, val);
}
