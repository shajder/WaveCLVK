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
kernel void jacobi( float4 info,
                    read_only image2d_t div,
                    read_only image2d_t press_src,
                    write_only image2d_t press_dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float2 pxc = convert_float2(uv) + (float2)(0.5f);

    float dc = read_imagef(div, sampler_repeat, px2tx(pxc, info)).x;

    float pl = read_imagef(press_src, sampler_repeat, px2tx(pxc - (float2)( 1.f, 0.f), info)).x;
    float pb = read_imagef(press_src, sampler_repeat, px2tx(pxc - (float2)( 0.f, 1.f), info)).x;

    float pr = read_imagef(press_src, sampler_repeat, px2tx(pxc + (float2)( 1.f, 0.f), info)).x;
    float pt = read_imagef(press_src, sampler_repeat, px2tx(pxc + (float2)( 0.f, 1.f), info)).x;

    const float viscosity = 0.1f; //0.001f;
    const float alpha = 1.f/(4.f+viscosity*info.z);
    float pcd = alpha * (pl + pr + pb + pt - (1.f/(alpha)) * dc);
    write_imagef(press_dst, uv, (float4)(pcd,0.f,0.f,0.f));
}
