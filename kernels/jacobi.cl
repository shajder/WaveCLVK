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
kernel void jacobi( float4 info,
                    read_only image2d_t div,
                    read_only image2d_t press_src,
                    write_only image2d_t press_dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));

    float dc = read_imagef(div, sampler_repeat, uv).x;

    float pl = read_imagef(press_src, sampler_repeat, (uv - (int2)(1, 0))).x;
    float pb = read_imagef(press_src, sampler_repeat, (uv - (int2)(0, 1))).x;

    float pr = read_imagef(press_src, sampler_repeat, (uv + (int2)(1, 0))).x;
    float pt = read_imagef(press_src, sampler_repeat, (uv + (int2)(0, 1))).x;

    const float alpha = 0.25f;
    float pcd = alpha * (pl + pr + pb + pt - dc);
    write_imagef(press_dst, uv, (float4)(pcd, 0.f, 0.f, 0.f));
}
