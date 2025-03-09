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

typedef float2 complex;

kernel void generate( int resolution, global int * bit_reversed, write_only image2d_t dst )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    float k = fmod(uv.y * ((float)(resolution) / pow(2.f, (float)(uv.x+1))), resolution);
    complex twiddle = (complex)( cos(2.0*PI*k/(float)(resolution)), sin(2.0*PI*k/(float)(resolution)));

    int butterflyspan = (int)(pow(2.f, (float)(uv.x)));
    int butterflywing;

    if (fmod(uv.y, pow(2.f, (float)(uv.x + 1))) < pow(2.f, (float)(uv.x)))
        butterflywing = 1;
    else
        butterflywing = 0;

    // first stage, bit reversed indices
    if (uv.x == 0) {
        // top butterfly wing
        if (butterflywing == 1)
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, bit_reversed[(int)(uv.y)], bit_reversed[(int)(uv.y + 1)]));
        // bot butterfly wing
        else
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, bit_reversed[(int)(uv.y - 1)], bit_reversed[(int)(uv.y)]));
    }
    // second to log2(resolution) stage
    else {
        // top butterfly wing
        if (butterflywing == 1)
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, uv.y, uv.y + butterflyspan));
        // bot butterfly wing
        else
            write_imagef(dst, uv, (float4)(twiddle.x, twiddle.y, uv.y - butterflyspan, uv.y));
    }
}
