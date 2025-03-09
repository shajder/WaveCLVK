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
constant sampler_t sampler_point = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;

float2 px2tx(float2 fuv, int2 info)
{
    return fuv / (float2)(info.x, info.y);
}

// patch_info.x - ocean patch size
// patch_info.y - ocean texture unified resolution
// params.s0 - z min
// params.s1 - z max
// params.s2 - foam delimiter
// params.s3 - wind x
// params.s4 - wind y
// params.s5 - dt
kernel void update_foam( int2 patch_info, float8 params,
                         read_only image2d_t noise,
                         read_only image2d_t displ,
                         read_only image2d_t flds,
                         read_only image2d_t src,
                         write_only image2d_t dst_flds,
                         write_only image2d_t dst_nmap )
{
    int2 uv = (int2)((int)get_global_id(0), (int)get_global_id(1));
    int2 uv_cfd = uv * (int2)(params.s7);
    float2 fuv = convert_float2(uv) / patch_info.y;

    float texel = 1.f / patch_info.y;
    float4 ndata = read_imagef(src, sampler_point, uv);
    float4 noise_val = read_imagef(noise, sampler_point, uv);

    float3 n0 = read_imagef(src, sampler, (float2)(fuv.x + 4.0 * texel, fuv.y)).xyz;
    float3 n1 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y + 4.0 * texel)).xyz;
    float3 n2 = read_imagef(src, sampler, (float2)(fuv.x - 4.0 * texel, fuv.y)).xyz;
    float3 n3 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y - 4.0 * texel)).xyz;

    float f0 = clamp(fabs(dot(n0, n2) * (-0.5f) + 0.5f), 0.0f, 1.0f);
    float f1 = clamp(fabs(dot(n1, n3) * (-0.5f) + 0.5f), 0.0f, 1.0f);

    f0 = pow(f0 * 10.0f, 8.0f);
    f1 = pow(f1 * 10.0f, 8.0f);

    float dz_c = read_imagef(displ, sampler, uv).y;
    float z_bias = fabs((dz_c - params.x) / (params.y - params.x));
    float foam_fac = noise_val.x * clamp(max(f0, f1), 0.0f, 1.0f) * pow(z_bias, params.z);

    float3 field = read_imagef(flds, sampler_point, uv_cfd).xyz;
    float dens_val = field.z;

    write_imagef(dst_nmap, uv, (float4)(ndata.xyz, ndata.w + (dens_val - ndata.w) * 0.5f * params.s5  ));

    // add new density
    float3 wind = (float3)(-params.s3, -params.s4, 0.f);
    float ext_eff = clamp(dot (ndata.xyz, wind), 0.f, 1.f);

    foam_fac = max(foam_fac, dens_val);
    foam_fac = max(0.f, foam_fac - 0.01f * params.s5);

    // apply external force only if new density applied
    float2 velocity = field.xy;
    if (foam_fac > dens_val)
    {
        velocity += params.s34 * (float2)(ext_eff * noise_val.x * params.s5 * params.s6);
    }
    float damp = params.s6 * 0.01f;
    velocity = max((float2)(0.f), velocity - (float2)(damp * params.s5));

    write_imagef(dst_flds, uv_cfd, (float4)(velocity, foam_fac, 0.f));

#if 0
    barrier(CLK_IMAGE_MEM_FENCE);

    float4 prev_vel = read_imagef(vels, sampler_point, uv_cfd);
    float2 velocity = prev_vel.xy + params.s34 * (float2)(ext_eff * noise_val.x * params.s5 * params.s6);

    float2 pxc = convert_float2(uv_cfd) + (float2)(0.5f);
    float2 pos = pxc - (float2)(params.s5) * velocity.xy;
    float4 val = read_imagef(dens, sampler, px2tx(pos, patch_info * (int2)(params.s7)));

    if (val.x < 0.05f)
    {
        velocity = prev_vel.xy;
    }

    float damp = params.s6 * 0.05f;
    velocity = max((float2)(0.f), velocity.xy - (float2)(damp * params.s5));

    write_imagef(dst_vels, uv_cfd, (float4)(velocity, 0.f, 0.f));

#endif

}
