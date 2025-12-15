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

    // Sampling neighbors for curvature
    float3 n0 = read_imagef(src, sampler, (float2)(fuv.x + 4.0 * texel, fuv.y)).xyz;
    float3 n1 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y + 4.0 * texel)).xyz;
    float3 n2 = read_imagef(src, sampler, (float2)(fuv.x - 4.0 * texel, fuv.y)).xyz;
    float3 n3 = read_imagef(src, sampler, (float2)(fuv.x, fuv.y - 4.0 * texel)).xyz;

    // Raw curvature computation (before exponentiation)
    float raw_c0 = clamp(fabs(dot(n0, n2) * (-0.5f) + 0.5f), 0.0f, 1.0f);
    float raw_c1 = clamp(fabs(dot(n1, n3) * (-0.5f) + 0.5f), 0.0f, 1.0f);

    // Initial scaling
    float c0_scaled = raw_c0 * 10.0f;
    float c1_scaled = raw_c1 * 10.0f;

    // Sharp edge for foam appearance
    float f0_sharp = pow(c0_scaled, 8.0f);
    float f1_sharp = pow(c1_scaled, 8.0f);
    float mask_sharp = clamp(max(f0_sharp, f1_sharp), 0.0f, 1.0f);

    // Soft edge for foam physics, use the same data, but create a wider, smoother "blob" for CFD.
    // Instead of power 8, use e.g. 2 or smoothstep, which gives wider slopes.
    // Thanks to this, the force is applied over a broader area around the wave crest.
    float f0_smooth = smoothstep(0.5f, 1.0f, c0_scaled); // Threshold 0.5 starts earlier than pow^8^8
    float f1_smooth = smoothstep(0.5f, 1.0f, c1_scaled);
    float mask_smooth = clamp(max(f0_smooth, f1_smooth), 0.0f, 1.0f);

    float dz_c = read_imagef(displ, sampler, uv).y;
    float z_bias = fabs((dz_c - params.x) / (params.y - params.x));

    // Height modifier affects both, but can be less restrictive for physics
    float height_factor = pow(z_bias, params.z);

    // Final coefficients
    float foam_fac_visual = noise_val.x * mask_sharp * height_factor;
    float foam_fac_physics = mask_smooth * height_factor; // Szum usunięty z fizyki dla stabilności!

    float3 field = read_imagef(flds, sampler_point, uv_cfd).xyz;
    float dens_val = field.z;

    // Write to visual texture (using sharp mask)
    write_imagef(dst_nmap, uv, (float4)(ndata.xyz, ndata.w + (dens_val - ndata.w) * 0.5f * params.s5  ));

    // Add new density (using sharp mask foam_fac_visual)
    float injected_density = max(foam_fac_visual, dens_val);
    // Fade old foam
    injected_density = max(0.f, injected_density - 0.01f * params.s5);

    float wlen = length((float3)(params.s3, params.s4, 0.f));
    float3 wind = (float3)(params.s3, params.s4, 0.f) / (float3)(wlen);
    // ext_eff - "exposure to wind" based on the normal
    float ext_eff = clamp(dot (ndata.xyz, -wind), 0.f, 1.f);

    float2 velocity = field.xy;

    // deviation for noise direction
    float max_angle_rad = 1.2f;
    float noise_scaled = noise_val.x*2.f-1.f;
    float angle = noise_scaled * max_angle_rad;
    float c = cos(angle);
    float s = sin(angle);
    float2 rotated_wind_dir;
    rotated_wind_dir.x = wind.x * c - wind.y * s;
    rotated_wind_dir.y = wind.x * s + wind.y * c;

    // do not apply noise because noise in the velocity field is the enemy of divergence
    float force_strength = wlen * foam_fac_physics * ext_eff;
    velocity += rotated_wind_dir * force_strength;

    // Damping – velocity component aligned with wind direction
    // and the wave attack coefficient included
    float wdamp = max(0.f, dot(velocity, wind.xy)) / wlen;
    const float total_dfac=0.001f;
    float damp = total_dfac * wdamp * 0.5f + total_dfac * ext_eff * 0.5f;
    velocity = velocity - (float2)(damp * wlen) * wind.xy;

    // Write to CFD fields: velocity updated smoothly, density (foam) updated sharply
    write_imagef(dst_flds, uv_cfd, (float4)(velocity, injected_density, 0.f));
}
