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
#version 450

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 1) in vec4 ec_pos;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform sampler2D u_normal_map;
layout(binding = 2) uniform ViewData {
    uniform mat4    view_mat;
    uniform mat4    proj_mat;
    uniform vec3    sun_dir;
    uniform float   z_range_min;
    uniform float   z_range_max;
    uniform float   choppiness;
    uniform float   alt_scale;
} view;

const vec3 env_specular = vec3(0.8);
const float specular_power = 32.0;
const float specular_scale = 0.75;

const float fresnel_approx_pow_factor = 2.0;
const float dyna_range = 0.8f;

const vec3 ocean_bright = vec3(0.5, 1.6, 2.15);
const vec3 ocean_dark = vec3(0.03, 0.06, 0.135);
const float exposure = 0.4;

vec3 hdr(vec3 color, float exposure)
{
    return 1.0 - exp(-color * exposure);
}

mat3 get_linear_part( mat4 m )
{
        mat3 result;

        result[0][0] = m[0][0];
        result[0][1] = m[0][1];
        result[0][2] = m[0][2];

        result[1][0] = m[1][0];
        result[1][1] = m[1][1];
        result[1][2] = m[1][2];

        result[2][0] = m[2][0];
        result[2][1] = m[2][1];
        result[2][2] = m[2][2];

        return result;
}

void main()
{
    // normal map computed in opencl kernel
    vec4 ndata = texture(u_normal_map, frag_tex_coord);

    // foam calculated in OpenCL code, could be improved with hi-res noise here
    float foam = ndata.w;

    // preparation of view space lighting computation
    mat3 norm_mat = get_linear_part(view.view_mat);
    vec3 normal = norm_mat * ndata.xyz;
    vec3 light_dir = normalize(norm_mat * view.sun_dir);
    vec3 view_dir = normalize(ec_pos.xyz);

    // diffuse + specular
    vec3 specular = vec3(0.0);
    float n_dot_vp = max(0.0, dot(normal, light_dir));
    float n_dot_e = dot(normal, -view_dir);
    float diffuse = clamp(dot(normal, light_dir), 0.0, 1.0);

    if (n_dot_vp > 0.0)
    {
      vec3 N = normal;
      vec3 E = -view_dir;
      vec3 R = normalize(reflect(-light_dir, N));

      // modulate specular scale value based on fragment direction
      float dirScale = mix ( pow ( abs(n_dot_e), 8.0 ),
                             1.0 - pow ( abs(1.0 - n_dot_e), 4.0 ), n_dot_e);
      specular = env_specular * vec3 ( pow(max(dot(R, E), 0.0), specular_power) *
                                           specular_scale * dirScale );
    }

    // refraction factors and final color assembly
    float fresnel = clamp(pow( 1.0 + n_dot_e, -fresnel_approx_pow_factor ) * dyna_range, 0.0, 1.0);
    vec3 bright = fresnel * ocean_bright;
    vec3 water = (1.0 - fresnel) * ocean_dark * ocean_bright * diffuse;

    vec3 color = bright + water + specular;
    out_color = vec4(hdr(color, exposure) + vec3(foam), 1.0);
}
