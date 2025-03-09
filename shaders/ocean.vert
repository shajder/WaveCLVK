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

layout(location = 0) out vec2 frag_tex_coord;
layout(location = 1) out vec4 ec_pos;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(set = 0, binding = 0) uniform sampler2D u_displacement_map;
layout(std140, set = 0, binding = 2) uniform ViewData {
    uniform mat4    view_mat;
    uniform mat4    proj_mat;
    uniform vec3    sun_dir;
    uniform float   z_range_min;
    uniform float   z_range_max;
    uniform float   choppiness;
    uniform float   alt_scale;
} view;

void main()
{
    vec3 displ = texture(u_displacement_map, in_tex_coords).rbg; // swizzle
    float z_bias = abs((displ.z - view.z_range_min) / (view.z_range_max - view.z_range_min));

    displ.xy *= view.choppiness * (1.0+z_bias);
    displ.z *= view.alt_scale * mix(1.0, 1.3, pow(z_bias, 10.0));

    vec3 ocean_vert = in_position + displ;
    ec_pos = view.view_mat * vec4(ocean_vert, 1.0);
    gl_Position = view.proj_mat * ec_pos;
    frag_tex_coord = in_tex_coords;
}
