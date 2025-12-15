# WaveCLVK

![Result footage](https://github.com/shajder/WaveCLVK/blob/main/poster.png)

## Overview

**WaveCLVK** is an advanced evolution of the ocean simulation example originally developed for the OpenCL-SDK. While the original sample provided a baseline for FFT-based water rendering, this project extends the scope significantly by introducing realistic wave spectra and a fully interactive, grid-based Fluid Dynamics (CFD) simulation for sea foam.

The goal was to move beyond procedural texturing and simulate foam that possesses genuine inertia, vorticity, and interaction with the wave surface.

## Key Features

### 1. CFD-Based Foam Simulation
The core innovation of this project is the generated sea foam, which is not merely a static texture map but a living fluid simulation running on the GPU.

### 2. JONSWAP Wave Spectrum
As addition to the standard Phillips spectrum often used in real-time demos, this project implements the **JONSWAP (Joint North Sea Wave Project)** spectrum.

#### The Challenge
Standard foam implementations often lack "memory"—foam appears and disappears instantly with the wave height. To solve this, I implemented a 2D Navier-Stokes solver (Stable Fluids approach) mapped to the ocean surface.

#### Technical Implementation Details
- **Decoupled Injection Scheme**: To maintain numerical stability while achieving sharp visuals, the simulation separates the **Visual** and **Physical** injection layers:
    - **Visual Density**: Injected using a sharp power function to create crisp, defined foam on wave crests.
    - **Physical Velocity**: Injected using a smoothed kernel to apply broad, gentle forces to the solver. This prevents "pressure explosions" and ensures the divergence-free condition is met without creating artifacts.
- **Vector Damping**: Replaced standard scalar damping with a vector-based approach that preserves flow direction, allowing for energy conservation in vortices and preventing the destruction of backward-flowing currents.
- **Advection & Diffusion**: Foam density is advected by the velocity field, creating natural swirling patterns and trails that linger behind moving waves.

## Visualization

Below is a visualization of the underlying CFD fields (Density and Velocity) without the ocean rendering overlay. This view demonstrates the vorticity and the "memory" of the foam as it is agitated by the wave peaks.

![CFD Simulation View](https://github.com/shajder/WaveCLVK/blob/main/poster_dist.png)

## Technologies
- **OpenCL**: For general-purpose GPU computing (FFT, Physics).
- **Vulkan**: For high-performance rendering.
- **C++**: Core application logic.
