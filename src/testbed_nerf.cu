/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_nerf.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>
#include <neural-graphics-primitives/nerf.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#if NVPV_USE_OPENVXL
#include <openvxl/compute/cuda.h>
#include <openvxl/platform/cuda/cuda.h>
#include <openvxl/math/AffineTransform.h> // for AffineTransform
#include <openvxl/math/Color.h>           // for colorFromPackedRGBA

namespace vxl {
using namespace openvxl;
using namespace openvxl::math;
} // namespace vxl
#endif

#ifdef copysign
#undef copysign
#endif

namespace ngp {

//----------------------------------------------------------------------------------------------
// convert a depth value and slice-count to a slice-index.
OPENVXL_CUDA_INLINE int getSliceIndexFromDepth(float depth, int sliceCount, float depthNear, float depthFar)
{
    float unitZ    = vxl::clamp((depth - depthNear) / (depthFar - depthNear), 0.f, 1.f);
    float warpedZ  = vxl::pow(unitZ, 1.f / 1);
    int sliceIndex = (sliceCount) *warpedZ;
    return vxl::clamp(sliceIndex, 0, sliceCount - 1);
}

//----------------------------------------------------------------------------------------------
// convert slice-index and slice-count to a depth value.
OPENVXL_CUDA_INLINE float getDepthFromSliceIndex(int sliceIndex, int sliceCount, float depthNear, float depthFar)
{
    float warpedZ = (float) sliceIndex / (sliceCount - 1);
    float unitZ   = vxl::pow(warpedZ, 1.f);
    return unitZ * (depthFar - depthNear) + depthNear;
}

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

Testbed::NetworkDims Testbed::network_dims_nerf() const
{
    NetworkDims dims;
    dims.n_input  = sizeof(NerfCoordinate) / sizeof(float);
    dims.n_output = 4;
    dims.n_pos    = sizeof(NerfPosition) / sizeof(float);
    return dims;
}

__global__ void extract_srgb_with_activation(const uint32_t n_elements,
                                             const uint32_t rgb_stride,
                                             const float* __restrict__ rgbd,
                                             float* __restrict__ rgb,
                                             ENerfActivation rgb_activation,
                                             bool from_linear)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    const uint32_t elem_idx = i / 3;
    const uint32_t dim_idx  = i - elem_idx * 3;

    float c = network_to_rgb(rgbd[elem_idx * 4 + dim_idx], rgb_activation);
    if (from_linear)
    {
        c = linear_to_srgb(c);
    }

    rgb[elem_idx * rgb_stride + dim_idx] = c;
}

__global__ void mark_untrained_density_grid(const uint32_t n_elements,
                                            float* __restrict__ grid_out,
                                            const uint32_t n_training_images,
                                            const TrainingImageMetadata* __restrict__ metadata,
                                            const TrainingXForm* training_xforms,
                                            bool clear_visible_voxels)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint32_t level   = i / NERF_GRID_N_CELLS();
    uint32_t pos_idx = i % NERF_GRID_N_CELLS();

    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
    vec3 pos = (vec3{(float) x, (float) y, (float) z} / (float) NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

    vec3 corners[8] = {
        pos + vec3{0.0f, 0.0f, 0.0f},
        pos + vec3{voxel_size, 0.0f, 0.0f},
        pos + vec3{0.0f, voxel_size, 0.0f},
        pos + vec3{voxel_size, voxel_size, 0.0f},
        pos + vec3{0.0f, 0.0f, voxel_size},
        pos + vec3{voxel_size, 0.0f, voxel_size},
        pos + vec3{0.0f, voxel_size, voxel_size},
        pos + vec3{voxel_size, voxel_size, voxel_size},
    };

    // Number of training views that need to see a voxel cell
    // at minimum for that cell to be marked trainable.
    // Floaters can be reduced by increasing this value to 2,
    // but at the cost of certain reconstruction artifacts.
    const uint32_t min_count = 1;
    uint32_t count           = 0;

    for (uint32_t j = 0; j < n_training_images && count < min_count; ++j)
    {
        const auto& xform = training_xforms[j].start;
        const auto& m     = metadata[j];

        if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong ||
            m.lens.mode == ELensMode::Equirectangular)
        {
            // FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
            // by definition see everything.
            ++count;
            continue;
        }

        for (uint32_t k = 0; k < 8; ++k)
        {
            // Only consider voxel corners in front of the camera
            vec3 dir = normalize(corners[k] - xform[3]);
            if (dot(dir, xform[2]) < 1e-4f)
            {
                continue;
            }

            // Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
            vec2 uv =
                pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

            // `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
            // So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
            Ray ray = uv_to_ray(0.0f,
                                uv,
                                m.resolution,
                                m.focal_length,
                                xform,
                                m.principal_point,
                                vec3(0.0f),
                                0.0f,
                                1.0f,
                                0.0f,
                                {},
                                {},
                                m.lens);
            if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f)
            {
                ++count;
                break;
            }
        }
    }

    if (clear_visible_voxels || (grid_out[i] < 0) != (count < min_count))
    {
        grid_out[i] = (count >= min_count) ? 0.f : -1.f;
    }
}

__global__ void generate_grid_samples_nerf_uniform(ivec3 res_3d,
                                                   const uint32_t step,
                                                   BoundingBox render_aabb,
                                                   mat3 render_aabb_to_local,
                                                   BoundingBox train_aabb,
                                                   NerfPosition* __restrict__ out)
{
    // check grid_in for negative values -> must be negative on output
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z)
    {
        return;
    }

    uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
    vec3 pos   = vec3{(float) x, (float) y, (float) z} / vec3(res_3d - 1);
    pos        = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);
    out[i]     = {warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE())};
}

// generate samples for uniform grid including constant ray direction
__global__ void generate_grid_samples_nerf_uniform_dir(ivec3 res_3d,
                                                       const uint32_t step,
                                                       BoundingBox render_aabb,
                                                       mat3 render_aabb_to_local,
                                                       BoundingBox train_aabb,
                                                       vec3 ray_dir,
                                                       PitchedPtr<NerfCoordinate> network_input,
                                                       const float* extra_dims,
                                                       bool voxel_centers)
{
    // check grid_in for negative values -> must be negative on output
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z)
    {
        return;
    }

    uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
    vec3 pos;
    if (voxel_centers)
    {
        pos = vec3{(float) x + 0.5f, (float) y + 0.5f, (float) z + 0.5f} / vec3(res_3d);
    }
    else
    {
        pos = vec3{(float) x, (float) y, (float) z} / vec3(res_3d - 1);
    }

    pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);

    network_input(i)->set_with_optional_extra_dims(warp_position(pos, train_aabb),
                                                   warp_direction(ray_dir),
                                                   warp_dt(MIN_CONE_STEPSIZE()),
                                                   extra_dims,
                                                   network_input.stride_in_bytes);
}

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements,
                                                      default_rng_t rng,
                                                      const uint32_t step,
                                                      BoundingBox aabb,
                                                      const float* __restrict__ grid_in,
                                                      NerfPosition* __restrict__ out,
                                                      uint32_t* __restrict__ indices,
                                                      uint32_t n_cascades,
                                                      float thresh)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    // 1 random number to select the level, 3 to select the position.
    rng.advance(i * 4);
    uint32_t level = (uint32_t) (random_val(rng) * n_cascades) % n_cascades;

    // Select grid cell that has density
    uint32_t idx;
    for (uint32_t j = 0; j < 10; ++j)
    {
        idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % NERF_GRID_N_CELLS();
        idx += level * NERF_GRID_N_CELLS();
        if (grid_in[idx] > thresh)
        {
            break;
        }
    }

    // Random position within that cellq
    uint32_t pos_idx = idx % NERF_GRID_N_CELLS();

    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    vec3 pos = ((vec3{(float) x, (float) y, (float) z} + random_val_3d(rng)) / (float) NERF_GRIDSIZE() - 0.5f) *
                   scalbnf(1.0f, level) +
               0.5f;

    out[i]     = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE())};
    indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements,
                                                             const uint32_t* __restrict__ indices,
                                                             const network_precision_t* network_output,
                                                             float* __restrict__ grid_out,
                                                             ENerfActivation rgb_activation,
                                                             ENerfActivation density_activation)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint32_t local_idx = indices[i];

    // Current setting: optical thickness of the smallest possible stepsize.
    // Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
    uint32_t level = 0; //local_idx / NERF_GRID_N_CELLS();

    float mlp               = network_to_density(float(network_output[i]), density_activation);
    float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

    // Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
    // uint atomicMax is thus perfectly acceptable.
    atomicMax((uint32_t*) &grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void grid_samples_half_to_float(const uint32_t n_elements,
                                           BoundingBox aabb,
                                           float* dst,
                                           const network_precision_t* network_output,
                                           ENerfActivation density_activation,
                                           const NerfPosition* __restrict__ coords_in,
                                           const float* __restrict__ grid_in,
                                           uint32_t max_cascade)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    // let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
    //float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
    float mlp = float(network_output[i]);

    if (grid_in)
    {
        vec3 pos           = unwarp_position(coords_in[i].p, aabb);
        float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos, max_cascade));
        if (grid_density < NERF_MIN_OPTICAL_THICKNESS())
        {
            mlp = -10000.0f;
        }
    }

    dst[i] = mlp;
}

__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
                                      float decay,
                                      const uint32_t count,
                                      float* __restrict__ grid_out,
                                      const float* __restrict__ grid_in)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    float importance = grid_in[i];

    // float ema_debias_old = 1 - (float)powf(decay, count);
    // float ema_debias_new = 1 - (float)powf(decay, count+1);

    // float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
    // grid_out[i] = filtered_val;

    // Maximum instead of EMA allows capture of very thin features.
    // Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

    float prev_val = grid_out[i];
    float val      = (prev_val < 0.f) ? prev_val : fmaxf(prev_val * decay, importance);
    grid_out[i]    = val;
}

__global__ void decay_sharpness_grid_nerf(const uint32_t n_elements, float decay, float* __restrict__ grid)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    grid[i] *= decay;
}

__global__ void grid_to_bitfield(const uint32_t n_elements,
                                 const uint32_t n_nonzero_elements,
                                 const float* __restrict__ grid,
                                 uint8_t* __restrict__ grid_bitfield,
                                 const float* __restrict__ mean_density_ptr)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    if (i >= n_nonzero_elements)
    {
        grid_bitfield[i] = 0;
        return;
    }

    uint8_t bits = 0;

    float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

    NGP_PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j)
    {
        bits |= grid[i * 8 + j] > thresh ? ((uint8_t) 1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
                                  const uint8_t* __restrict__ prev_level,
                                  uint8_t* __restrict__ next_level)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint8_t bits = 0;

    NGP_PRAGMA_UNROLL
    for (uint8_t j = 0; j < 8; ++j)
    {
        // If any bit is set in the previous level, set this
        // level's bit. (Max pooling.)
        bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t) 1 << j) : 0;
    }

    uint32_t x = morton3D_invert(i >> 0) + NERF_GRIDSIZE() / 8;
    uint32_t y = morton3D_invert(i >> 1) + NERF_GRIDSIZE() / 8;
    uint32_t z = morton3D_invert(i >> 2) + NERF_GRIDSIZE() / 8;

    next_level[morton3D(x, y, z)] |= bits;
}

// move to next position along the ray.
__device__ void advance_pos_nerf(NerfPayload& payload,
                                 const BoundingBox& render_aabb,
                                 const mat3& render_aabb_to_local,
                                 const vec3& camera_fwd,
                                 const vec2& focal_length,
                                 uint32_t sample_index,
                                 const uint8_t* __restrict__ density_grid,
                                 uint32_t min_mip,
                                 uint32_t max_mip,
                                 float cone_angle_constant)
{
    if (!payload.alive)
    {
        return;
    }

    vec3 origin = payload.origin;
    vec3 dir    = payload.dir;
    vec3 idir   = vec3(1.0f) / dir;

    // we use a ray cone because the pixel has area!
    float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

    // advance a random amount of steps in logarithmic cone-space.
    float t = advance_n_steps(payload.t, cone_angle, ld_random_val(sample_index, payload.idx * 786433));

    t = if_unoccupied_advance_to_next_occupied_voxel(
        t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);

    if (t >= MAX_DEPTH())
    {
        // terminate the ray if we reach the max depth.
        payload.alive = false;
    }
    else
    {
        payload.t = t;
    }
}

// move to next position along the ray.
__global__ void advance_pos_nerf_kernel(const uint32_t n_elements,
                                        BoundingBox render_aabb,
                                        mat3 render_aabb_to_local,
                                        vec3 camera_fwd,
                                        vec2 focal_length,
                                        uint32_t sample_index,
                                        NerfPayload* __restrict__ payloads,
                                        const uint8_t* __restrict__ density_grid,
                                        uint32_t min_mip,
                                        uint32_t max_mip,
                                        float cone_angle_constant)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    advance_pos_nerf(payloads[i],
                     render_aabb,
                     render_aabb_to_local,
                     camera_fwd,
                     focal_length,
                     sample_index,
                     density_grid,
                     min_mip,
                     max_mip,
                     cone_angle_constant);
}

// move to next position along the ray.
__global__ void step_pos_nerf_kernel(const uint32_t n_elements,
                                     BoundingBox render_aabb,
                                     mat3 render_aabb_to_local,
                                     vec3 camera_fwd,
                                     vec2 focal_length,
                                     uint32_t sample_index,
                                     NerfPayload* __restrict__ payloads,
                                     const uint8_t* __restrict__ density_grid,
                                     uint32_t min_mip,
                                     uint32_t max_mip,
                                     float cone_angle_constant,
                                     float delta_z)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    auto& payload = payloads[i];

    /*if (!payload.alive)
    {
        return;
    }*/

    vec3 origin = payload.origin;
    vec3 dir    = payload.dir;
    vec3 idir   = vec3(1.0f) / dir;

    // we use a ray cone because the pixel has area!
    float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

    // advance a random amount of steps in logarithmic cone-space.
    //int n   = delta_z; // ld_random_val(sample_index, payload.idx * 786433)
    float t =
        advance_n_steps(payload.t, cone_angle, delta_z + 0.2f * ld_random_val(sample_index, payload.idx * 786433));

#if 0
    t = if_unoccupied_advance_to_next_occupied_voxel(
        t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
#endif

    if (t >= MAX_DEPTH())
    {
        // terminate the ray if we reach the max depth.
        payload.alive = false;
    }
    else
    {
        payload.t = t;
    }
}

__global__ void generate_next_nerf_network_inputs_slices(const uint32_t n_elements,
                                                         BoundingBox render_aabb,
                                                         mat3 render_aabb_to_local,
                                                         BoundingBox aabb,
                                                         int sample_index,
                                                         vec2 focal_length,
                                                         vec3 camera_fwd,
                                                         NerfPayload* __restrict__ payloads,
                                                         PitchedPtr<NerfCoordinate> network_input,
                                                         uint32_t n_steps,
                                                         const uint8_t* __restrict__ density_grid,
                                                         uint32_t min_mip,
                                                         uint32_t max_mip,
                                                         float cone_angle_constant,
                                                         const float* extra_dims)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    NerfPayload& payload = payloads[i];
    /*
    if (!payload.alive)
    {
        return;
    }*/

    vec3 origin = payload.origin;
    vec3 dir    = payload.dir;
    vec3 idir   = vec3(1.0f) / dir;

    float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

    float t = payload.t;

    for (uint32_t j = 0; j < n_steps; ++j)
    {
        // calc the step size.
        float dt = calc_dt(t, cone_angle);

        // setup the network for the new position
        network_input(i + j * n_elements)
            ->set_with_optional_extra_dims(warp_position(origin + dir * t, aabb),
                                           warp_direction(dir),
                                           warp_dt(dt),
                                           extra_dims,
                                           network_input.stride_in_bytes); // XXXCONE

        // advance to next voxel.
        //t = if_unoccupied_advance_to_next_occupied_voxel(
        //    t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
        t = advance_n_steps(t, cone_angle, 1 + 0.2f * ld_random_val(sample_index, payload.idx * 786433));

        if (t >= MAX_DEPTH())
        {
            // this ray has left the volume, so record the final step.
            payload.n_steps = j;
            return;
        }

        t += dt;
    }

    payload.t       = t;
    payload.n_steps = n_steps;
}

__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements,
                                                            BoundingBox aabb,
                                                            const vec3* __restrict__ pos,
                                                            PitchedPtr<NerfCoordinate> network_input,
                                                            const float* extra_dims)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    vec3 dir = normalize(pos[i] - 0.5f); // choose outward pointing directions, for want of a better choice
    network_input(i)->set_with_optional_extra_dims(warp_position(pos[i], aabb),
                                                   warp_direction(dir),
                                                   warp_dt(MIN_CONE_STEPSIZE()),
                                                   extra_dims,
                                                   network_input.stride_in_bytes);
}

__global__ void generate_nerf_network_inputs_at_current_position(const uint32_t n_elements,
                                                                 BoundingBox aabb,
                                                                 const NerfPayload* __restrict__ payloads,
                                                                 PitchedPtr<NerfCoordinate> network_input,
                                                                 const float* extra_dims)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    vec3 dir = payloads[i].dir;
    network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb),
                                                   warp_direction(dir),
                                                   warp_dt(MIN_CONE_STEPSIZE()),
                                                   extra_dims,
                                                   network_input.stride_in_bytes);
}

__device__ vec4 compute_nerf_rgba(const vec4& network_output,
                                  ENerfActivation rgb_activation,
                                  ENerfActivation density_activation,
                                  float depth,
                                  bool density_as_alpha = false)
{
    vec4 rgba = network_output;

    float density = network_to_density(rgba.a, density_activation);
    float alpha   = 1.f;
    if (density_as_alpha)
    {
        rgba.a = density;
    }
    else
    {
        rgba.a = alpha = clamp(1.f - __expf(-density * depth), 0.0f, 1.0f);
    }

    rgba.rgb() = network_to_rgb_vec(rgba.rgb(), rgb_activation) * alpha;
    return rgba;
}

__global__ void compute_nerf_rgba_kernel(const uint32_t n_elements,
                                         vec4* network_output,
                                         ENerfActivation rgb_activation,
                                         ENerfActivation density_activation,
                                         float depth,
                                         bool density_as_alpha = false)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    network_output[i] =
        compute_nerf_rgba(network_output[i], rgb_activation, density_activation, depth, density_as_alpha);
}

__global__ void generate_next_nerf_network_inputs(const uint32_t n_elements,
                                                  BoundingBox render_aabb,
                                                  mat3 render_aabb_to_local,
                                                  BoundingBox train_aabb,
                                                  vec2 focal_length,
                                                  vec3 camera_fwd,
                                                  NerfPayload* __restrict__ payloads,
                                                  PitchedPtr<NerfCoordinate> network_input,
                                                  uint32_t n_steps,
                                                  const uint8_t* __restrict__ density_grid,
                                                  uint32_t min_mip,
                                                  uint32_t max_mip,
                                                  float cone_angle_constant,
                                                  const float* extra_dims)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive)
    {
        return;
    }

    vec3 origin = payload.origin;
    vec3 dir    = payload.dir;
    vec3 idir   = vec3(1.0f) / dir;

    float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

    float t = payload.t;

    for (uint32_t j = 0; j < n_steps; ++j)
    {
        // advance to next voxel.
        t = if_unoccupied_advance_to_next_occupied_voxel(
            t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);

        if (t >= MAX_DEPTH())
        {
            // this ray has left the volume, so record the final step.
            payload.n_steps = j;
            return;
        }

        // calc the step size.
        float dt = calc_dt(t, cone_angle);

        // setup the network for the new position
        network_input(i + j * n_elements)
            ->set_with_optional_extra_dims(warp_position(origin + dir * t, train_aabb),
                                           warp_direction(dir),
                                           warp_dt(dt),
                                           extra_dims,
                                           network_input.stride_in_bytes); // XXXCONE

        t += dt;
    }

    payload.t       = t;
    payload.n_steps = n_steps;
}

// for each ray, composite n steps.
__global__ void composite_kernel_nerf(const uint32_t n_elements,
                                      const uint32_t stride,
                                      const uint32_t current_step,
                                      ivec2 resolution,
                                      int slice_count,
                                      BoundingBox aabb,
                                      float glow_y_cutoff,
                                      int glow_mode,
                                      mat4x3 camera_matrix,
                                      vec2 focal_length,
                                      float depth_scale,
                                      RaysNerfSoa::ColorType* __restrict__ rgba,
                                      RaysNerfSoa::DepthType* __restrict__ depth,
                                      NerfPayload* payloads,
                                      PitchedPtr<NerfCoordinate> network_input,
                                      const network_precision_t* __restrict__ network_output,
                                      uint32_t padded_output_width,
                                      uint32_t n_steps,
                                      ERenderMode render_mode,
                                      const uint8_t* __restrict__ density_grid,
                                      ENerfActivation rgb_activation,
                                      ENerfActivation density_activation,
                                      int show_accel,
                                      float min_transmittance)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive)
    {
        return;
    }

    int slice_i = 0;

    // fetch the current framebuffer color...
    vec4 local_rgba   = rgba[i][slice_i];
    float local_depth = depth[i][slice_i];

    float local_depth_max = local_depth;

    vec3 origin  = payload.origin;
    vec3 cam_fwd = camera_matrix[2];

    // Composite n steps.
    // the ray payload contains the amount of available steps of network data.
    // it may be less that what is requested.
    uint32_t actual_n_steps = payload.n_steps;
    uint32_t j              = 0;

    ivec2 debug_pixel = {(int) payload.idx % resolution.x, (int) payload.idx / resolution.x};
    /*
	{
        rgba[i]         = vec4(1, (float)i / n_elements, 0, 1);
		payload.alive   = false;
        payload.n_steps = 0 + current_step;
		return;
	}
	*/

    //if (debug_pixel.x == 100 && debug_pixel.y == 100)
    //    OPENVXL_LOG_VAR3(0, slice_i, local_rgba.a);

    bool ray_terminated = false;

    float depthNear = 0.5f;
    float depthFar  = 1.5f;

    // march n steps...
    for (; j < actual_n_steps; ++j)
    {
        tvec<network_precision_t, 4> local_network_output;
        local_network_output[0]     = network_output[i + j * n_elements + 0 * stride];
        local_network_output[1]     = network_output[i + j * n_elements + 1 * stride];
        local_network_output[2]     = network_output[i + j * n_elements + 2 * stride];
        local_network_output[3]     = network_output[i + j * n_elements + 3 * stride];
        const NerfCoordinate* input = network_input(i + j * n_elements);
        vec3 warped_pos             = input->pos.p;
        vec3 pos                    = unwarp_position(warped_pos, aabb);
        local_depth                 = dot(cam_fwd, pos - camera_matrix[3]);

#if 0

        int slice_index = getSliceIndexFromDepth(local_depth, slice_count, depthNear, depthFar);

        //if (debug_pixel.x == 100 && debug_pixel.y == 100)
        //    OPENVXL_LOG_VAR3(j, local_depth, slice_index);

        // only do one slice at a time. once the slice changes we stop processing this ray.
        if (slice_i != slice_index)
        {
            // finish this slice..
            rgba[i][slice_i]  = local_rgba;
            depth[i][slice_i] = local_depth_max;

            slice_i = slice_index;

            // fetch the current framebuffer color for the slice...
            local_rgba  = rgba[i][slice_i];
            local_depth = depth[i][slice_i];

#if 0
            if (debug_pixel.x == 100 && debug_pixel.y == 100)
                OPENVXL_LOG_VAR3(j, slice_i, local_rgba.a);
#endif
            //break;
        }
		
        // we have to have some criteria for termination, or we will trace to the maxdepth (very slow)!
        if (local_depth > 2)
        {
            //ray_terminated = true;
            //if (debug_pixel.x == 100 && debug_pixel.y == 100)
            //    OPENVXL_LOG_VAR2(ray_terminated, local_depth);
            break;
        }

        //if (sliceIndex != 1)
        //	break;

        //if (i == 100)
        //    OPENVXL_LOG_VAR4(sliceIndex, payload.origin.x, payload.origin.y, payload.origin.z);
#endif

        float T     = 1.f - local_rgba.a;
        float dt    = unwarp_dt(input->dt);
        float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
        if (show_accel >= 0)
        {
            alpha = 1.f;
        }
        float weight = alpha * T;

        vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

        /*
        {
            auto color      = vxl::Vec3f32(rgb.x, rgb.y, rgb.z);
            auto debugTint  = vxl::Vec3f32(vxl::math::rgbaIntToFloat(vxl::fasthash<uint32_t>(slice_index)));
            auto debugAlpha = 0.01f;
            color           = vxl::mix(color, debugTint, debugAlpha);

            rgb = vec3(color[0], color[1], color[2]);
        }
		*/
        if (glow_mode)
        { // random grid visualizations ftw!
#if 0
			if (0) {  // extremely startrek edition
				float glow_y = (pos.y - (glow_y_cutoff - 0.5f)) * 2.f;
				if (glow_y>1.f) glow_y=max(0.f,21.f-glow_y*20.f);
				if (glow_y>0.f) {
					float line;
					line =max(0.f,cosf(pos.y*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.x*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.z*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.y*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.x*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.z*4.f*3.141592653589793f * 16.f)-0.975f);
					glow_y=glow_y*glow_y*0.5f + glow_y*line*25.f;
					rgb.y+=glow_y;
					rgb.z+=glow_y*0.5f;
					rgb.x+=glow_y*0.25f;
				}
			}
#endif
            float glow = 0.f;

            bool green_grid    = glow_mode & 1;
            bool green_cutline = glow_mode & 2;
            bool mask_to_alpha = glow_mode & 4;

            // less used?
            bool radial_mode = glow_mode & 8;
            bool grid_mode   = glow_mode & 16; // makes object rgb go black!

            {
                float dist;
                if (radial_mode)
                {
                    dist = distance(pos, camera_matrix[3]);
                    dist = min(dist, (4.5f - pos.y) * 0.333f);
                }
                else
                {
                    dist = pos.y;
                }

                if (grid_mode)
                {
                    glow = 1.f / max(1.f, dist);
                }
                else
                {
                    float y    = glow_y_cutoff - dist; // - (ii*0.005f);
                    float mask = 0.f;
                    if (y > 0.f)
                    {
                        y *= 80.f;
                        mask = min(1.f, y);
                        //if (mask_mode) {
                        //	rgb.x=rgb.y=rgb.z=mask; // mask mode
                        //} else
                        {
                            if (green_cutline)
                            {
                                glow += max(0.f, 1.f - abs(1.f - y)) * 4.f;
                            }

                            if (y > 1.f)
                            {
                                y = 1.f - (y - 1.f) * 0.05f;
                            }

                            if (green_grid)
                            {
                                glow += max(0.f, y / max(1.f, dist));
                            }
                        }
                    }
                    if (mask_to_alpha)
                    {
                        weight *= mask;
                    }
                }
            }

            if (glow > 0.f)
            {
                float line;
                line = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.y * 8.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.x * 8.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.z * 8.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.y * 16.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.x * 16.f * 3.141592653589793f * 16.f) - 0.975f);
                line += max(0.f, cosf(pos.z * 16.f * 3.141592653589793f * 16.f) - 0.975f);
                if (grid_mode)
                {
                    glow  = /*glow*glow*0.75f + */ glow * line * 15.f;
                    rgb.y = glow;
                    rgb.z = glow * 0.5f;
                    rgb.x = glow * 0.25f;
                }
                else
                {
                    glow = glow * glow * 0.25f + glow * line * 15.f;
                    rgb.y += glow;
                    rgb.z += glow * 0.5f;
                    rgb.x += glow * 0.25f;
                }
            }
        } // glow

        if (render_mode == ERenderMode::Normals)
        {
            // Network input contains the gradient of the network output w.r.t. input.
            // So to compute density gradients, we need to apply the chain rule.
            // The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
            vec3 normal =
                -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
            rgb = normalize(normal);
        }
        else if (render_mode == ERenderMode::Positions)
        {
            rgb = (pos - 0.5f) / 2.0f + 0.5f;
        }
        else if (render_mode == ERenderMode::EncodingVis)
        {
            rgb = warped_pos;
        }
        else if (render_mode == ERenderMode::Depth)
        {
            rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
        }
        else if (render_mode == ERenderMode::AO)
        {
            rgb = vec3(alpha);
        }

        if (show_accel >= 0)
        {
            uint32_t mip = max((uint32_t) show_accel, mip_from_pos(pos));
            uint32_t res = NERF_GRIDSIZE() >> mip;
            int ix       = pos.x * res;
            int iy       = pos.y * res;
            int iz       = pos.z * res;
            default_rng_t rng(ix + iy * 232323 + iz * 727272);
            rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
            rgb.y = rng.next_float();
            rgb.z = rng.next_float();
        }

        local_rgba += vec4(rgb * weight, weight);

        if (weight > payload.max_weight)
        {
            payload.max_weight = weight;
            // get eyeZ for hit pos.
            local_depth     = dot(cam_fwd, pos - camera_matrix[3]);
            local_depth_max = local_depth;

            // get distance from hit pos to camera origin.
            //local_depth = length(pos - camera_matrix[3]);
            //printf("local_depth = %f\n", local_depth);
        }

        if (local_rgba.a > (1.0f - min_transmittance))
        {
            local_rgba /= local_rgba.a;
            //ray_terminated = true;
            break;
        }

        /*
        // terminate early if we reach opacity
        if (slice_i == slice_count - 1 && local_rgba.a > (1.0f - min_transmittance))
        {
            local_rgba /= local_rgba.a;
            ray_terminated = true;
            break;
        }
		*/
    }

#if 1
    // if we are not rendering slices, then we can terminate if a ray
    // breaks because of opacity threshold.
    // but slices need the data even though the ray terminated.

    if (j < n_steps) //if (ray_terminated)
    {
        payload.alive   = false;
        payload.n_steps = j + current_step;
    }
#endif

    rgba[i][slice_i]  = local_rgba;
    depth[i][slice_i] = local_depth_max;
}

// for each ray, composite n steps.
__global__ void composite_kernel_nerf_slice(const uint32_t n_elements,
                                            const uint32_t stride,
                                            const uint32_t current_step,
                                            ivec2 resolution,
                                            int slice_count,
                                            BoundingBox aabb,
                                            mat4x3 camera_matrix,
                                            vec2 focal_length,
                                            float depth_scale,
                                            RaysNerfSoa::ColorType* __restrict__ rgba,
                                            RaysNerfSoa::DepthType* __restrict__ depth,
                                            NerfPayload* payloads,
                                            PitchedPtr<NerfCoordinate> network_input,
                                            const network_precision_t* __restrict__ network_output,
                                            uint32_t padded_output_width,
                                            uint32_t n_steps,
                                            const uint8_t* __restrict__ density_grid,
                                            ENerfActivation rgb_activation,
                                            ENerfActivation density_activation,
                                            float min_transmittance)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    NerfPayload& payload = payloads[i];

    if (!payload.alive)
        return;

    int current_slice_i = -1;

    // fetch the nearest slice color...
    vec4 local_rgba;   //   = rgba[i][current_slice_i];
    float local_depth; // = depth[i][current_slice_i];
    float local_depth_max = 0;

    vec3 origin  = payload.origin;
    vec3 cam_fwd = camera_matrix[2];

    // Composite n steps.
    // the ray payload contains the amount of available steps of network data.
    // it may be less that what is requested.
    uint32_t actual_n_steps = payload.n_steps;
    uint32_t j              = 0;

    ivec2 debug_pixel = {(int) payload.idx % resolution.x, (int) payload.idx / resolution.x};
    /*
	{
        rgba[i]         = vec4(1, (float)i / n_elements, 0, 1);
		payload.alive   = false;
        payload.n_steps = 0 + current_step;
		return;
	}
	*/

    //if (debug_pixel.x == 100 && debug_pixel.y == 100)
    //    OPENVXL_LOG_VAR3(0, slice_i, local_rgba.a);

    bool ray_terminated = false;

    float depthNear = 1.0f;
    float depthFar  = 1.25f;

    // march n steps...
    for (; j < actual_n_steps; ++j)
    {
        tvec<network_precision_t, 4> local_network_output;
        local_network_output[0]     = network_output[i + j * n_elements + 0 * stride];
        local_network_output[1]     = network_output[i + j * n_elements + 1 * stride];
        local_network_output[2]     = network_output[i + j * n_elements + 2 * stride];
        local_network_output[3]     = network_output[i + j * n_elements + 3 * stride];
        const NerfCoordinate* input = network_input(i + j * n_elements);
        vec3 warped_pos             = input->pos.p;
        vec3 pos                    = unwarp_position(warped_pos, aabb);
        float dt                    = unwarp_dt(input->dt);
        local_depth                 = dot(cam_fwd, pos - camera_matrix[3]);

        int slice_index = getSliceIndexFromDepth(local_depth, slice_count, depthNear, depthFar);

#if 1
        if (debug_pixel.x == resolution.x / 2 && debug_pixel.y == resolution.y / 2)
            OPENVXL_LOG_VAR4(j, local_depth, slice_index, dt * 1000.f);
#endif

#if 1

        // only do one slice at a time. once the slice changes we stop processing this ray.
        if (current_slice_i != slice_index)
        {
            if (current_slice_i >= 0)
            {
                // finish this slice..
                rgba[i][current_slice_i]  = local_rgba;
                depth[i][current_slice_i] = local_depth_max;
            }

            // advance to next slice.
            current_slice_i = slice_index;
            local_rgba      = rgba[i][current_slice_i];
            local_depth     = depth[i][current_slice_i];
            local_depth_max = local_depth;

#if 0
            if (debug_pixel.x == resolution.x / 2 && debug_pixel.y == resolution.y / 2)
                OPENVXL_LOG_VAR3(j, current_slice_i, local_rgba.a);
#endif
        }

        // earlier slices might have reached opacity threshold.
        // so we don't need to process them anymore.
        //if (local_rgba.a > 0.75f)
        //     continue;

        /*
        // we have to have some criteria for termination, or we will trace to the maxdepth (very slow)!
        if (local_depth > 2)
        {
            //ray_terminated = true;
            //if (debug_pixel.x == 100 && debug_pixel.y == 100)
            //    OPENVXL_LOG_VAR2(ray_terminated, local_depth);
            break;
        }
		*/
        //if (sliceIndex != 1)
        //	break;

        //if (i == 100)
        //    OPENVXL_LOG_VAR4(sliceIndex, payload.origin.x, payload.origin.y, payload.origin.z);
#endif

        float T     = 1.f - local_rgba.a;
        float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);

#if 0
        if (alpha < 1.0f - min_transmittance)
        {
            // this voxel is very transparent.
            // let's just show the transparent ones.
            local_rgba = vec4(1, 0, 0, 1);
            break;
        }
#endif
        // HACK: make every voxel opaque.
        //alpha = 1.f;

        float weight = alpha * T;

        vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

        // color by the slice...
        {
            auto color      = vxl::Vec3f32(rgb.x, rgb.y, rgb.z);
            auto debugTint  = vxl::Vec3f32(vxl::math::rgbaIntToFloat(vxl::fasthash<uint32_t>(slice_index)));
            auto debugAlpha = 0.3f;
            color           = vxl::mix(color, debugTint, debugAlpha);

            rgb = vec3(color[0], color[1], color[2]);
        }

#if 0
        if (alpha < 1.0f - min_transmittance)
        {
            // this voxel is very transparent.
            // let's just show the transparent ones.
            local_rgba = vec4(rgb, 1);
            break;
        }
#endif

        // accumulate transmitted color...
        local_rgba += vec4(rgb * weight, weight);

        // store the depth value with the maximum weight.
        if (weight > payload.max_weight)
        {
            payload.max_weight = weight;
            local_depth_max    = local_depth; // eye-space Z

            // get distance from hit pos to camera origin.
            //local_depth = length(pos - camera_matrix[3]);
            //printf("local_depth = %f\n", local_depth);
        }

        // if we reach at least (1-min_transmittance) opacity, then we can stop.
        // i.e. boost value it to be fully opaque.
        // NOTE: breaking early will cause ray termination,
        // but note that the alpha is set, so it gets moved to hit buffer

        if (local_rgba.a > 1.0f - min_transmittance)
        {
            local_rgba /= local_rgba.a;
            //ray_terminated = true;

            local_rgba = vec4(1, 0, 0, 1);

            // we can only terminate if we are on the last slice.
            if (current_slice_i >= slice_count - 1)
            {
                //payload.t = MAX_DEPTH();
                //OPENVXL_LOG_VAR3(current_slice_i, debug_pixel.x, debug_pixel.y);
                break;
            }
        }

        /*
        // terminate early if we reach opacity
        if (slice_i == slice_count - 1 && local_rgba.a > (1.0f - min_transmittance))
        {
            local_rgba /= local_rgba.a;
            ray_terminated = true;
            break;
        }
		*/
    }

#if 1
    // if we are not rendering slices, then we can terminate if a ray
    // breaks because of opacity threshold.
    // but slices need the data even though the ray terminated.

    if (j < n_steps) //if (ray_terminated)
    {
        payload.alive   = false;
        payload.n_steps = j + current_step;
    }
#endif

    // store data in the final processed slice.
    rgba[i][current_slice_i]  = vec4(1, 0, 1, 1); //local_rgba;
    depth[i][current_slice_i] = local_depth_max;
}

__global__ void generate_training_samples_nerf(const uint32_t n_rays,
                                               BoundingBox aabb,
                                               const uint32_t max_samples,
                                               const uint32_t n_rays_total,
                                               default_rng_t rng,
                                               uint32_t* __restrict__ ray_counter,
                                               uint32_t* __restrict__ numsteps_counter,
                                               uint32_t* __restrict__ ray_indices_out,
                                               Ray* __restrict__ rays_out_unnormalized,
                                               uint32_t* __restrict__ numsteps_out,
                                               PitchedPtr<NerfCoordinate> coords_out,
                                               const uint32_t n_training_images,
                                               const TrainingImageMetadata* __restrict__ metadata,
                                               const TrainingXForm* training_xforms,
                                               const uint8_t* __restrict__ density_grid,
                                               uint32_t max_mip,
                                               bool max_level_rand_training,
                                               float* __restrict__ max_level_ptr,
                                               bool snap_to_pixel_centers,
                                               bool train_envmap,
                                               float cone_angle_constant,
                                               Buffer2DView<const vec2> distortion,
                                               const float* __restrict__ cdf_x_cond_y,
                                               const float* __restrict__ cdf_y,
                                               const float* __restrict__ cdf_img,
                                               const ivec2 cdf_res,
                                               const float* __restrict__ extra_dims_gpu,
                                               uint32_t n_extra_dims)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays)
        return;

    uint32_t img     = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
    ivec2 resolution = metadata[img].resolution;

    rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
    vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

    // Negative values indicate masked-away regions
    size_t pix_idx = pixel_idx(uv, resolution, 0);
    if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f)
    {
        return;
    }

    float max_level = max_level_rand_training ? (random_val(rng) * 2.0f)
                                              : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

    float motionblur_time = random_val(rng);

    const vec2 focal_length    = metadata[img].focal_length;
    const vec2 principal_point = metadata[img].principal_point;
    const float* extra_dims    = extra_dims_gpu + img * n_extra_dims;
    const Lens lens            = metadata[img].lens;

    const mat4x3 xform =
        get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

    Ray ray_unnormalized;
    const Ray* rays_in_unnormalized = metadata[img].rays;
    if (rays_in_unnormalized)
    {
        // Rays have been explicitly supplied. Read them.
        ray_unnormalized = rays_in_unnormalized[pix_idx];

        /* DEBUG - compare the stored rays to the computed ones
		const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, 0.f);
		Ray ray2;
		ray2.o = xform[3];
		ray2.d = f_theta_distortion(uv, principal_point, lens);
		ray2.d = (xform.block<3, 3>(0, 0) * ray2.d).normalized();
		if (i==1000) {
			printf("\n%d uv %0.3f,%0.3f pixel %0.2f,%0.2f transform from [%0.5f %0.5f %0.5f] to [%0.5f %0.5f %0.5f]\n"
				" origin    [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
				" direction [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
			, img,uv.x, uv.y, uv.x*resolution.x, uv.y*resolution.y,
				training_xforms[img].start[3].x,training_xforms[img].start[3].y,training_xforms[img].start[3].z,
				training_xforms[img].end[3].x,training_xforms[img].end[3].y,training_xforms[img].end[3].z,
				ray_unnormalized.o.x,ray_unnormalized.o.y,ray_unnormalized.o.z,
				ray2.o.x,ray2.o.y,ray2.o.z,
				ray_unnormalized.d.x,ray_unnormalized.d.y,ray_unnormalized.d.z,
				ray2.d.x,ray2.d.y,ray2.d.z);
		}
		*/
    }
    else
    {
        ray_unnormalized = uv_to_ray(0,
                                     uv,
                                     resolution,
                                     focal_length,
                                     xform,
                                     principal_point,
                                     vec3(0.0f),
                                     0.0f,
                                     1.0f,
                                     0.0f,
                                     {},
                                     {},
                                     lens,
                                     distortion);
        if (!ray_unnormalized.is_valid())
        {
            ray_unnormalized = {xform[3], xform[2]};
        }
    }

    vec3 ray_d_normalized = normalize(ray_unnormalized.d);

    vec2 tminmax     = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
    float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

    // The near distance prevents learning of camera-specific fudge right in front of the camera
    tminmax.x = fmaxf(tminmax.x, 0.0f);

    float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
    vec3 idir    = vec3(1.0f) / ray_d_normalized;

    // first pass to compute an accurate number of steps
    uint32_t j = 0;
    float t    = startt;
    vec3 pos;

    while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS())
    {
        float dt     = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos, max_mip);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {
            ++j;
            t += dt;
        }
        else
        {
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
        }
    }
    if (j == 0 && !train_envmap)
    {
        return;
    }
    uint32_t numsteps = j;
    uint32_t base     = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter
    if (base + numsteps > max_samples)
    {
        return;
    }

    coords_out += base;

    uint32_t ray_idx = atomicAdd(ray_counter, 1);

    ray_indices_out[ray_idx]       = i;
    rays_out_unnormalized[ray_idx] = ray_unnormalized;
    numsteps_out[ray_idx * 2 + 0]  = numsteps;
    numsteps_out[ray_idx * 2 + 1]  = base;

    vec3 warped_dir = warp_direction(ray_d_normalized);
    t               = startt;
    j               = 0;
    while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps)
    {
        float dt     = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos, max_mip);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {
            coords_out(j)->set_with_optional_extra_dims(
                warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
            ++j;
            t += dt;
        }
        else
        {
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
        }
    }

    if (max_level_rand_training)
    {
        max_level_ptr += base;
        for (j = 0; j < numsteps; ++j)
        {
            max_level_ptr[j] = max_level;
        }
    }
}

__global__ void compute_loss_kernel_train_nerf(const uint32_t n_rays,
                                               BoundingBox aabb,
                                               const uint32_t n_rays_total,
                                               default_rng_t rng,
                                               const uint32_t max_samples_compacted,
                                               const uint32_t* __restrict__ rays_counter,
                                               float loss_scale,
                                               int padded_output_width,
                                               Buffer2DView<const vec4> envmap,
                                               float* __restrict__ envmap_gradient,
                                               const ivec2 envmap_resolution,
                                               ELossType envmap_loss_type,
                                               vec3 background_color,
                                               EColorSpace color_space,
                                               bool train_with_random_bg_color,
                                               bool train_in_linear_colors,
                                               const uint32_t n_training_images,
                                               const TrainingImageMetadata* __restrict__ metadata,
                                               const network_precision_t* network_output,
                                               uint32_t* __restrict__ numsteps_counter,
                                               const uint32_t* __restrict__ ray_indices_in,
                                               const Ray* __restrict__ rays_in_unnormalized,
                                               uint32_t* __restrict__ numsteps_in,
                                               PitchedPtr<const NerfCoordinate> coords_in,
                                               PitchedPtr<NerfCoordinate> coords_out,
                                               network_precision_t* dloss_doutput,
                                               ELossType loss_type,
                                               ELossType depth_loss_type,
                                               float* __restrict__ loss_output,
                                               bool max_level_rand_training,
                                               float* __restrict__ max_level_compacted_ptr,
                                               ENerfActivation rgb_activation,
                                               ENerfActivation density_activation,
                                               bool snap_to_pixel_centers,
                                               float* __restrict__ error_map,
                                               const float* __restrict__ cdf_x_cond_y,
                                               const float* __restrict__ cdf_y,
                                               const float* __restrict__ cdf_img,
                                               const ivec2 error_map_res,
                                               const ivec2 error_map_cdf_res,
                                               const float* __restrict__ sharpness_data,
                                               ivec2 sharpness_resolution,
                                               float* __restrict__ sharpness_grid,
                                               float* __restrict__ density_grid,
                                               const float* __restrict__ mean_density_ptr,
                                               uint32_t max_mip,
                                               const vec3* __restrict__ exposure,
                                               vec3* __restrict__ exposure_gradient,
                                               float depth_supervision_lambda,
                                               float near_distance)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= *rays_counter)
    {
        return;
    }

    // grab the number of samples for this ray, and the first sample
    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base     = numsteps_in[i * 2 + 1];

    coords_in += base;
    network_output += base * padded_output_width;

    float T = 1.f;

    float EPSILON = 1e-4f;

    vec3 rgb_ray  = vec3(0.0f);
    vec3 hitpoint = vec3(0.0f);

    float depth_ray             = 0.f;
    uint32_t compacted_numsteps = 0;
    vec3 ray_o                  = rays_in_unnormalized[i].o;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        if (T < EPSILON)
        {
            break;
        }

        const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*) network_output;
        const vec3 rgb  = network_to_rgb_vec(local_network_output, rgb_activation);
        const vec3 pos  = unwarp_position(coords_in.ptr->pos.p, aabb);
        const float dt  = unwarp_dt(coords_in.ptr->dt);
        float cur_depth = distance(pos, ray_o);
        float density   = network_to_density(float(local_network_output[3]), density_activation);

        const float alpha  = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;
        hitpoint += weight * pos;
        depth_ray += weight * cur_depth;
        T *= (1.f - alpha);

        network_output += padded_output_width;
        coords_in += 1;
    }
    hitpoint /= (1.0f - T);

    // Must be same seed as above to obtain the same
    // background color.
    uint32_t ray_idx = ray_indices_in[i];
    rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

    float img_pdf    = 1.0f;
    uint32_t img     = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
    ivec2 resolution = metadata[img].resolution;

    float uv_pdf = 1.0f;
    vec2 uv      = nerf_random_image_pos_training(
        rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
    float max_level = max_level_rand_training ? (random_val(rng) * 2.0f)
                                              : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
    rng.advance(1);                                   // motionblur_time

    if (train_with_random_bg_color)
    {
        background_color = random_val_3d(rng);
    }
    vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

    // Composit background behind envmap
    vec4 envmap_value;
    vec3 dir;
    if (envmap)
    {
        dir              = normalize(rays_in_unnormalized[i].d);
        envmap_value     = read_envmap(envmap, dir);
        background_color = envmap_value.rgb() + background_color * (1.0f - envmap_value.a);
    }

    vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
    // vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
    // vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
    vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

    vec3 rgbtarget;
    if (train_in_linear_colors || color_space == EColorSpace::Linear)
    {
        rgbtarget = exposure_scale * texsamp.rgb() + (1.0f - texsamp.a) * background_color;

        if (!train_in_linear_colors)
        {
            rgbtarget        = linear_to_srgb(rgbtarget);
            background_color = linear_to_srgb(background_color);
        }
    }
    else if (color_space == EColorSpace::SRGB)
    {
        background_color = linear_to_srgb(background_color);
        if (texsamp.a > 0)
        {
            rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb() / texsamp.a) * texsamp.a +
                        (1.0f - texsamp.a) * background_color;
        }
        else
        {
            rgbtarget = background_color;
        }
    }

    if (compacted_numsteps == numsteps)
    {
        // support arbitrary background colors
        rgb_ray += T * background_color;
    }

    // Step again, this time computing loss
    network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
    coords_in -= compacted_numsteps;

    uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
    compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
    numsteps_in[i * 2 + 0] = compacted_numsteps;
    numsteps_in[i * 2 + 1] = compacted_base;
    if (compacted_numsteps == 0)
    {
        return;
    }

    max_level_compacted_ptr += compacted_base;
    coords_out += compacted_base;

    dloss_doutput += compacted_base * padded_output_width;

    LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
    lg.loss /= img_pdf * uv_pdf;

    float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth)
                                                                  ? read_depth(uv, resolution, metadata[img].depth)
                                                                  : -1.0f);
    LossAndGradient lg_depth  = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
    float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

    // Note: dividing the gradient by the PDF would cause unbiased loss estimates.
    // Essentially: variance reduction, but otherwise the same optimization.
    // We _dont_ want that. If importance sampling is enabled, we _do_ actually want
    // to change the weighting of the loss function. So don't divide.
    // lg.gradient /= img_pdf * uv_pdf;

    float mean_loss = mean(lg.loss);
    if (loss_output)
    {
        loss_output[i] = mean_loss / (float) n_rays;
    }

    if (error_map)
    {
        const vec2 pos      = clamp(uv * vec2(error_map_res) - 0.5f, 0.0f, vec2(error_map_res) - (1.0f + 1e-4f));
        const ivec2 pos_int = pos;
        const vec2 weight   = pos - vec2(pos_int);

        ivec2 idx = clamp(pos_int, 0, resolution - 2);

        auto deposit_val = [&](int x, int y, float val) {
            atomicAdd(&error_map[img * product(error_map_res) + y * error_map_res.x + x], val);
        };

        if (sharpness_data && aabb.contains(hitpoint))
        {
            ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), 0, sharpness_resolution - 1);
            float sharp         = sharpness_data[img * product(sharpness_resolution) +
                                         sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] +
                          1e-6f;

            // The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
            float grid_sharp = __uint_as_float(
                atomicMax((uint32_t*) &cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)),
                          __float_as_uint(sharp)));
            grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

            mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
        }

        deposit_val(idx.x, idx.y, (1 - weight.x) * (1 - weight.y) * mean_loss);
        deposit_val(idx.x + 1, idx.y, weight.x * (1 - weight.y) * mean_loss);
        deposit_val(idx.x, idx.y + 1, (1 - weight.x) * weight.y * mean_loss);
        deposit_val(idx.x + 1, idx.y + 1, weight.x * weight.y * mean_loss);
    }

    loss_scale /= n_rays;

    const float output_l2_reg         = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
    const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

    // now do it again computing gradients
    vec3 rgb_ray2    = {0.f, 0.f, 0.f};
    float depth_ray2 = 0.f;
    T                = 1.f;
    for (uint32_t j = 0; j < compacted_numsteps; ++j)
    {
        if (max_level_rand_training)
        {
            max_level_compacted_ptr[j] = max_level;
        }
        // Compact network inputs
        NerfCoordinate* coord_out      = coords_out(j);
        const NerfCoordinate* coord_in = coords_in(j);
        coord_out->copy(*coord_in, coords_out.stride_in_bytes);

        const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
        float depth    = distance(pos, ray_o);

        float dt                                                = unwarp_dt(coord_in->dt);
        const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*) network_output;
        const vec3 rgb      = network_to_rgb_vec(local_network_output, rgb_activation);
        const float density = network_to_density(float(local_network_output[3]), density_activation);
        const float alpha   = 1.f - __expf(-density * dt);
        const float weight  = alpha * T;
        rgb_ray2 += weight * rgb;
        depth_ray2 += weight * depth;
        T *= (1.f - alpha);

        // we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
        const vec3 suffix        = rgb_ray - rgb_ray2;
        const vec3 dloss_by_drgb = weight * lg.gradient;

        tvec<network_precision_t, 4> local_dL_doutput;

        // chain rule to go from dloss/drgb to dloss/dmlp_output
        local_dL_doutput[0] =
            loss_scale *
            (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) +
             fmaxf(0.0f, output_l2_reg * (float) local_network_output[0])); // Penalize way too large color values
        local_dL_doutput[1] =
            loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) +
                          fmaxf(0.0f, output_l2_reg * (float) local_network_output[1]));
        local_dL_doutput[2] =
            loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) +
                          fmaxf(0.0f, output_l2_reg * (float) local_network_output[2]));

        float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
        const float depth_suffix = depth_ray - depth_ray2;
        const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

        float dloss_by_dmlp = density_derivative * (dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision));

        //static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
        //dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

        local_dL_doutput[3] = loss_scale * dloss_by_dmlp +
                              (float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
                              (float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
        ;

        *(tvec<network_precision_t, 4>*) dloss_doutput = local_dL_doutput;

        dloss_doutput += padded_output_width;
        network_output += padded_output_width;
    }

    if (exposure_gradient)
    {
        // Assume symmetric loss
        vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

        if (!train_in_linear_colors)
        {
            dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
        }

        // 2^exposure * log(2)
        vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
        atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
        atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
        atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
    }

    if (compacted_numsteps == numsteps && envmap_gradient)
    {
        vec3 loss_gradient = lg.gradient;
        if (envmap_loss_type != loss_type)
        {
            loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
        }

        vec3 dloss_by_dbackground = T * loss_gradient;
        if (!train_in_linear_colors)
        {
            dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
        }

        tvec<network_precision_t, 4> dL_denvmap;
        dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
        dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
        dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;

        float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

        // dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
        dL_denvmap[3] = (network_precision_t) 0;

        deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
    }
}

__global__ void compute_cam_gradient_train_nerf(const uint32_t n_rays,
                                                const uint32_t n_rays_total,
                                                default_rng_t rng,
                                                const BoundingBox aabb,
                                                const uint32_t* __restrict__ rays_counter,
                                                const TrainingXForm* training_xforms,
                                                bool snap_to_pixel_centers,
                                                vec3* cam_pos_gradient,
                                                vec3* cam_rot_gradient,
                                                const uint32_t n_training_images,
                                                const TrainingImageMetadata* __restrict__ metadata,
                                                const uint32_t* __restrict__ ray_indices_in,
                                                const Ray* __restrict__ rays_in_unnormalized,
                                                uint32_t* __restrict__ numsteps_in,
                                                PitchedPtr<NerfCoordinate> coords,
                                                PitchedPtr<NerfCoordinate> coords_gradient,
                                                float* __restrict__ distortion_gradient,
                                                float* __restrict__ distortion_gradient_weight,
                                                const ivec2 distortion_resolution,
                                                vec2* cam_focal_length_gradient,
                                                const float* __restrict__ cdf_x_cond_y,
                                                const float* __restrict__ cdf_y,
                                                const float* __restrict__ cdf_img,
                                                const ivec2 error_map_res)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= *rays_counter)
    {
        return;
    }

    // grab the number of samples for this ray, and the first sample
    uint32_t numsteps = numsteps_in[i * 2 + 0];
    if (numsteps == 0)
    {
        // The ray doesn't matter. So no gradient onto the camera
        return;
    }

    uint32_t base = numsteps_in[i * 2 + 1];
    coords += base;
    coords_gradient += base;

    // Must be same seed as above to obtain the same
    // background color.
    uint32_t ray_idx = ray_indices_in[i];
    uint32_t img     = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
    ivec2 resolution = metadata[img].resolution;

    const mat4x3& xform = training_xforms[img].start;

    Ray ray          = rays_in_unnormalized[i];
    ray.d            = normalize(ray.d);
    Ray ray_gradient = {vec3(0.0f), vec3(0.0f)};

    // Compute ray gradient
    for (uint32_t j = 0; j < numsteps; ++j)
    {
        const vec3 warped_pos   = coords(j)->pos.p;
        const vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);
        ray_gradient.o += pos_gradient;
        const vec3 pos = unwarp_position(warped_pos, aabb);

        // Scaled by t to account for the fact that further-away objects' position
        // changes more rapidly as the direction changes.
        float t                 = distance(pos, ray.o);
        const vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);
        ray_gradient.d += pos_gradient * t + dir_gradient;
    }

    rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
    float uv_pdf = 1.0f;

    vec2 uv = nerf_random_image_pos_training(
        rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &uv_pdf);

    if (distortion_gradient)
    {
        // Projection of the raydir gradient onto the plane normal to raydir,
        // because that's the only degree of motion that the raydir has.
        vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

        // Rotate ray gradient to obtain image plane gradient.
        // This has the effect of projecting the (already projected) ray gradient from the
        // tangent plane of the sphere onto the image plane (which is correct!).
        vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

        // Splat the resulting 2D image plane gradient into the distortion params
        deposit_image_gradient(image_plane_gradient.xy() / uv_pdf,
                               distortion_gradient,
                               distortion_gradient_weight,
                               distortion_resolution,
                               uv);
    }

    if (cam_pos_gradient)
    {
        // Atomically reduce the ray gradient into the xform gradient
        NGP_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; ++j)
        {
            atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
        }
    }

    if (cam_rot_gradient)
    {
        // Rotation is averaged in log-space (i.e. by averaging angle-axes).
        // Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
        // orthogonal, leading to the angle_axis magnitude to equal the magnitude
        // of ray_gradient.d.
        vec3 angle_axis = cross(ray.d, ray_gradient.d);

        // Atomically reduce the ray gradient into the xform gradient
        NGP_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; ++j)
        {
            atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
        }
    }
}

__global__ void compute_extra_dims_gradient_train_nerf(const uint32_t n_rays,
                                                       const uint32_t n_rays_total,
                                                       const uint32_t* __restrict__ rays_counter,
                                                       float* extra_dims_gradient,
                                                       uint32_t n_extra_dims,
                                                       const uint32_t n_training_images,
                                                       const uint32_t* __restrict__ ray_indices_in,
                                                       uint32_t* __restrict__ numsteps_in,
                                                       PitchedPtr<NerfCoordinate> coords_gradient,
                                                       const float* __restrict__ cdf_img)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= *rays_counter)
    {
        return;
    }

    // grab the number of samples for this ray, and the first sample
    uint32_t numsteps = numsteps_in[i * 2 + 0];
    if (numsteps == 0)
    {
        // The ray doesn't matter. So no gradient onto the camera
        return;
    }
    uint32_t base = numsteps_in[i * 2 + 1];
    coords_gradient += base;
    // Must be same seed as above to obtain the same
    // background color.
    uint32_t ray_idx = ray_indices_in[i];
    uint32_t img     = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

    extra_dims_gradient += n_extra_dims * img;

    for (uint32_t j = 0; j < numsteps; ++j)
    {
        const float* src = coords_gradient(j)->get_extra_dims();
        for (uint32_t k = 0; k < n_extra_dims; ++k)
        {
            atomicAdd(&extra_dims_gradient[k], src[k]);
        }
    }
}

__global__ void shade_kernel_nerf(const uint32_t n_elements,
                                  bool gbuffer_hard_edges,
                                  mat4x3 camera_matrix,
                                  float depth_scale,
                                  const RaysNerfSoa::ColorType* __restrict__ rgba,
                                  const RaysNerfSoa::DepthType* __restrict__ depth,
                                  NerfPayload* __restrict__ payloads,
                                  ERenderMode render_mode,
                                  bool train_in_linear_colors,
                                  vec4* __restrict__ frame_buffer,
                                  float* __restrict__ depth_buffer)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements || render_mode == ERenderMode::Distortion)
        return;
    NerfPayload& payload = payloads[i];

    const int slice_i = 0;

    vec4 tmp = rgba[i][slice_i];
    if (render_mode == ERenderMode::Normals)
    {
        vec3 n    = normalize(tmp.xyz());
        tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
    }
    else if (render_mode == ERenderMode::Cost)
    {
        float col = (float) payload.n_steps / 128;
        tmp       = {col, col, col, 1.0f};
    }
    else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth)
    {
        tmp.rgb() = vec3(depth[i][slice_i] * depth_scale);
    }
    else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions)
    {
        vec3 pos  = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i][slice_i];
        tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
    }

    if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice))
    {
        // Accumulate in linear colors
        tmp.rgb() = srgb_to_linear(tmp.rgb());
    }

    // HACK: we will bend "slice" render mode to our bidding!
    if (render_mode == ERenderMode::Slice)
    {
        vxl::Vec3f32 color(1, 0, 1);

        // if it is alive or has any visible value...
        if (payload.alive || tmp.a > 0.f)
        {
            {
                // NOTE: t depends on nearClip, but not farClip! It gors from [nearClip -> render_bbox end?]
                color = vxl::pseudoTemperature(payload.t);
                // this ray reached the end, so show black!
                if (payload.t > MAX_DEPTH() - 0.001f)
                    color = vxl::Vec3f32(0);
            }
            /*
            {
                // NOTE: dir is a normalized world-space ray direction
                color = vxl::Vec3f32{payload.dir.x, payload.dir.y, payload.dir.z};
            }

            {
                // NOTE: just normalize by a max step count of 128
                float unitCost = (float) payload.n_steps / 128;
                color          = vxl::pseudoTemperature(unitCost);
            }

            {
                // just use the composited rgb.
                color = {tmp.r, tmp.g, tmp.b};
            }
			*/
        }

        tmp = vec4(color[0], color[1], color[2], 1.f);
    }

    frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);

    if (render_mode != ERenderMode::Slice && tmp.a > 0.2f)
    {
        depth_buffer[payload.idx] = depth[i][slice_i];
    }
}

//----------------------------------------------------------------------------------------------
// map quilt-tile index to offset angle.
OPENVXL_CUDA_INLINE vxl::Vec2f32
    getPlaneOffset(int vi, int vj, int vx, int vy, float halfViewFovX, float halfViewFovY, float focalDistance)
{
#if 1
    // the looking glass quilt format...
    // 6 7 8
    // 3 4 5
    // 0 1 2
    // where value is xoffset starting from left-most.
    int n = (vx * vy - 1);
    if (n == 0)
        return {0, 0};
    int v_offset    = vi + vj * vx;
    float tileU     = (float) v_offset / n;
    float offAngleX = tileU * (2.0f * halfViewFovX) - halfViewFovX;
    return {vxl::tan(offAngleX) * focalDistance, 0.f};
#else

    // just a patch work... but not the format!
    vxl::swap(vx, vy);

    float offAngleY = 0.f;
    float offAngleX = 0.f;
    if (vx > 1)
    {
        float tileU = (float) vi / (vx - 1);
        offAngleX   = tileU * (2.0f * halfViewFovX) - halfViewFovX;
    }
    if (vy > 1)
    {
        float tileV = (float) vj / (vy - 1);
        offAngleY   = tileV * (2.0f * halfViewFovY) - halfViewFovY;
    }
    // get the 2d offset on the focal plane from the view ray...
    float offsetX = vxl::tan(offAngleX) * focalDistance;
    float offsetY = vxl::tan(offAngleY) * focalDistance;
    return {offsetX, offsetY};
#endif

    return {0, 0};
}

__global__ void kernel_slice_blend(uint32_t n_elements,
                                   ivec2 quiltResolution,
                                   int vx,
                                   int vy,
                                   vec4* __restrict__ frame_buffer,
                                   float* __restrict__ depth_buffer,
                                   float alphaThreshold,

                                   float focusDistance, // viewport focusdistance
                                   float halfViewFovX,  // viewport camera
                                   float halfViewFovY,  // viewport camera
                                   float refTanFovX,    // reference camera
                                   float refTanFovY,    // reference camera

                                   float baseTanFovX,
                                   float baseTanFovY,
                                   float refForwardDist,
                                   float sliceBatchDistance,
                                   ivec2 sliceResolution,
                                   int sliceBatchCount,
                                   int sliceCount,
                                   const vec4* __restrict__ sliceColorPtr)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    int viewWidth  = quiltResolution.x / vx;
    int viewHeight = quiltResolution.y / vy;
    int srcWidth   = sliceResolution.x;
    int srcHeight  = sliceResolution.y;

    float viewAspect    = (float) viewWidth / viewHeight;
    float viewAspectInv = 1.f / viewAspect;

    // get quilt coordinates...
    int quiltX = i % quiltResolution.x;
    int quiltY = i / quiltResolution.x;
    int vx_i   = (quiltX / viewWidth);
    int vy_i   = (quiltY / viewHeight);
    int view_i = quiltX % viewWidth;
    int view_j = quiltY % viewHeight;

    // setup slice-plane intersection vars...
    auto planeOffset = getPlaneOffset(vx_i, vy_i, vx, vy, halfViewFovX, halfViewFovY, focusDistance);
    float view_u     = ((view_i + 0.5f) / viewWidth) * 2.f - 1.f;  // -1 to +1
    float view_v     = ((view_j + 0.5f) / viewHeight) * 2.f - 1.f; // -1 to +1
    float mx         = baseTanFovX * view_u - planeOffset[0] / focusDistance;
    float my         = baseTanFovY * view_v - planeOffset[1] / focusDistance;

    //float offsetX = focusDistance * vxl::tan(offAngleX);
    //float offsetY = focusDistance * vxl::tan(offAngleY);
    //float mx = baseTanFovX * view_u - vxl::tan(offAngleX);
    //float my = baseTanFovY * view_v - vxl::tan(offAngleY);

    // NOTE: we stored the additive-inverse in framebuffer alpha.
    vec4 dstColor = frame_buffer[i];

    float accumT    = 1.0f - dstColor.a;
    vec3 accumColor = dstColor.rgb();

    //frame_buffer[i] = vec4(vec3(view_u, view_v, 0), 1.f);
    //frame_buffer[i] = vec4(vec3((float) vx_i / vx, (float) vy_i / vy, 0), 1.f);
    //return;
#if 1
    const float kSliceAlphaThreshold = 0.001f;

    for (int k = 0; k < sliceBatchCount && accumT > kSliceAlphaThreshold; ++k)
    {
        // get distance to slice...
        float planeDist = (sliceBatchDistance + 0 * k);

        if (sliceCount == 1)
            planeDist = focusDistance;

        // get normalized coordinates on plane (-1 to +1)...
        float base_x = (planeDist - refForwardDist) * refTanFovX;
        float base_y = (planeDist - refForwardDist) * refTanFovY;
        float srcU   = (planeOffset[0] + mx * planeDist) / base_x;
        float srcV   = (planeOffset[1] + my * planeDist) / base_y;

        // fetch source data...
        int srcImageX = int((srcU * 0.5f + 0.5f) * srcWidth - 0.5f);
        int srcImageY = int((srcV * 0.5f + 0.5f) * srcHeight - 0.5f);
        if (srcImageX < 0 || srcImageX >= srcWidth || srcImageY < 0 || srcImageY >= srcHeight)
            continue;

        //frame_buffer[i] = vec4((float) srcImageX / srcWidth, (float) srcImageY / srcHeight, 0, 1.f);
        //return;

        int srcSliceOffset = (k * srcWidth * srcHeight);
        int srcPixelOffset = srcSliceOffset + (srcImageX + srcWidth * srcImageY);

        auto sliceColor = sliceColorPtr[srcPixelOffset];
        if (sliceColor.a < 0.001f)
            continue;

        float srcTransmission = 1.0f - sliceColor.a;

        accumColor += accumT * (sliceColor.rgb() * sliceColor.a);
        accumT *= srcTransmission;
    }

    frame_buffer[i] = vec4(accumColor, 1.f - accumT);
    depth_buffer[i] = 0.f;

    //frame_buffer[i] = vec4(vec3((view_i + 0.5f) / viewWidth, (view_j + 0.5f) / viewHeight, 0), 1.f);
#endif
}

__global__ void shade_kernel_nerf_accumulate_slice(uint32_t n_elements,
                                                   uint32_t stride,
                                                   ivec2 resolution,
                                                   BoundingBox aabb,
                                                   int slice_i,
                                                   float depth_scale,
                                                   const NerfPayload* __restrict__ payloads,
                                                   const vec4* __restrict__ rgba,
                                                   const float* __restrict__ depth,

                                                   PitchedPtr<NerfCoordinate> network_input,
                                                   const network_precision_t* __restrict__ network_output,
                                                   uint32_t padded_output_width,
                                                   ENerfActivation rgb_activation,
                                                   ENerfActivation density_activation,
                                                   uint32_t n_steps,

                                                   bool train_in_linear_colors,
                                                   vec4* __restrict__ frame_buffer,
                                                   float* __restrict__ depth_buffer)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    const NerfPayload& payload = payloads[i];

    // scatter coordinates...
    /*
    int dstX = payload.idx % resolution.x; // [0 - 256)
    int dstY = payload.idx / resolution.x;
    dstX          = vxl::clamp(dstX, 0, resolution.x - 1); // [0 - 255]
    dstY          = vxl::clamp(dstY, 0, resolution.y - 1);
    int dstOffset = dstX + dstY * resolution.x;
	*/
    int dstOffset = payload.idx;

    if (!rgba)
    {
        vec4 local_rgba(0.f, 0, 0, 1);

        for (int j = 0; j < n_steps; ++j)
        {
            tvec<network_precision_t, 4> local_network_output;
            local_network_output[0]     = network_output[i + j * n_elements + 0 * stride];
            local_network_output[1]     = network_output[i + j * n_elements + 1 * stride];
            local_network_output[2]     = network_output[i + j * n_elements + 2 * stride];
            local_network_output[3]     = network_output[i + j * n_elements + 3 * stride];
            const NerfCoordinate* input = network_input(i + j * n_elements);
            vec3 warped_pos             = input->pos.p;
            vec3 pos                    = unwarp_position(warped_pos, aabb);

            float dt    = unwarp_dt(input->dt);
            float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
            vec3 rgb    = network_to_rgb_vec(local_network_output, rgb_activation);
            if (!train_in_linear_colors)
                rgb = srgb_to_linear(rgb);

            float T = local_rgba.a;

            // skip very transparent pixels in this slice...
            if (alpha < 0.001f)
                continue;

            local_rgba.rgb() = rgb * alpha * local_rgba.a;
            local_rgba.a *= 1.0f - alpha;
        }

        frame_buffer[dstOffset + 0 * resolution.x * resolution.y] = local_rgba;
    }
    else
    {
        vec4 srcColor = rgba[i];
        if (!train_in_linear_colors)
            srcColor.rgb() = srgb_to_linear(srcColor.rgb());

        {
            auto color      = vxl::Vec3f32(srcColor.x, srcColor.y, srcColor.z);
            auto debugTint  = vxl::Vec3f32(vxl::math::rgbaIntToFloat(vxl::fasthash<uint32_t>(slice_i)));
            auto debugAlpha = 0.0f;
            color           = vxl::mix(color, debugTint, debugAlpha);

            srcColor.rgb() = vec3(color[0], color[1], color[2]);
        }

        //vec4 prevColor = frame_buffer[dstOffset];

        //float T    = 1.f - prevColor.a;
        //float newT = T * (1.f - srcColor.a);

        //vec4 newColor = vec4(prevColor.rgb() + srcColor.rgb() * srcColor.a * T, 1.f - newT);

        frame_buffer[dstOffset] = srcColor;
    }
}

__global__ void shade_kernel_nerf_slice(const uint32_t n_elements,
                                        const uint32_t stride,
                                        ivec2 resolution,
                                        int slice_i,
                                        int mosaicSize,
                                        bool gbuffer_hard_edges,
                                        mat4x3 camera_matrix,
                                        float depth_scale,
                                        const vec4* __restrict__ rgba,
                                        const float* __restrict__ depth,
                                        NerfPayload* __restrict__ payloads,
                                        ERenderMode render_mode,

                                        BoundingBox aabb,
                                        PitchedPtr<NerfCoordinate> network_input,
                                        const network_precision_t* __restrict__ network_output,
                                        uint32_t padded_output_width,
                                        ENerfActivation rgb_activation,
                                        ENerfActivation density_activation,
                                        uint32_t n_steps,

                                        bool train_in_linear_colors,
                                        vec4* __restrict__ frame_buffer,
                                        float* __restrict__ depth_buffer)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements || render_mode == ERenderMode::Distortion)
        return;
    NerfPayload& payload = payloads[i];

    vec4 local_rgba(0.f);

    for (int j = 0; j < n_steps; ++j)
    {
        tvec<network_precision_t, 4> local_network_output;
        local_network_output[0]     = network_output[i + j * n_elements + 0 * stride];
        local_network_output[1]     = network_output[i + j * n_elements + 1 * stride];
        local_network_output[2]     = network_output[i + j * n_elements + 2 * stride];
        local_network_output[3]     = network_output[i + j * n_elements + 3 * stride];
        const NerfCoordinate* input = network_input(i + j * n_elements);
        vec3 warped_pos             = input->pos.p;
        vec3 pos                    = unwarp_position(warped_pos, aabb);

        float dt    = unwarp_dt(input->dt);
        float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
        vec3 rgb    = network_to_rgb_vec(local_network_output, rgb_activation);

        //float T = 1.f - local_rgba.a;
        local_rgba.a     = alpha;
        local_rgba.rgb() = rgb * alpha;
        if (!train_in_linear_colors)
            local_rgba.rgb() = srgb_to_linear(local_rgba.rgb());
    }

    vec4 srcColor = local_rgba;
    if (!train_in_linear_colors && render_mode == ERenderMode::Slice)
        srcColor.rgb() = srgb_to_linear(srcColor.rgb());

    int srcX = payload.idx % resolution.x; // [0 - 256)
    int srcY = payload.idx / resolution.x;

    float srcU       = fmodf(srcX, resolution.x) / (resolution.x - 1); // [0 - 1)
    float srcV       = fmodf(srcY, resolution.y) / (resolution.y - 1);
    int sliceMosaicX = slice_i % mosaicSize; // [0 - 3]
    int sliceMosaicY = slice_i / mosaicSize;
    int dstW         = resolution.x / mosaicSize; // 256/4 = 64
    int dstH         = resolution.y / mosaicSize;
    int dstX         = dstW * (sliceMosaicX + srcU); // 64 * [0 - 3.9999]
    int dstY         = dstH * (sliceMosaicY + srcV);

    dstX          = vxl::clamp(dstX, 0, resolution.x - 1); // [0 - 255]
    dstY          = vxl::clamp(dstY, 0, resolution.y - 1);
    int dstOffset = dstX + dstY * resolution.x;

    {
        auto color      = vxl::Vec3f32(srcColor.x, srcColor.y, srcColor.z);
        auto debugTint  = vxl::Vec3f32(vxl::math::rgbaIntToFloat(vxl::fasthash<uint32_t>(slice_i)));
        auto debugAlpha = 0.0f;
        color           = vxl::mix(color, debugTint, debugAlpha);

        srcColor.rgb() = vec3(color[0], color[1], color[2]);
    }

    frame_buffer[dstOffset] = srcColor; // + frame_buffer[dstOffset] * (1.0f - srcColor.a);
    //frame_buffer[dstOffset] = vec4(srcU, srcV, 0.f, 1.f);
    /*
    if (render_mode != ERenderMode::Slice && srcColor.a > 0.2f)
    {
        depth_buffer[payload.idx] = depth[i];
    }*/
}

__global__ void shade_kernel_nerf_draw_slices(const uint32_t n_elements,
                                              ivec2 resolution,
                                              int slice_count,
                                              int mosaicSize,
                                              bool gbuffer_hard_edges,
                                              mat4x3 camera_matrix,
                                              float depth_scale,
                                              const RaysNerfSoa::ColorType* __restrict__ rgba,
                                              const RaysNerfSoa::DepthType* __restrict__ depth,
                                              const NerfPayload* __restrict__ payloads,
                                              ERenderMode render_mode,
                                              bool train_in_linear_colors,
                                              vec4* __restrict__ frame_buffer,
                                              float* __restrict__ depth_buffer)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    int slice_i                = i % slice_count;
    int ray_i                  = i / slice_count;
    const NerfPayload& payload = payloads[ray_i];

    vec4 srcColor = rgba[ray_i][slice_i];
    if (!train_in_linear_colors && render_mode == ERenderMode::Slice)
        srcColor.rgb() = srgb_to_linear(srcColor.rgb());

    int srcX = payload.idx % resolution.x; // [0 - 256)
    int srcY = payload.idx / resolution.x;

    float srcU       = fmodf(srcX, resolution.x) / (resolution.x - 1); // [0 - 1)
    float srcV       = fmodf(srcY, resolution.y) / (resolution.y - 1);
    int sliceMosaicX = slice_i % mosaicSize; // [0 - 3]
    int sliceMosaicY = slice_i / mosaicSize;
    int dstW         = resolution.x / mosaicSize; // 256/4 = 64
    int dstH         = resolution.y / mosaicSize;
    int dstX         = dstW * (sliceMosaicX + srcU); // 64 * [0 - 3.9999]
    int dstY         = dstH * (sliceMosaicY + srcV);

    dstX          = vxl::clamp(dstX, 0, resolution.x - 1); // [0 - 255]
    dstY          = vxl::clamp(dstY, 0, resolution.y - 1);
    int dstOffset = dstX + dstY * resolution.x;

    {
        auto color      = vxl::Vec3f32(srcColor.x, srcColor.y, srcColor.z);
        auto debugTint  = vxl::Vec3f32(vxl::math::rgbaIntToFloat(vxl::fasthash<uint32_t>(slice_i)));
        auto debugAlpha = 0.0f;
        color           = vxl::mix(color, debugTint, debugAlpha);

        srcColor.rgb() = vec3(color[0], color[1], color[2]);
    }

    srcColor.a              = 1.0f;
    frame_buffer[dstOffset] = srcColor; // + frame_buffer[dstOffset] * (1.0f - srcColor.a);
    //frame_buffer[dstOffset] = vec4(mosaicU, mosaicV, 0.f, 1.f);

    //frame_buffer[dstOffset] = vec4(srcColor.a,srcColor.a,srcColor.a,1.0f);

    if (srcColor.a > 0.2f)
    {
        depth_buffer[payload.idx] = depth[ray_i][slice_i];
    }

    // HACK: raw the center pixel..
    int centerX                                            = dstW * (sliceMosaicX + 0.5f);
    int centerY                                            = dstH * (sliceMosaicY + 0.5f);
    frame_buffer[(centerX) + (centerY) *resolution.x]      = vec4(1, 1, 1, 1);
    frame_buffer[(centerX) + (centerY + 1) * resolution.x] = vec4(0, 0, 0, 1);
    frame_buffer[(centerX) + (centerY - 1) * resolution.x] = vec4(0, 0, 0, 1);
    frame_buffer[(centerX - 1) + (centerY) *resolution.x]  = vec4(0, 0, 0, 1);
    frame_buffer[(centerX + 1) + (centerY) *resolution.x]  = vec4(0, 0, 0, 1);
}

__global__ void shade_kernel_nerf_2d(const uint32_t n_elements,
                                     bool gbuffer_hard_edges,
                                     mat4x3 camera_matrix,
                                     float depth_scale,
                                     const vec4* __restrict__ rgba,
                                     const float* __restrict__ depth,
                                     NerfPayload* __restrict__ payloads,
                                     ERenderMode render_mode,
                                     bool train_in_linear_colors,
                                     vec4* __restrict__ frame_buffer,
                                     float* __restrict__ depth_buffer)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements || render_mode == ERenderMode::Distortion)
        return;
    NerfPayload& payload = payloads[i];

    vec4 tmp = rgba[i];
    if (render_mode == ERenderMode::Normals)
    {
        vec3 n    = normalize(tmp.xyz());
        tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
    }
    else if (render_mode == ERenderMode::Cost)
    {
        float col = (float) payload.n_steps / 128;
        tmp       = {col, col, col, 1.0f};
    }
    else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth)
    {
        tmp.rgb() = vec3(depth[i] * depth_scale);
    }
    else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions)
    {
        vec3 pos  = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i];
        tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
    }

    if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice))
    {
        // Accumulate in linear colors
        tmp.rgb() = srgb_to_linear(tmp.rgb());
    }

    frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);

    if (render_mode != ERenderMode::Slice && tmp.a > 0.2f)
    {
        depth_buffer[payload.idx] = depth[i];
    }
}

__global__ void compact_kernel_nerf(const uint32_t n_elements,
                                    const RaysNerfSoa::ColorType* src_rgba,
                                    const RaysNerfSoa::DepthType* src_depth,
                                    const NerfPayload* src_payloads,
                                    RaysNerfSoa::ColorType* dst_rgba,
                                    RaysNerfSoa::DepthType* dst_depth,
                                    NerfPayload* dst_payloads,
                                    RaysNerfSoa::ColorType* dst_final_rgba,
                                    RaysNerfSoa::DepthType* dst_final_depth,
                                    NerfPayload* dst_final_payloads,
                                    uint32_t* aliveCounter,
                                    uint32_t* finalCounter)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    const NerfPayload& src_payload = src_payloads[i];

    if (src_payload.alive)
    {
        uint32_t idx      = atomicAdd(aliveCounter, 1);
        dst_payloads[idx] = src_payload;

        for (int slice_i = 0; slice_i < RaysNerfSoa::ColorType::kSize; ++slice_i)
        {
            dst_rgba[idx][slice_i]  = src_rgba[i][slice_i];
            dst_depth[idx][slice_i] = src_depth[i][slice_i];
        }
    }
    else if (src_rgba[i][0].a > 0.001f)
    {
        // the ray is finished!
        // so if the ray is not alive anymore and if the ray's opacity is enough,
        // then put this in the "hit" buffer.

        uint32_t idx            = atomicAdd(finalCounter, 1);
        dst_final_payloads[idx] = src_payload;

        for (int slice_i = 0; slice_i < RaysNerfSoa::ColorType::kSize; ++slice_i)
        {
            dst_final_rgba[idx][slice_i]  = src_rgba[i][slice_i];
            dst_final_depth[idx][slice_i] = src_depth[i][slice_i];
        }
    }
}

__global__ void debug_rays_kernel(const uint32_t n_elements,
                                  float plane_z,
                                  const RaysNerfSoa::ColorType* src_rgba,
                                  const RaysNerfSoa::DepthType* src_depth,
                                  const NerfPayload* src_payloads,
                                  RaysNerfSoa::ColorType* dst_final_rgba,
                                  RaysNerfSoa::DepthType* dst_final_depth,
                                  NerfPayload* dst_final_payloads,
                                  uint32_t* counter,
                                  uint32_t* finalCounter)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    const NerfPayload& src_payload = src_payloads[i];

    {
        // finish the ray.
        // so if the ray is not alive anymore and if the ray's opacity is enough,
        // then put this in the "hit" buffer.

        uint32_t idx            = atomicAdd(finalCounter, 1);
        dst_final_payloads[idx] = src_payload;
        //dst_final_payloads[idx].alive = false;

        // HACK: for slice just set it back to the plane-z.
        {
            float n = length(dst_final_payloads[idx].dir);
            //dst_final_payloads.origin = ray.o;
            //dst_final_payloads.dir = (1.0f / n) * ray.d;
            dst_final_payloads[idx].t = vxl::abs(plane_z) * n;
        }

        for (int slice_i = 0; slice_i < RaysNerfSoa::ColorType::kSize; ++slice_i)
        {
            dst_final_rgba[idx][slice_i]  = src_rgba[i][slice_i];
            dst_final_depth[idx][slice_i] = src_depth[i][slice_i];
        }

        if (src_payload.t == MAX_DEPTH())
        {
            OPENVXL_LOG_VAR(src_payload.t);
        }
    }
}

// for every pixel in the output image, build the ray.
// we also setup the depthbuffer.
__global__ void init_rays_with_payload_kernel_nerf(uint32_t sample_index,
                                                   NerfPayload* __restrict__ payloads,
                                                   ivec2 resolution,
                                                   vec2 focal_length,
                                                   mat4x3 camera_matrix0,
                                                   mat4x3 camera_matrix1,
                                                   vec4 rolling_shutter,
                                                   vec2 screen_center,
                                                   vec3 parallax_shift,
                                                   bool snap_to_pixel_centers,
                                                   BoundingBox render_aabb,
                                                   mat3 render_aabb_to_local,
                                                   float near_distance,
                                                   float plane_z,
                                                   float aperture_size,
                                                   Foveation foveation,
                                                   Lens lens,
                                                   Buffer2DView<const vec4> envmap,
                                                   vec4* __restrict__ frame_buffer,
                                                   float* __restrict__ depth_buffer,
                                                   Buffer2DView<const uint8_t> hidden_area_mask,
                                                   Buffer2DView<const vec2> distortion,
                                                   ERenderMode render_mode)
{
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= resolution.x || y >= resolution.y)
    {
        return;
    }

    uint32_t idx = x + resolution.x * y;

    if (plane_z < 0)
    {
        aperture_size = 0.0;
    }

    // get the uv from the (optionally jittered) pixel position...
    vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
    vec2 uv           = vec2{(float) x + pixel_offset.x, (float) y + pixel_offset.y} / vec2(resolution);

    mat4x3 camera = get_xform_given_rolling_shutter(
        {camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

    // get the ray for the uv position...
    Ray ray = uv_to_ray(sample_index,
                        uv,
                        resolution,
                        focal_length,
                        camera,
                        screen_center,
                        parallax_shift,
                        near_distance,
                        plane_z,
                        aperture_size,
                        foveation,
                        hidden_area_mask,
                        lens,
                        distortion);

    // initialize the payload...
    NerfPayload& payload = payloads[idx];
    payload.max_weight   = 0.0f;

    // initialize depth here. as good a place as any.
    depth_buffer[idx] = MAX_DEPTH();

    // if the ray is invalid, then kill it...
    if (!ray.is_valid())
    {
        payload.origin = ray.o;
        payload.alive  = false;
        return;
    }

    // if we specified a plane_z then set the ray to start there...
    // we terminate the ray to ensure it immediately get put in the rays_hit buffer.
    if (plane_z < 0)
    {
        float n           = length(ray.d);
        payload.origin    = ray.o;
        payload.dir       = (1.0f / n) * ray.d;
        payload.t         = -plane_z * n;
        payload.idx       = idx;
        payload.n_steps   = 0;
        payload.alive     = false;
        depth_buffer[idx] = -plane_z;
        return;
    }

#if 1
    if (render_mode == ERenderMode::Distortion)
    {
        // draw the distortion as color...
        // We kill the ray so no other ray processing happens,
        // i.e. This is the final output for this framebuffer pixel.
        vec2 uv_after_distortion =
            pos_to_uv(ray(1.0f), resolution, focal_length, camera, screen_center, parallax_shift, foveation);

        frame_buffer[idx].rgb() = to_rgb((uv_after_distortion - uv) * 64.0f);
        frame_buffer[idx].a     = 1.0f;
        depth_buffer[idx]       = 1.0f;
        payload.origin          = ray(MAX_DEPTH());
        payload.alive           = false;
        return;
    }
#endif

    ray.d = normalize(ray.d);

    if (envmap)
    {
        // initialize the framebuffer to a spherical environment map.
        frame_buffer[idx] = read_envmap(envmap, ray.d);
    }

    // intersect the ray with the "render" bounds.
    // HACK: we boost it by a tiny amount for numerical safety.
    float t =
        fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

    // QUESTION: why are we checking if we are in bounds, if we just did this in the line above!
    if (!render_aabb.contains(render_aabb_to_local * ray(t)))
    {
        payload.origin = ray.o;
        payload.t      = t;   // for debugging.
        payload.idx    = idx; // for debugging.
        payload.alive  = false;
        return;
    }

    // now we have pruned everything, so setup the ray...
    payload.origin  = ray.o;
    payload.dir     = ray.d;
    payload.t       = t;
    payload.idx     = idx;
    payload.n_steps = 0;
    payload.alive   = true;
}

static constexpr float MIN_PDF = 0.01f;

__global__ void construct_cdf_2d(uint32_t n_images,
                                 uint32_t height,
                                 uint32_t width,
                                 const float* __restrict__ data,
                                 float* __restrict__ cdf_x_cond_y,
                                 float* __restrict__ cdf_y)
{
    const uint32_t y   = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= height || img >= n_images)
        return;

    const uint32_t offset_xy = img * height * width + y * width;
    data += offset_xy;
    cdf_x_cond_y += offset_xy;

    float cum = 0;
    for (uint32_t x = 0; x < width; ++x)
    {
        cum += data[x] + 1e-10f;
        cdf_x_cond_y[x] = cum;
    }

    cdf_y[img * height + y] = cum;
    float norm              = __frcp_rn(cum);

    for (uint32_t x = 0; x < width; ++x)
    {
        cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float) (x + 1) / (float) width;
    }
}

__global__ void construct_cdf_1d(uint32_t n_images,
                                 uint32_t height,
                                 float* __restrict__ cdf_y,
                                 float* __restrict__ cdf_img)
{
    const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
    if (img >= n_images)
        return;

    cdf_y += img * height;

    float cum = 0;
    for (uint32_t y = 0; y < height; ++y)
    {
        cum += cdf_y[y];
        cdf_y[y] = cum;
    }

    cdf_img[img] = cum;

    float norm = __frcp_rn(cum);
    for (uint32_t y = 0; y < height; ++y)
    {
        cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float) (y + 1) / (float) height;
    }
}

__global__ void safe_divide(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_elements)
        return;

    float local_divisor = divisor[i];
    inout[i]            = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
}

//----------------------------------------------------------------------------------------------
void Testbed::NerfTracer::init_rays_from_camera(uint32_t sample_index,
                                                uint32_t padded_output_width,
                                                uint32_t n_extra_dims,
                                                const ivec2& resolution,
                                                int slice_count,
                                                const vec2& focal_length,
                                                const mat4x3& camera_matrix0,
                                                const mat4x3& camera_matrix1,
                                                const vec4& rolling_shutter,
                                                const vec2& screen_center,
                                                const vec3& parallax_shift,
                                                bool snap_to_pixel_centers,
                                                const BoundingBox& render_aabb,
                                                const mat3& render_aabb_to_local,
                                                float near_distance,
                                                float plane_z,
                                                float aperture_size,
                                                const Foveation& foveation,
                                                const Lens& lens,
                                                const Buffer2DView<const vec4>& envmap,
                                                const Buffer2DView<const vec2>& distortion,
                                                vec4* frame_buffer,
                                                float* depth_buffer,
                                                const Buffer2DView<const uint8_t>& hidden_area_mask,
                                                const uint8_t* grid,
                                                int show_accel,
                                                uint32_t max_mip,
                                                float cone_angle_constant,
                                                ERenderMode render_mode,
                                                cudaStream_t stream)
{
    // Make sure we have enough memory reserved to render at the requested resolution
    size_t n_pixels = (size_t) resolution.x * resolution.y;

    // ensure we have memory for our ray data (the payload)...
    enlarge(n_pixels, padded_output_width, n_extra_dims, 1, stream);

    // build the rays...
    const dim3 threads = {16, 8, 1};
    const dim3 blocks  = {
        div_round_up((uint32_t) resolution.x, threads.x), div_round_up((uint32_t) resolution.y, threads.y), 1};
    init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(sample_index,
                                                                       m_rays[0].payload,
                                                                       resolution,
                                                                       focal_length,
                                                                       camera_matrix0,
                                                                       camera_matrix1,
                                                                       rolling_shutter,
                                                                       screen_center,
                                                                       parallax_shift,
                                                                       snap_to_pixel_centers,
                                                                       render_aabb,
                                                                       render_aabb_to_local,
                                                                       near_distance,
                                                                       plane_z,
                                                                       aperture_size,
                                                                       foveation,
                                                                       lens,
                                                                       envmap,
                                                                       frame_buffer,
                                                                       depth_buffer,
                                                                       hidden_area_mask,
                                                                       distortion,
                                                                       render_mode);

    m_n_rays_initialized = resolution.x * resolution.y;

    // CAVEAT: clear the original color+depth of the first (double-buffered) ray framebuffer.
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

    if (plane_z < 0)
    {
        // HACK: when in original slice rendermode, we negate the plane_z.
        // we are rendering in original 2d slice mode. so do not bother to trace forward in the volume.
        // we already terminated the rays.
    }
    else
    {
        // find a suitable first position in the volume...
        // when this function is over, the ray will either have terminated, or it will
        // at the first occupied voxel based on the density grid.
        linear_kernel(advance_pos_nerf_kernel,
                      0,
                      stream,
                      m_n_rays_initialized,
                      render_aabb,
                      render_aabb_to_local,
                      camera_matrix1[2],
                      focal_length,
                      sample_index,
                      m_rays[0].payload,
                      grid,
                      (show_accel >= 0) ? show_accel : 0,
                      max_mip,
                      cone_angle_constant);
    }
}

uint32_t Testbed::NerfTracer::trace(const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
                                    const BoundingBox& render_aabb,
                                    const mat3& render_aabb_to_local,
                                    const BoundingBox& train_aabb,
                                    const ivec2& resolution,
                                    int slice_count,
                                    const vec2& focal_length,
                                    float cone_angle_constant,
                                    const uint8_t* grid,
                                    ERenderMode render_mode,
                                    const mat4x3& camera_matrix,
                                    float depth_scale,
                                    int visualized_layer,
                                    int visualized_dim,
                                    ENerfActivation rgb_activation,
                                    ENerfActivation density_activation,
                                    int show_accel,
                                    uint32_t max_mip,
                                    float min_transmittance,
                                    float glow_y_cutoff,
                                    int glow_mode,
                                    const float* extra_dims_gpu,
                                    cudaStream_t stream)
{
    if (m_n_rays_initialized == 0)
    {
        return 0;
    }

    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

    uint32_t n_alive = m_n_rays_initialized;
    // m_n_rays_initialized = 0;

    uint32_t i                   = 1;
    uint32_t double_buffer_index = 0;
    while (i < MARCH_ITER)
    {
        RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
        RaysNerfSoa& rays_tmp     = m_rays[double_buffer_index % 2];
        ++double_buffer_index;

        // Compact rays that did not diverge yet.
        // also, any terminated rays are moved into the rays_hit buffer, and the hit_counter incremented.
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
            linear_kernel(compact_kernel_nerf,
                          0,
                          stream,
                          n_alive,
                          rays_tmp.rgba,
                          rays_tmp.depth,
                          rays_tmp.payload,
                          rays_current.rgba,
                          rays_current.depth,
                          rays_current.payload,
                          m_rays_hit.rgba,
                          m_rays_hit.depth,
                          m_rays_hit.payload,
                          m_alive_counter,
                          m_hit_counter);
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        }

        if (n_alive == 0)
        {
            // break if all rays have terminated.
            break;
        }

        // generate inference network inputs for all the alive rays, for a count of steps.

        // Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
        uint32_t target_n_queries           = 2 * 1024 * 1024;
        uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive,
                                                    (uint32_t) MIN_STEPS_INBETWEEN_COMPACTION,
                                                    (uint32_t) MAX_STEPS_INBETWEEN_COMPACTION);

        uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
        PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*) m_network_input, 1, 0, extra_stride);
        linear_kernel(generate_next_nerf_network_inputs,
                      0,
                      stream,
                      n_alive,
                      render_aabb,
                      render_aabb_to_local,
                      train_aabb,
                      focal_length,
                      camera_matrix[2],
                      rays_current.payload,
                      input_data,
                      n_steps_between_compaction,
                      grid,
                      (show_accel >= 0) ? show_accel : 0,
                      max_mip,
                      cone_angle_constant,
                      extra_dims_gpu);

        uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
        GPUMatrix<float> positions_matrix(
            (float*) m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
        GPUMatrix<network_precision_t, RM> rgbsigma_matrix(
            (network_precision_t*) m_network_output, network->padded_output_width(), n_elements);
        network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

        if (render_mode == ERenderMode::Normals)
        {
            network->input_gradient(stream, 3, positions_matrix, positions_matrix);
        }
        else if (render_mode == ERenderMode::EncodingVis)
        {
            network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
        }

        // composite all inference steps into the ray's rgba+depth...
        // TODO: composite into different slices for the ray.
        // this means the ray payload needs space for each rgba slice.
        linear_kernel(composite_kernel_nerf,
                      0,
                      stream,
                      n_alive,
                      n_elements,
                      i,
                      resolution,
                      slice_count,
                      train_aabb,
                      glow_y_cutoff,
                      glow_mode,
                      camera_matrix,
                      focal_length,
                      depth_scale,
                      rays_current.rgba,
                      rays_current.depth,
                      rays_current.payload,
                      input_data,
                      m_network_output,
                      network->padded_output_width(),
                      n_steps_between_compaction,
                      render_mode,
                      grid,
                      rgb_activation,
                      density_activation,
                      show_accel,
                      min_transmittance);

        i += n_steps_between_compaction;

        //break;
    }

    uint32_t n_hit;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return n_hit;
}

uint32_t Testbed::NerfTracer::debug_rays(uint32_t n_alive, const RaysNerfSoa& rays, cudaStream_t stream, float plane_z)
{
    CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
    linear_kernel(debug_rays_kernel,
                  0,
                  stream,
                  n_alive,
                  plane_z,
                  rays.rgba,
                  rays.depth,
                  rays.payload,
                  m_rays_hit.rgba,
                  m_rays_hit.depth,
                  m_rays_hit.payload,
                  m_alive_counter,
                  m_hit_counter);
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
#if 1
    // how many rays have finished. i.e. are terminated and have some opacity
    uint32_t n_hit = 0;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
#endif
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    return n_alive;
}

uint32_t Testbed::NerfTracer::traceSlices(const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
                                          const BoundingBox& render_aabb,
                                          const mat3& render_aabb_to_local,
                                          const BoundingBox& train_aabb,
                                          const ivec2& resolution,
                                          int slice_count,
                                          const vec2& focal_length,
                                          float cone_angle_constant,
                                          const uint8_t* grid,
                                          const mat4x3& camera_matrix,
                                          float depth_scale,
                                          ENerfActivation rgb_activation,
                                          ENerfActivation density_activation,
                                          float depthNear,
                                          uint32_t max_mip,
                                          float min_transmittance,
                                          const float* extra_dims_gpu,
                                          cudaStream_t stream)
{
    if (m_n_rays_initialized == 0)
    {
        return 0;
    }

    // CAVEAT: remember to zero the accumulated hit counter...
    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

    uint32_t n_hit   = 0;
    uint32_t n_alive = m_n_rays_initialized;
    // m_n_rays_initialized = 0;

    uint32_t i                   = 1;
    uint32_t double_buffer_index = 0;
    while (i < MARCH_ITER)
    {
        RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
        RaysNerfSoa& rays_tmp     = m_rays[double_buffer_index % 2];
        ++double_buffer_index;
#if 0
        if (debug_rays(n_alive, rays_tmp, stream) == 0)
            break;
#endif
        // Compact rays that did not diverge yet.
        // also, any terminated rays are moved into the rays_hit buffer, and the hit_counter incremented.
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
            linear_kernel(compact_kernel_nerf,
                          0,
                          stream,
                          n_alive,
                          rays_tmp.rgba,
                          rays_tmp.depth,
                          rays_tmp.payload,
                          rays_current.rgba,
                          rays_current.depth,
                          rays_current.payload,
                          m_rays_hit.rgba,
                          m_rays_hit.depth,
                          m_rays_hit.payload,
                          m_alive_counter,
                          m_hit_counter);
            CUDA_CHECK_THROW(
                cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        }

        if (n_alive == 0)
        {
            break;
        }

#if 1
        if (debug_rays(n_alive, rays_current, stream, depthNear) == 0)
            break;
#endif
#if 0
        // now advance the ray by n steps in one go...

        // generate inference network inputs for all the alive rays, for a count of steps.

        // Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
        uint32_t target_n_queries           = 2 * 1024 * 1024;
        uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive,
                                                    (uint32_t) MIN_STEPS_INBETWEEN_COMPACTION,
                                                    (uint32_t) MAX_STEPS_INBETWEEN_COMPACTION);

        uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
        PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*) m_network_input, 1, 0, extra_stride);
        linear_kernel(generate_next_nerf_network_inputs,
                      0,
                      stream,
                      n_alive,
                      render_aabb,
                      render_aabb_to_local,
                      train_aabb,
                      focal_length,
                      camera_matrix[2],
                      rays_current.payload,
                      input_data,
                      n_steps_between_compaction,
                      grid,
                      (show_accel >= 0) ? show_accel : 0,
                      max_mip,
                      cone_angle_constant,
                      extra_dims_gpu);

        uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
        GPUMatrix<float> positions_matrix(
            (float*) m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
        GPUMatrix<network_precision_t, RM> rgbsigma_matrix(
            (network_precision_t*) m_network_output, network->padded_output_width(), n_elements);
        network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

        if (render_mode == ERenderMode::Normals)
        {
            network->input_gradient(stream, 3, positions_matrix, positions_matrix);
        }
        else if (render_mode == ERenderMode::EncodingVis)
        {
            network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
        }

        // composite all inference steps into the ray's rgba+depth...
        // TODO: composite into different slices for the ray.
        // this means the ray payload needs space for each rgba slice.
        linear_kernel(composite_kernel_nerf_slice,
                      0,
                      stream,
                      n_alive,
                      n_elements,
                      i,
                      resolution,
                      slice_count,
                      train_aabb,
                      camera_matrix,
                      focal_length,
                      depth_scale,
                      rays_current.rgba,
                      rays_current.depth,
                      rays_current.payload,
                      input_data,
                      m_network_output,
                      network->padded_output_width(),
                      n_steps_between_compaction,
                      grid,
                      rgb_activation,
                      density_activation,
                      min_transmittance);

        i += n_steps_between_compaction;

#if 0
        if (debug_rays(n_alive, rays_current, stream) == 0)
            break;
#endif
#endif
    }

    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return n_hit;
}

void Testbed::NerfTracer::enlarge(size_t n_elements,
                                  uint32_t padded_output_width,
                                  uint32_t n_extra_dims,
                                  int slice_count,
                                  cudaStream_t stream)
{
    n_elements        = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
    size_t num_floats = sizeof(NerfCoordinate) / sizeof(float) + n_extra_dims;
    auto scratch =
        allocate_workspace_and_distribute<RaysNerfSoa::ColorType,
                                          RaysNerfSoa::DepthType,
                                          NerfPayload, // m_rays[0]

                                          RaysNerfSoa::ColorType,
                                          RaysNerfSoa::DepthType,
                                          NerfPayload, // m_rays[1]

                                          RaysNerfSoa::ColorType,
                                          RaysNerfSoa::DepthType,
                                          NerfPayload, // m_rays_hit

                                          network_precision_t,
                                          float,
                                          uint32_t,
                                          uint32_t>(stream,
                                                    &m_scratch_alloc,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements,
                                                    n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
                                                    n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
                                                    32, // 2 full cache lines to ensure no overlap
                                                    32  // 2 full cache lines to ensure no overlap
        );

    m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements, slice_count);
    m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements, slice_count);
    m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements, slice_count);

    m_network_output = std::get<9>(scratch);
    m_network_input  = std::get<10>(scratch);

    m_hit_counter   = std::get<11>(scratch);
    m_alive_counter = std::get<12>(scratch);
}

std::vector<float> Testbed::Nerf::Training::get_extra_dims_cpu(int trainview) const
{
    if (dataset.n_extra_dims() == 0)
    {
        return {};
    }

    if (trainview < 0 || trainview >= dataset.n_images)
    {
        throw std::runtime_error{"Invalid training view."};
    }

    const float* extra_dims_src = extra_dims_gpu.data() + trainview * dataset.n_extra_dims();

    std::vector<float> extra_dims_cpu(dataset.n_extra_dims());
    CUDA_CHECK_THROW(cudaMemcpy(
        extra_dims_cpu.data(), extra_dims_src, dataset.n_extra_dims() * sizeof(float), cudaMemcpyDeviceToHost));

    return extra_dims_cpu;
}

void Testbed::Nerf::Training::update_extra_dims()
{
    uint32_t n_extra_dims = dataset.n_extra_dims();
    std::vector<float> extra_dims_cpu(extra_dims_gpu.size());
    for (uint32_t i = 0; i < extra_dims_opt.size(); ++i)
    {
        const std::vector<float>& value = extra_dims_opt[i].variable();
        for (uint32_t j = 0; j < n_extra_dims; ++j)
        {
            extra_dims_cpu[i * n_extra_dims + j] = value[j];
        }
    }

    CUDA_CHECK_THROW(cudaMemcpyAsync(extra_dims_gpu.data(),
                                     extra_dims_cpu.data(),
                                     extra_dims_opt.size() * n_extra_dims * sizeof(float),
                                     cudaMemcpyHostToDevice));
}

vec3 interpolate2(float xb, float yb, int ni, int nj, int nx, int ny)
{
    float p = (float) ni / (nx - 1);
    float q = (float) nj / (ny - 1);

    vec3 uv(0, 0, 1);
    uv.x = (p) * (xb * 2) - (xb);
    uv.y = (q) * (yb * 2) - (yb);
    return uv;
}

std::vector<vec3> display_space_ro(float xb, float yb, int nx, int ny)
{
    std::vector<vec3> result;
    result.resize(nx * ny);

    //torch.linspace(-xb/2, xb/2, nx, dtype=torch.float32, device=device),
    //torch.linspace(yb/2, -yb/2, ny, dtype=torch.float32, device=device),

    for (int x = 0; x < nx; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            result[x + y * nx] = interpolate2(xb, -yb, x, y, nx, ny);
        }
    }

    return result;
}

vec3 get_display_space_rd(float theta_tot_x, float theta_tot_y, int ni, int nj, int nx, int ny)
{
    float half_x = tanf(vxl::radians(theta_tot_x / 2));
    float half_y = tanf(vxl::radians(theta_tot_y / 2));

    vec3 uv(0, 0, 1);
    uv.x = ((float) ni / (nx - 1)) * (half_x * 2) - (half_x);
    uv.y = -(((float) nj / (ny - 1)) * (half_y * 2) - (half_x));
    return uv;
}

std::vector<vec3> display_space_rd(float theta_tot_x, float theta_tot_y, int nx, int ny)
{
    std::vector<vec3> result;
    result.resize(nx * ny);

    float half_x = tanf(vxl::radians(theta_tot_x / 2));
    float half_y = tanf(vxl::radians(theta_tot_y / 2));

    // torch.linspace(-half_x, half_x, vx, dtype = torch.float32, device = device),
    //torch.linspace(-half_y, half_y, vy, dtype = torch.float32, device = device),

    vec3 uv(0, 0, 1);

    for (int x = 0; x < nx; ++x)
    {
        uv.x = ((float) x / (nx - 1)) * (half_x * 2) - (half_x);
        for (int y = 0; y < ny; ++y)
        {
            uv.y = ((float) y / (ny - 1)) * (half_y * 2) - (half_y);

            result[x + y * nx] = uv;
        }
    }

    return result;
}

void Testbed::render_nerf(cudaStream_t stream,
                          CudaDevice& device,
                          const CudaRenderBufferView& render_buffer,
                          const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
                          const uint8_t* density_grid_bitfield,
                          const vec2& focal_length,
                          const mat4x3& camera_matrix0,
                          const mat4x3& camera_matrix1,
                          const vec4& rolling_shutter,
                          const vec2& screen_center,
                          const Foveation& foveation,
                          int visualized_dimension)
{
    float plane_z       = m_slice_plane_z + m_scale;
    float aperture_size = m_aperture_size;
#if 1

    // HACK: flip this to negative to signify it is a slice plane!
    if (m_render_mode == ERenderMode::Slice)
    {
        //plane_z = -plane_z; // force it into 2d mode where we don't advance the ray or have aperture > 0!

        aperture_size = 0.f;
    }
#endif

    ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

    const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(stream);

    NerfTracer tracer;

    // create a 2D buffer to store the 2D distortion vectors.
    //
    // Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
    // (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
    auto grid_distortion = m_nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion)
                               ? m_distortion.inference_view()
                               : Buffer2DView<const vec2>{};

    Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

    auto resolution = render_buffer.resolution;
    int slice_count = 32; //std::min(4, RaysNerfSoa::ColorType::kSize);

#if 0
    if (false && render_mode == ERenderMode::Slice)
    {
        int nx = 16;
        int ny = 16;
        int nz = 4;

        int vx = 8;
        int vy = 8;

        float xb = 0.5;
        float yb = 0.5;
        float zb = 0.5;

        float theta_tot_x = 20.f;
        float theta_tot_y = 20.f;

        float half_x = tanf(vxl::radians(theta_tot_x / 2));
        float half_y = tanf(vxl::radians(theta_tot_y / 2));

        vec3 bboxPos(0.f);
        mat4 bboxRotMat;

        // sweeping plane
        float forward_step_size = zb / nz;

        float alpha_thres = 0.0001f;

        std::vector<float> sweeping_alpha(nx * ny);
        std::vector<vec3> sweeping_rgb(nx * ny);
        std::vector<float> sweeping_T(nx * ny);
        std::vector<int> sampleIndices(nx * ny);
        std::vector<int> newSampleIndices(nx * ny);

        {
            int vi       = 0;
            int vj       = 0;
            auto base_rd = interpolate2(half_x, half_y, vi, vj, vx, vy);

            // define quilt re - sampling pattern on the sweeping plane...

            const float x_fullangle = 2 * vxl::tan(vxl::radians(theta_tot_x / 2)) * forward_step_size;
            const float y_fullangle = 2 * vxl::tan(vxl::radians(theta_tot_y / 2)) * forward_step_size;
            const float x_per_pix   = xb / (nx - 1);
            const float y_per_pix   = yb / (ny - 1);
            const float x_per_angle = x_fullangle / (vx - 1);
            const float dx          = -x_per_angle / x_per_pix;
            const float y_per_angle = y_fullangle / (vy - 1);
            const float dy          = y_per_angle / y_per_pix;

            float now_dx = -dx * (nz / 2);
            float now_dy = -dy * (nz / 2);

            for (int slice_i = 0; slice_i < nz; ++slice_i)
            {
                // Move sweeping plane forward
                if (slice_i == 0)
                    continue;

                const vec3 sweeping_rd_c = vec3(0, 0, -1);
                const vec3 sweeping_rd   = /*bboxRotMat * */ sweeping_rd_c;
                const vec3 sweeping_step = sweeping_rd * forward_step_size;
                const float offsetz      = -sweeping_rd_c.z * (forward_step_size * nz / 2);

                vec3 slice_pos = vec3(0, 0, offsetz) + bboxPos;

                sampleIndices.resize(nx * ny);
                for (int i = 0; i < nx * ny; ++i)
                    sampleIndices[i] = i;

                // for each pixel in the sweep plane...
                for (int nj = 0; nj < ny; ++nj)
                {
                    for (int ni = 0; ni < nx; ++ni)
                    {
                        int pixel_i = ni + nj * nx;

                        // get the ray origin for this pixel...
                        auto base_ro     = interpolate2(xb / 2, -yb / 2, ni, nj, nx, ny);
                        base_ro.z        = 0;
                        vec3 sweeping_ro = /*bboxRotMat **/ base_ro + slice_pos;

                        OPENVXL_LOG_VAR2(ni, sweeping_ro.x);
                        // transmission buffer.
                        //float sweeping_T    = torch.ones([self.ny, self.nx, 1], device = device)

                        //quilt_T = torch.ones([self.vy, self.vx, self.ny, self.nx], device = device)
                        // //quilt_color =torch.zeros([self.vy, self.vx, self.ny, self.nx, 3], device = device)

                        // filter to keep points...
                        {
                            newSampleIndices.clear();
                            for (int i = 0; i < sampleIndices.size(); ++i)
                            {
                                if (sweeping_T[pixel_i] > alpha_thres)
                                {
                                    newSampleIndices.push_back(ni + nj * nx);
                                }
                            }

                            // filter by density at all points...
                            for (int i = 0; i < newSampleIndices.size(); ++i)
                            {
                                /*
                                for (density_fn : pipeline.model.density_fns)
                                {
                                    if (len(pt) == 0)
                                        break;
                                    density = density_fn(pt);
                                    alphas  = 1 - torch.exp(-forward_step_size * density);

                                    sel     = torch.where(alphas.squeeze() > self.alpha_thres)[0];
                                    pt      = pt[sel];
                                    pt_ii   = pt_ii[sel];
                                    pt_jj   = pt_jj[sel];
                                }
								*/
                            }

                            sampleIndices = newSampleIndices;
                        }

                        if (sampleIndices.size() == 0)
                            continue;

                        // Sample the remaining points...
                        vec3 rgb;
                        float alpha = 0;

                        /*
                        auto ray_samples = RaySamples(frustums = Frustums(
                                origins = pt, directions = sweeping_rd, starts = 0, ends = 0, pixel_area = 1, ),
                            camera_indices = torch.zeros([1], dtype = torch.int64), );

						float density, embedding = pipeline.model.field.get_density(ray_samples);
						alpha = 1 - exp(-forward_step_size * density);

						if ((alpha <= self.alpha_thres).all())
                            continue;
						else if(self.sweep_T_filtering)
							sweeping_T[pixel_i] *= (1 - alpha);
						*/

                        // sample color into all the sample points.
                        //rgb = pipeline.model.field.get_outputs(ray_samples, density_embedding = embedding)[FieldHeadNames.RGB];

                        sweeping_alpha[pixel_i] = alpha; //.squeeze();
                        sweeping_rgb[pixel_i]   = alpha * rgb;
                    }
                }

                // no we have the slice... we can render it for all views.
                /*
				swizzle_alpha_comp_cuda.forward_warping(plane_alpha = sweeping_alpha,
                                                        plane_color = sweeping_rgb,
                                                        quilt_T     = quilt_T,
                                                        quilt_color = quilt_color,
                                                        dx          = now_dx,
                                                        dy          = now_dy,
                                                        alpha_thres = self.alpha_thres)*/

                now_dx += dx;
                now_dy += dy;
                slice_pos.z -= forward_step_size;
            }
        }

        return;
    }
#endif

    // create the camera rays...
    tracer.init_rays_from_camera(
        render_buffer.spp,
        nerf_network->padded_output_width(),
        nerf_network->n_extra_dims(),
        render_buffer.resolution,
        slice_count,
        focal_length,
        camera_matrix0,
        camera_matrix1,
        rolling_shutter,
        screen_center,
        m_parallax_shift,
        m_snap_to_pixel_centers,
        m_render_aabb,
        m_render_aabb_to_local,
        m_render_near_distance,
        plane_z,
        aperture_size,
        foveation,
        lens,
        m_envmap.inference_view(),
        grid_distortion,
        render_buffer.frame_buffer,
        render_buffer.depth_buffer,
        render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
        density_grid_bitfield,
        m_nerf.show_accel,
        m_nerf.max_cascade,
        m_nerf.cone_angle_constant,
        render_mode,
        stream);

    float depth_scale = 1.0f / m_nerf.training.dataset.scale;

    // render_2d is true if we are rendering volume, but instead making a diagnostic "flat" image.
    bool render_2d = /*m_render_mode == ERenderMode::Slice || */ m_render_mode == ERenderMode::Distortion;

    uint32_t n_hit;
    if (render_2d)
    {
        n_hit = tracer.n_rays_initialized();
    }
    else
    {
        n_hit = tracer.trace(nerf_network,
                             m_render_aabb,
                             m_render_aabb_to_local,
                             m_aabb,
                             resolution,
                             slice_count,
                             focal_length,
                             m_nerf.cone_angle_constant,
                             density_grid_bitfield,
                             render_mode,
                             camera_matrix1,
                             depth_scale,
                             m_visualized_layer,
                             visualized_dimension,
                             m_nerf.rgb_activation,
                             m_nerf.density_activation,
                             m_nerf.show_accel,
                             m_nerf.max_cascade,
                             m_nerf.render_min_transmittance,
                             m_nerf.glow_y_cutoff,
                             m_nerf.glow_mode,
                             extra_dims_gpu,
                             stream);
    }

    RaysNerfSoa& rays_hit = render_2d ? tracer.rays_init() : tracer.rays_hit();

    if (render_2d)
    {
        // Store colors in the normal buffer
        uint32_t n_elements             = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);
        const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + nerf_network->n_extra_dims();
        const uint32_t extra_stride =
            nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

        GPUMatrix<float> positions_matrix{floats_per_coord, n_elements, stream};
        GPUMatrix<float> rgbsigma_matrix{4, n_elements, stream};

        linear_kernel(generate_nerf_network_inputs_at_current_position,
                      0,
                      stream,
                      n_hit,
                      m_aabb,
                      rays_hit.payload,
                      PitchedPtr<NerfCoordinate>((NerfCoordinate*) positions_matrix.data(), 1, 0, extra_stride),
                      extra_dims_gpu);

        if (visualized_dimension == -1)
        {
            nerf_network->inference(stream, positions_matrix, rgbsigma_matrix);
            linear_kernel(compute_nerf_rgba_kernel,
                          0,
                          stream,
                          n_hit,
                          (vec4*) rgbsigma_matrix.data(),
                          m_nerf.rgb_activation,
                          m_nerf.density_activation,
                          0.01f,
                          false);
        }
        else
        {
            nerf_network->visualize_activation(
                stream, m_visualized_layer, visualized_dimension, positions_matrix, rgbsigma_matrix);
        }

        linear_kernel(shade_kernel_nerf_2d,
                      0,
                      stream,
                      n_hit,
                      m_nerf.render_gbuffer_hard_edges,
                      camera_matrix1,
                      depth_scale,
                      (vec4*) rgbsigma_matrix.data(),
                      nullptr,
                      rays_hit.payload,
                      m_render_mode,
                      m_nerf.training.linear_colors,
                      render_buffer.frame_buffer,
                      render_buffer.depth_buffer);

        return;
    }

#if 1
    // DEBUGGING:
    render_buffer.clear(stream);
#endif

    if (m_render_mode == ERenderMode::Slice && 0)
    {
        int mosaic_size = vxl::sqrt(vxl::pow2((int) vxl::ceil(vxl::sqrt((float) slice_count))));

        linear_kernel(shade_kernel_nerf_draw_slices,
                      0,
                      stream,
                      n_hit * slice_count,
                      resolution,
                      slice_count,
                      mosaic_size,
                      m_nerf.render_gbuffer_hard_edges,
                      camera_matrix1,
                      depth_scale,
                      rays_hit.rgba,
                      rays_hit.depth,
                      rays_hit.payload,
                      m_render_mode,
                      m_nerf.training.linear_colors,
                      render_buffer.frame_buffer,
                      render_buffer.depth_buffer);
    }
    else
    {
        linear_kernel(shade_kernel_nerf,
                      0,
                      stream,
                      n_hit,
                      m_nerf.render_gbuffer_hard_edges,
                      camera_matrix1,
                      depth_scale,
                      rays_hit.rgba,
                      rays_hit.depth,
                      rays_hit.payload,
                      m_render_mode,
                      m_nerf.training.linear_colors,
                      render_buffer.frame_buffer,
                      render_buffer.depth_buffer);
    }

    if (render_mode == ERenderMode::Cost)
    {
        std::vector<NerfPayload> payloads_final_cpu(n_hit);
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        size_t total_n_steps = 0;
        for (uint32_t i = 0; i < n_hit; ++i)
        {
            total_n_steps += payloads_final_cpu[i].n_steps;
        }
        tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = "
                     << ((float) total_n_steps / (float) n_hit);
    }
}

//------------------------------------------------------------------------------
void Testbed::render_nerf_slices(cudaStream_t stream,
                                 CudaDevice& device,
                                 const CudaRenderBufferView& render_buffer,
                                 const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
                                 const uint8_t* density_grid_bitfield,
                                 const vec2& focal_length,
                                 const mat4x3& camera_matrix0,
                                 const mat4x3& camera_matrix1,
                                 const vec4& rolling_shutter,
                                 const vec2& screen_center,
                                 const Foveation& foveation,
                                 int visualized_dimension)
{
    float aperture_size = m_aperture_size;

    // HACK: flip this to negative to signify it is a slice plane!
    if (m_render_mode == ERenderMode::Slice)
    {
        //plane_z = -plane_z; // force it into 2d mode where we don't advance the ray or have aperture > 0!

        aperture_size = 0.f;
    }

    const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(stream);

    NerfTracer tracer;

    // create a 2D buffer to store the 2D distortion vectors.
    //
    // Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
    // (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
    auto grid_distortion = m_nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion)
                               ? m_distortion.inference_view()
                               : Buffer2DView<const vec2>{};

    Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

    auto resolution = render_buffer.resolution;
    int slice_count = m_slice_count;

    ivec2 sliceResolution = render_buffer.resolution;
    if (sliceResolution.x == 0)
        return; // not yet ready!

    int vx                     = m_view_tiles.x;
    int vy                     = m_view_tiles.y;
    int nx                     = sliceResolution.x;
    int ny                     = sliceResolution.y;
    float quiltViewConeFovDegX = m_quiltViewConeFovDegX;
    float quiltViewConeFovDegY = quiltViewConeFovDegX;
    float refForwardDist       = 0.f;
    float alphaThreshold       = 0.001f;
    float focalDistance        = vxl::max(0.01f, m_quiltFocusDistance);
    float fovDegX              = focal_length_to_fov(ivec2{nx, ny}, focal_length).x;

    int imageViewWidth       = nx;
    int imageViewHeight      = ny;
    float imageViewAspect    = (float) imageViewWidth / imageViewHeight;
    float imageViewAspectInv = 1.f / imageViewAspect;

    float referenceFovX           = vxl::radians(fovDegX);
    float referenceCamTanHalfFovX = vxl::tan(vxl::radians(fovDegX) * 0.5f);
    float referenceCamTanHalfFovY = referenceCamTanHalfFovX * imageViewAspectInv;

    // store this original view
    auto baseCamTanHalfFovX = referenceCamTanHalfFovX;
    auto baseCamTanHalfFovY = baseCamTanHalfFovX * imageViewAspectInv;

    // Get width & height of the focal plane...
    float focalPlaneSizeX = baseCamTanHalfFovX * focalDistance * 2;
    float focalPlaneSizeY = baseCamTanHalfFovY * focalDistance * 2;

    // Compute viewcone camera position
    {
        // determine the reference fov for enlarging the view by the
        // offset-camera's max displacement.
        float maxOffsetX = focalDistance * vxl::tan(vxl::radians(quiltViewConeFovDegX) * 0.5f);
        float maxOffsetY = maxOffsetX * imageViewAspectInv;

        // Also calc the distance we can push forward to keep objects the same size...
        float forwardX = focalDistance * maxOffsetX / (maxOffsetX + 0.5f * focalPlaneSizeX);
        float forwardY = forwardX * imageViewAspectInv;
        forwardX       = vxl::clamp(forwardX, 0.f, focalDistance);
        forwardY       = vxl::clamp(forwardY, 0.f, focalDistance);

        // calc the reference camera fov...
        float renderCamHalfFovX = vxl::atan2(maxOffsetX, forwardX);

        // overwrite the reference camera...
        referenceFovX           = 2.0f * renderCamHalfFovX;
        referenceCamTanHalfFovX = vxl::tan(renderCamHalfFovX);
        referenceCamTanHalfFovY = referenceCamTanHalfFovX * imageViewAspectInv;
    }

    /*
			auto worldToViewMat = view.getLeftViewMatrix();
            auto modelToViewMat = worldToViewMat * modelToWorldMat;
			
			// get the view bounds for the slice distances calculation...
            // NOTE: we do this before shifting the view in Z.
            auto viewBounds = vxl::AffineTransform3f(modelToViewMat).apply(mesh.bounds_);
            float depthNear = -viewBounds.max()[2];
            float depthFar  = -viewBounds.min()[2];
            depthNear       = vxl::max(view.nearDistance, depthNear);
            depthFar        = vxl::min(view.farDistance, depthFar);
            depthFar        = vxl::max(depthFar, depthNear);
            //NVPV_LOGV2(depthNear, depthFar);
			*/
    float depthNear = m_slice_plane_z;
    float depthFar  = depthNear + 10.f;
    /*
            // Get width & height of the focal plane...
            float focalPlaneSizeX = camTanHalfFovX * focusDistance * 2;
            float focalPlaneSizeY = camTanHalfFovY * focusDistance * 2;
            float maxOffsetX      = focusDistance * vxl::tan(vxl::radians(viewConeFovDegX) * 0.5f);
            float forwardX        = focusDistance * maxOffsetX / (maxOffsetX + 0.5f * focalPlaneSizeX);
            forwardX              = vxl::clamp(forwardX, 0.f, focusDistance);
            float camHalfFovX     = vxl::atan2(maxOffsetX, forwardX);
            float refTanFovX      = vxl::tan(camHalfFovX);
            float viewConeFovDegY = viewConeFovDegX * viewAspectInv;
            float maxOffsetY      = maxOffsetX * viewAspectInv;
            float forwardY        = forwardX * viewAspectInv;
            forwardY              = vxl::clamp(forwardY, 0.f, focusDistance);
            float refTanFovY      = refTanFovX * viewAspectInv;
            float halfViewFovX    = vxl::radians(0.5f * fovDegX);
            float halfViewFovY    = halfViewFovX * viewAspectInv;
            float refForwardDist  = 0.f;
			*/

    int mosaicSize = vxl::sqrt(vxl::pow2((int) vxl::ceil(vxl::sqrt((float) slice_count))));
    //float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

    vec2 ref_focal_length =
        fov_to_focal_length(ivec2{nx, ny}, {vxl::degrees(referenceFovX), vxl::degrees(referenceFovX)});

    // create the camera rays...
    tracer.init_rays_from_camera(
        render_buffer.spp,
        nerf_network->padded_output_width(),
        nerf_network->n_extra_dims(),
        sliceResolution,
        slice_count,
        ref_focal_length,
        camera_matrix0,
        camera_matrix1,
        rolling_shutter,
        screen_center,
        m_parallax_shift,
        m_snap_to_pixel_centers,
        m_render_aabb,
        m_render_aabb_to_local,
        m_render_near_distance,
        depthNear,
        aperture_size,
        foveation,
        lens,
        m_envmap.inference_view(),
        grid_distortion,
        render_buffer.frame_buffer,
        render_buffer.depth_buffer,
        render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
        density_grid_bitfield,
        m_nerf.show_accel,
        m_nerf.max_cascade,
        m_nerf.cone_angle_constant,
        ERenderMode::Slice,
        stream);

    float depth_scale = 1.0f / m_nerf.training.dataset.scale;

    uint32_t n_hit = tracer.traceSlices(nerf_network,
                                        m_render_aabb,
                                        m_render_aabb_to_local,
                                        m_aabb,
                                        sliceResolution,
                                        slice_count,
                                        ref_focal_length,
                                        m_nerf.cone_angle_constant,
                                        density_grid_bitfield,
                                        camera_matrix1,
                                        depth_scale,
                                        m_nerf.rgb_activation,
                                        m_nerf.density_activation,
                                        depthNear,
                                        m_nerf.max_cascade,
                                        m_nerf.render_min_transmittance,
                                        extra_dims_gpu,
                                        stream);

    RaysNerfSoa& rays_hit = tracer.rays_hit();

    if (n_hit == 0)
        return;
#if 1
    uint32_t n_elements             = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);
    const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + nerf_network->n_extra_dims();
    const uint32_t extra_stride =
        nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    GPUMatrix<float> positions_matrix{floats_per_coord, n_elements, stream};
    GPUMatrix<float> rgbsigma_matrix{4, n_elements, stream};

    uint32_t n_steps_between_compaction = 1;
    PitchedPtr<NerfCoordinate> input_data;

#else
    // Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
    uint32_t target_n_queries           = 2 * 1024 * 1024;
    uint32_t n_steps_between_compaction = 4; /*clamp(target_n_queries / n_hit,
                                                        (uint32_t) MIN_STEPS_INBETWEEN_COMPACTION,
                                                        (uint32_t) MAX_STEPS_INBETWEEN_COMPACTION);*/

    uint32_t extra_stride = nerf_network->n_extra_dims() * sizeof(float);
    PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*) tracer.m_network_input, 1, 0, extra_stride);

    uint32_t n_elements = next_multiple(n_hit * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
    GPUMatrix<float> positions_matrix(
        (float*) tracer.m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
    GPUMatrix<network_precision_t, RM> rgbsigma_matrix(
        (network_precision_t*) tracer.m_network_output, nerf_network->padded_output_width(), n_elements);

#endif

    // make slice buffer large enough for a batch of slices...
    static auto sliceColorBuf = std::make_shared<Buffer2D<vec4>>();
    sliceColorBuf->resize(ivec2{sliceResolution.x, sliceResolution.y * (int) 1});
    vec4* sliceColorPtr = sliceColorBuf->data();

    // CAVEAT: do we need to clear?
    render_buffer.clear(stream);

    // track the depth in world units...
    float sliceBatchDistance = depthNear;

    for (int slice_i = 0; slice_i < slice_count; slice_i += 1)
    {
#if 1
        linear_kernel(generate_nerf_network_inputs_at_current_position,
                      0,
                      stream,
                      n_hit,
                      m_aabb,
                      rays_hit.payload,
                      PitchedPtr<NerfCoordinate>((NerfCoordinate*) positions_matrix.data(), 1, 0, extra_stride),
                      extra_dims_gpu);

        nerf_network->inference(stream, positions_matrix, rgbsigma_matrix);

        linear_kernel(compute_nerf_rgba_kernel,
                      0,
                      stream,
                      n_hit,
                      (vec4*) rgbsigma_matrix.data(),
                      m_nerf.rgb_activation,
                      m_nerf.density_activation,
                      from_stepping_space(m_delta_z, 0.0f),
                      false);
#else
        linear_kernel(generate_next_nerf_network_inputs_slices,
                      0,
                      stream,
                      n_hit,
                      m_render_aabb,
                      m_render_aabb_to_local,
                      m_aabb,
                      render_buffer.spp,
                      focal_length,
                      camera_matrix1[2],
                      rays_hit.payload,
                      input_data,
                      n_steps_between_compaction,
                      density_grid_bitfield,
                      (m_nerf.show_accel >= 0) ? m_nerf.show_accel : 0,
                      m_nerf.max_cascade,
                      m_nerf.cone_angle_constant,
                      extra_dims_gpu);

        nerf_network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);
#endif

#if 1
        // draw all the slices in a mosaic!

        linear_kernel(shade_kernel_nerf_slice,
                      0,
                      stream,
                      n_hit,
                      n_elements,
                      resolution,
                      slice_i,
                      mosaicSize,
                      m_nerf.render_gbuffer_hard_edges,
                      camera_matrix1,
                      depth_scale,
                      (vec4*) rgbsigma_matrix.data(),
                      nullptr,
                      rays_hit.payload,
                      m_render_mode,

					  m_aabb,
                      input_data,
                      tracer.m_network_output,
                      nerf_network->padded_output_width(),
                      m_nerf.rgb_activation,
                      m_nerf.density_activation,
                      n_steps_between_compaction,

                      m_nerf.training.linear_colors,
                      render_buffer.frame_buffer,
                      render_buffer.depth_buffer);
#else
        // composite into slice into accumulated buffer!
        // for a test, just use the same framebuffer!
        // we can treat the framebuffer's alpha as 1-transmission.

        linear_kernel(shade_kernel_nerf_accumulate_slice,
                      0,
                      stream,
                      n_hit,
                      n_elements,
                      sliceResolution,
                      m_aabb,
                      slice_i,
                      depth_scale,
                      rays_hit.payload,
                      (vec4*) rgbsigma_matrix.data(),
                      nullptr,

                      input_data,
                      tracer.m_network_output,
                      nerf_network->padded_output_width(),
                      m_nerf.rgb_activation,
                      m_nerf.density_activation,
                      n_steps_between_compaction,

                      m_nerf.training.linear_colors,
                      sliceColorPtr,
                      render_buffer.depth_buffer);

        linear_kernel(kernel_slice_blend,
                      0,
                      stream,
                      resolution.x * resolution.y,
                      resolution,
                      vx,
                      vy,
                      render_buffer.frame_buffer,
                      render_buffer.depth_buffer,
                      alphaThreshold,
                      focalDistance,
                      0.5f * vxl::radians(quiltViewConeFovDegX),
                      0.5f * vxl::radians(quiltViewConeFovDegY),
                      referenceCamTanHalfFovX,
                      referenceCamTanHalfFovY,
                      baseCamTanHalfFovX,
                      baseCamTanHalfFovY,
                      refForwardDist,
                      sliceBatchDistance,
                      sliceResolution,
                      1, //n_steps_between_compaction,
                      slice_count,
                      sliceColorPtr);

#endif

        // now step forward the slice distance.
        // we use some randomness for the distance to get a smooth blend between slices...
        linear_kernel(step_pos_nerf_kernel,
                      0,
                      stream,
                      n_hit,
                      m_render_aabb,
                      m_render_aabb_to_local,
                      camera_matrix1[2],
                      focal_length,
                      render_buffer.spp,
                      rays_hit.payload,
                      density_grid_bitfield,
                      (m_nerf.show_accel >= 0) ? m_nerf.show_accel : 0,
                      m_nerf.max_cascade,
                      m_nerf.cone_angle_constant,
                      m_delta_z * 1);

        sliceBatchDistance += from_stepping_space(1 * m_delta_z, 0.f);
    }

    return;
}

//------------------------------------------------------------------------------
void Testbed::Nerf::Training::set_camera_intrinsics(int frame_idx,
                                                    float fx,
                                                    float fy,
                                                    float cx,
                                                    float cy,
                                                    float k1,
                                                    float k2,
                                                    float p1,
                                                    float p2,
                                                    float k3,
                                                    float k4,
                                                    bool is_fisheye)
{
    if (frame_idx < 0 || frame_idx >= dataset.n_images)
    {
        return;
    }
    if (fx <= 0.f)
        fx = fy;
    if (fy <= 0.f)
        fy = fx;
    auto& m = dataset.metadata[frame_idx];
    if (cx < 0.f)
        cx = -cx;
    else
        cx = cx / m.resolution.x;
    if (cy < 0.f)
        cy = -cy;
    else
        cy = cy / m.resolution.y;
    m.lens = {ELensMode::Perspective};
    if (k1 || k2 || k3 || k4 || p1 || p2)
    {
        if (is_fisheye)
        {
            m.lens = {ELensMode::OpenCVFisheye, k1, k2, k3, k4};
        }
        else
        {
            m.lens = {ELensMode::OpenCV, k1, k2, p1, p2};
        }
    }

    m.principal_point = {cx, cy};
    m.focal_length    = {fx, fy};
    dataset.update_metadata(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::set_camera_extrinsics_rolling_shutter(int frame_idx,
                                                                    mat4x3 camera_to_world_start,
                                                                    mat4x3 camera_to_world_end,
                                                                    const vec4& rolling_shutter,
                                                                    bool convert_to_ngp)
{
    if (frame_idx < 0 || frame_idx >= dataset.n_images)
    {
        return;
    }

    if (convert_to_ngp)
    {
        camera_to_world_start = dataset.nerf_matrix_to_ngp(camera_to_world_start);
        camera_to_world_end   = dataset.nerf_matrix_to_ngp(camera_to_world_end);
    }

    dataset.xforms[frame_idx].start             = camera_to_world_start;
    dataset.xforms[frame_idx].end               = camera_to_world_end;
    dataset.metadata[frame_idx].rolling_shutter = rolling_shutter;
    dataset.update_metadata(frame_idx, frame_idx + 1);

    cam_rot_offset[frame_idx].reset_state();
    cam_pos_offset[frame_idx].reset_state();
    cam_exposure[frame_idx].reset_state();
    update_transforms(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::set_camera_extrinsics(int frame_idx, mat4x3 camera_to_world, bool convert_to_ngp)
{
    set_camera_extrinsics_rolling_shutter(frame_idx, camera_to_world, camera_to_world, vec4(0.0f), convert_to_ngp);
}

void Testbed::Nerf::Training::reset_camera_extrinsics()
{
    for (auto&& opt : cam_rot_offset)
    {
        opt.reset_state();
    }

    for (auto&& opt : cam_pos_offset)
    {
        opt.reset_state();
    }

    for (auto&& opt : cam_exposure)
    {
        opt.reset_state();
    }
}

void Testbed::Nerf::Training::export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format)
{
    tlog::info() << "Saving a total of " << n_images_for_training << " poses to " << path.str();
    nlohmann::json trajectory;
    for (int i = 0; i < n_images_for_training; ++i)
    {
        nlohmann::json frame{{"id", i}};

        const mat4x3 p_nerf = get_camera_extrinsics(i);
        if (export_extrinsics_in_quat_format)
        {
            // Assume 30 fps
            frame["time"] = i * 0.033f;
            // Convert the pose from NeRF to Quaternion format.
            const mat3 conv_coords_l{
                0.f,
                0.f,
                -1.f,
                1.f,
                0.f,
                0.f,
                0.f,
                -1.f,
                0.f,
            };
            const mat4 conv_coords_r{
                1.f,
                0.f,
                0.f,
                0.f,
                0.f,
                -1.f,
                0.f,
                0.f,
                0.f,
                0.f,
                -1.f,
                0.f,
                0.f,
                0.f,
                0.f,
                1.f,
            };
            const mat4x3 p_quat = conv_coords_l * p_nerf * conv_coords_r;

            const quat rot_q = mat3(p_quat);
            frame["q"]       = rot_q;
            frame["t"]       = p_quat[3];
        }
        else
        {
            frame["transform_matrix"] = p_nerf;
        }

        trajectory.emplace_back(frame);
    }

    std::ofstream file{native_string(path)};
    file << std::setw(2) << trajectory << std::endl;
}

mat4x3 Testbed::Nerf::Training::get_camera_extrinsics(int frame_idx)
{
    if (frame_idx < 0 || frame_idx >= dataset.n_images)
    {
        return mat4x3::identity();
    }
    return dataset.ngp_matrix_to_nerf(transforms[frame_idx].start);
}

void Testbed::Nerf::Training::update_transforms(int first, int last)
{
    if (last < 0)
    {
        last = dataset.n_images;
    }

    if (last > dataset.n_images)
    {
        last = dataset.n_images;
    }

    int n = last - first;
    if (n <= 0)
    {
        return;
    }

    if (transforms.size() < last)
    {
        transforms.resize(last);
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        auto xform      = dataset.xforms[i + first];
        float det_start = determinant(mat3(xform.start));
        float det_end   = determinant(mat3(xform.end));
        if (distance(det_start, 1.0f) > 0.01f || distance(det_end, 1.0f) > 0.01f)
        {
            tlog::warning() << "Rotation of camera matrix in frame " << i + first
                            << " has a scaling component (determinant!=1).";
            tlog::warning() << "Normalizing the matrix. This hints at an issue in your data generation pipeline and "
                               "should be fixed.";

            xform.start[0] /= std::cbrt(det_start);
            xform.start[1] /= std::cbrt(det_start);
            xform.start[2] /= std::cbrt(det_start);
            xform.end[0] /= std::cbrt(det_end);
            xform.end[1] /= std::cbrt(det_end);
            xform.end[2] /= std::cbrt(det_end);
            dataset.xforms[i + first] = xform;
        }

        mat3 rot       = rotmat(cam_rot_offset[i + first].variable());
        auto rot_start = rot * mat3(xform.start);
        auto rot_end   = rot * mat3(xform.end);
        xform.start    = mat4x3(rot_start[0], rot_start[1], rot_start[2], xform.start[3]);
        xform.end      = mat4x3(rot_end[0], rot_end[1], rot_end[2], xform.end[3]);

        xform.start[3] += cam_pos_offset[i + first].variable();
        xform.end[3] += cam_pos_offset[i + first].variable();
        transforms[i + first] = xform;
    }

    transforms_gpu.enlarge(last);
    CUDA_CHECK_THROW(cudaMemcpy(
        transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
}

void Testbed::create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr)
{
    m_data_path = {};
    set_mode(ETestbedMode::Nerf);
    m_nerf.training.dataset = ngp::create_empty_nerf_dataset(n_images, aabb_scale, is_hdr);
    load_nerf(m_data_path);
    m_nerf.training.n_images_for_training = 0;
    m_training_data_available             = true;
}

void Testbed::load_nerf_post()
{ // moved the second half of load_nerf here
    m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

    m_nerf.training.n_images_for_training = (int) m_nerf.training.dataset.n_images;

    m_nerf.training.dataset.update_metadata();

    m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
    m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

    m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
    m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
    m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
    m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

    m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
    m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

    m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
    m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
    m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

    m_nerf.training.cam_focal_length_gradient = vec2(0.0f);
    m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient,
                                                                            1);

    m_nerf.reset_extra_dims(m_rng);
    m_nerf.training.optimize_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;

    if (m_nerf.training.dataset.has_rays)
    {
        m_nerf.training.near_distance = 0.0f;
    }

    // Perturbation of the training cameras -- for debugging the online extrinsics learning code
    // float perturb_amount = 0.01f;
    // if (perturb_amount > 0.f) {
    // 	for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
    // 		vec3 rot = (random_val_3d(m_rng) * 2.0f - 1.0f) * perturb_amount;
    // 		vec3 trans = (random_val_3d(m_rng) * 2.0f - 1.0f) * perturb_amount;
    // 		float angle = length(rot);
    // 		rot /= angle;

    // 		auto rot_start = rotmat(angle, rot) * mat3(m_nerf.training.dataset.xforms[i].start);
    // 		auto rot_end = rotmat(angle, rot) * mat3(m_nerf.training.dataset.xforms[i].end);
    // 		m_nerf.training.dataset.xforms[i].start = mat4x3(rot_start[0], rot_start[1], rot_start[2], m_nerf.training.dataset.xforms[i].start[3] + trans);
    // 		m_nerf.training.dataset.xforms[i].end = mat4x3(rot_end[0], rot_end[1], rot_end[2], m_nerf.training.dataset.xforms[i].end[3] + trans);
    // 	}
    // }

    m_nerf.training.update_transforms();

    if (!m_nerf.training.dataset.metadata.empty())
    {
        m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
        m_screen_center    = vec2(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
    }

    if (!is_pot(m_nerf.training.dataset.aabb_scale))
    {
        throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.",
                                             m_nerf.training.dataset.aabb_scale)};
    }

    int max_aabb_scale = 1 << (NERF_CASCADES() - 1);
    if (m_nerf.training.dataset.aabb_scale > max_aabb_scale)
    {
        throw std::runtime_error{fmt::format(
            "NeRF dataset must have `aabb_scale <= {}`, but is {}. "
            "You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
            max_aabb_scale,
            m_nerf.training.dataset.aabb_scale)};
    }

    m_aabb = BoundingBox{vec3(0.5f), vec3(0.5f)};
    m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES() - 1), m_nerf.training.dataset.aabb_scale));
    m_raw_aabb             = m_aabb;
    m_render_aabb          = m_aabb;
    m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
    if (!m_nerf.training.dataset.render_aabb.is_empty())
    {
        m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
    }

    m_nerf.max_cascade = 0;
    while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale)
    {
        ++m_nerf.max_cascade;
    }

    // Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
    // stepping in larger scenes.
    m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

    m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::load_nerf(const fs::path& data_path)
{
    if (!data_path.empty())
    {
        std::vector<fs::path> json_paths;
        if (data_path.is_directory())
        {
            for (const auto& path : fs::directory{data_path})
            {
                if (path.is_file() && equals_case_insensitive(path.extension(), "json"))
                {
                    json_paths.emplace_back(path);
                }
            }
        }
        else if (equals_case_insensitive(data_path.extension(), "json"))
        {
            json_paths.emplace_back(data_path);
        }
        else
        {
            throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
        }

        const auto prev_aabb_scale = m_nerf.training.dataset.aabb_scale;

        m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);

        // Check if the NeRF network has been previously configured.
        // If it has not, don't reset it.
        if (m_nerf.training.dataset.aabb_scale != prev_aabb_scale && m_nerf_network)
        {
            // The AABB scale affects network size indirectly. If it changed after loading,
            // we need to reset the previously configured network to keep a consistent internal state.
            reset_network();
        }
    }

    load_nerf_post();
}

void Testbed::update_density_grid_nerf(float decay,
                                       uint32_t n_uniform_density_grid_samples,
                                       uint32_t n_nonuniform_density_grid_samples,
                                       cudaStream_t stream)
{
    const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);

    m_nerf.density_grid.resize(n_elements);

    const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

    const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

    GPUMemoryArena::Allocation alloc;
    auto scratch = allocate_workspace_and_distribute<
        NerfPosition, // positions at which the NN will be queried for density evaluation
        uint32_t,     // indices of corresponding density grid cells
        float,        // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
        network_precision_t // output of the MLP before being converted to densities.
        >(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

    NerfPosition* density_grid_positions = std::get<0>(scratch);
    uint32_t* density_grid_indices       = std::get<1>(scratch);
    float* density_grid_tmp              = std::get<2>(scratch);
    network_precision_t* mlp_out         = std::get<3>(scratch);

    if (m_training_step == 0 || m_nerf.training.n_images_for_training != m_nerf.training.n_images_for_training_prev)
    {
        m_nerf.training.n_images_for_training_prev = m_nerf.training.n_images_for_training;
        if (m_training_step == 0)
        {
            m_nerf.density_grid_ema_step = 0;
        }
        // Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
        if (!m_nerf.training.dataset.has_rays)
        {
            linear_kernel(mark_untrained_density_grid,
                          0,
                          stream,
                          n_elements,
                          m_nerf.density_grid.data(),
                          m_nerf.training.n_images_for_training,
                          m_nerf.training.dataset.metadata_gpu.data(),
                          m_nerf.training.transforms_gpu.data(),
                          m_training_step == 0);
        }
        else
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float) * n_elements, stream));
        }
    }

    uint32_t n_steps = 1;
    for (uint32_t i = 0; i < n_steps; ++i)
    {
        CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float) * n_elements, stream));

        linear_kernel(generate_grid_samples_nerf_nonuniform,
                      0,
                      stream,
                      n_uniform_density_grid_samples,
                      m_nerf.training.density_grid_rng,
                      m_nerf.density_grid_ema_step,
                      m_aabb,
                      m_nerf.density_grid.data(),
                      density_grid_positions,
                      density_grid_indices,
                      m_nerf.max_cascade + 1,
                      -0.01f);
        m_nerf.training.density_grid_rng.advance();

        linear_kernel(generate_grid_samples_nerf_nonuniform,
                      0,
                      stream,
                      n_nonuniform_density_grid_samples,
                      m_nerf.training.density_grid_rng,
                      m_nerf.density_grid_ema_step,
                      m_aabb,
                      m_nerf.density_grid.data(),
                      density_grid_positions + n_uniform_density_grid_samples,
                      density_grid_indices + n_uniform_density_grid_samples,
                      m_nerf.max_cascade + 1,
                      NERF_MIN_OPTICAL_THICKNESS());
        m_nerf.training.density_grid_rng.advance();

        // Evaluate density at the spawned locations in batches.
        // Otherwise, we can exhaust the maximum index range of cutlass
        size_t batch_size = NERF_GRID_N_CELLS() * 2;

        for (size_t i = 0; i < n_density_grid_samples; i += batch_size)
        {
            batch_size = std::min(batch_size, n_density_grid_samples - i);

            GPUMatrix<network_precision_t, RM> density_matrix(mlp_out + i, padded_output_width, batch_size);
            GPUMatrix<float> density_grid_position_matrix(
                (float*) (density_grid_positions + i), sizeof(NerfPosition) / sizeof(float), batch_size);
            m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);
        }

        linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor,
                      0,
                      stream,
                      n_density_grid_samples,
                      density_grid_indices,
                      mlp_out,
                      density_grid_tmp,
                      m_nerf.rgb_activation,
                      m_nerf.density_activation);
        linear_kernel(ema_grid_samples_nerf,
                      0,
                      stream,
                      n_elements,
                      decay,
                      m_nerf.density_grid_ema_step,
                      m_nerf.density_grid.data(),
                      density_grid_tmp);

        ++m_nerf.density_grid_ema_step;
    }

    update_density_grid_mean_and_bitfield(stream);
}

void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream)
{
    const uint32_t n_elements = NERF_GRID_N_CELLS();

    size_t size_including_mips = grid_mip_offset(NERF_CASCADES()) / 8;
    m_nerf.density_grid_bitfield.enlarge(size_including_mips);
    m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

    CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
    reduce_sum(
        m_nerf.density_grid.data(),
        [n_elements] __device__(float val) { return fmaxf(val, 0.f) / (n_elements); },
        m_nerf.density_grid_mean.data(),
        n_elements,
        stream);

    linear_kernel(grid_to_bitfield,
                  0,
                  stream,
                  n_elements / 8 * NERF_CASCADES(),
                  n_elements / 8 * (m_nerf.max_cascade + 1),
                  m_nerf.density_grid.data(),
                  m_nerf.density_grid_bitfield.data(),
                  m_nerf.density_grid_mean.data());

    for (uint32_t level = 1; level < NERF_CASCADES(); ++level)
    {
        linear_kernel(bitfield_max_pool,
                      0,
                      stream,
                      n_elements / 64,
                      m_nerf.get_density_grid_bitfield_mip(level - 1),
                      m_nerf.get_density_grid_bitfield_mip(level));
    }

    set_all_devices_dirty();
}

__global__ void mark_density_grid_in_sphere_empty_kernel(const uint32_t n_elements,
                                                         float* density_grid,
                                                         vec3 pos,
                                                         float radius)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    // Random position within that cellq
    uint32_t level   = i / NERF_GRID_N_CELLS();
    uint32_t pos_idx = i % NERF_GRID_N_CELLS();

    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    float cell_radius = scalbnf(SQRT3(), level) / NERF_GRIDSIZE();
    vec3 cell_pos = ((vec3{(float) x + 0.5f, (float) y + 0.5f, (float) z + 0.5f}) / (float) NERF_GRIDSIZE() - 0.5f) *
                        scalbnf(1.0f, level) +
                    0.5f;

    // Disable if the cell touches the sphere (conservatively, by bounding the cell with a sphere)
    if (distance(pos, cell_pos) < radius + cell_radius)
    {
        density_grid[i] = -1.0f;
    }
}

void Testbed::mark_density_grid_in_sphere_empty(const vec3& pos, float radius, cudaStream_t stream)
{
    const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);
    if (m_nerf.density_grid.size() != n_elements)
    {
        return;
    }

    linear_kernel(
        mark_density_grid_in_sphere_empty_kernel, 0, stream, n_elements, m_nerf.density_grid.data(), pos, radius);

    update_density_grid_mean_and_bitfield(stream);
}

void Testbed::NerfCounters::prepare_for_training_steps(cudaStream_t stream)
{
    numsteps_counter.enlarge(1);
    numsteps_counter_compacted.enlarge(1);
    loss.enlarge(rays_per_batch);
    CUDA_CHECK_THROW(
        cudaMemsetAsync(numsteps_counter.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
    CUDA_CHECK_THROW(cudaMemsetAsync(
        numsteps_counter_compacted.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
    CUDA_CHECK_THROW(cudaMemsetAsync(loss.data(), 0, sizeof(float) * rays_per_batch, stream));
}

float Testbed::NerfCounters::update_after_training(uint32_t target_batch_size,
                                                   bool get_loss_scalar,
                                                   cudaStream_t stream)
{
    std::vector<uint32_t> counter_cpu(1);
    std::vector<uint32_t> compacted_counter_cpu(1);
    numsteps_counter.copy_to_host(counter_cpu);
    numsteps_counter_compacted.copy_to_host(compacted_counter_cpu);
    measured_batch_size                   = 0;
    measured_batch_size_before_compaction = 0;

    if (counter_cpu[0] == 0 || compacted_counter_cpu[0] == 0)
    {
        return 0.f;
    }

    measured_batch_size_before_compaction = counter_cpu[0];
    measured_batch_size                   = compacted_counter_cpu[0];

    float loss_scalar = 0.0;
    if (get_loss_scalar)
    {
        loss_scalar =
            reduce_sum(loss.data(), rays_per_batch, stream) * (float) measured_batch_size / (float) target_batch_size;
    }

    rays_per_batch = (uint32_t) ((float) rays_per_batch * (float) target_batch_size / (float) measured_batch_size);
    rays_per_batch = std::min(next_multiple(rays_per_batch, BATCH_SIZE_GRANULARITY), 1u << 18);

    return loss_scalar;
}

void Testbed::train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream)
{
    if (m_nerf.training.n_images_for_training == 0)
    {
        return;
    }

    if (m_nerf.training.include_sharpness_in_error)
    {
        size_t n_cells = NERF_GRID_N_CELLS() * NERF_CASCADES();
        if (m_nerf.training.sharpness_grid.size() < n_cells)
        {
            m_nerf.training.sharpness_grid.enlarge(NERF_GRID_N_CELLS() * NERF_CASCADES());
            CUDA_CHECK_THROW(cudaMemsetAsync(
                m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
        }

        if (m_training_step == 0)
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(
                m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
        }
        else
        {
            linear_kernel(decay_sharpness_grid_nerf,
                          0,
                          stream,
                          m_nerf.training.sharpness_grid.size(),
                          0.95f,
                          m_nerf.training.sharpness_grid.data());
        }
    }
    m_nerf.training.counters_rgb.prepare_for_training_steps(stream);

    if (m_nerf.training.n_steps_since_cam_update == 0)
    {
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(),
                                         0,
                                         m_nerf.training.cam_exposure_gradient_gpu.get_bytes(),
                                         stream));
        CUDA_CHECK_THROW(
            cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float) * m_distortion.map->n_params(), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_distortion.map->gradient_weights(), 0, sizeof(float) * m_distortion.map->n_params(), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(),
                                         0,
                                         m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(),
                                         stream));
    }

    bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
    uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
    if (train_extra_dims)
    {
        uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
        m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
        CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(),
                                         0,
                                         m_nerf.training.extra_dims_gradient_gpu.get_bytes(),
                                         stream));
    }

    if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata.empty())
    {
        uint32_t n_samples_per_image =
            (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb.rays_per_batch) /
            m_nerf.training.dataset.n_images;
        ivec2 res = m_nerf.training.dataset.metadata[0].resolution;
        m_nerf.training.error_map.resolution =
            min(ivec2((int) (std::sqrt(std::sqrt((float) n_samples_per_image)) * 3.5f)), res);
        m_nerf.training.error_map.data.resize(product(m_nerf.training.error_map.resolution) *
                                              m_nerf.training.dataset.n_images);
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
    }

    float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
    if (envmap_gradient)
    {
        CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float) * m_envmap.envmap->n_params(), stream));
    }

    train_nerf_step(target_batch_size, m_nerf.training.counters_rgb, stream);

    m_trainer->optimizer_step(stream, LOSS_SCALE());

    ++m_training_step;

    if (envmap_gradient)
    {
        m_envmap.trainer->optimizer_step(stream, LOSS_SCALE());
    }

    float loss_scalar = m_nerf.training.counters_rgb.update_after_training(target_batch_size, get_loss_scalar, stream);
    bool zero_records = m_nerf.training.counters_rgb.measured_batch_size == 0;
    if (get_loss_scalar)
    {
        m_loss_scalar.update(loss_scalar);
    }

    if (zero_records)
    {
        m_loss_scalar.set(0.f);
        tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
        m_train = false;
    }

    // Compute CDFs from the error map
    m_nerf.training.n_steps_since_error_map_update += 1;
    // This is low-overhead enough to warrant always being on.
    // It makes for useful visualizations of the training error.
    bool accumulate_error = true;
    if (accumulate_error &&
        m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates)
    {
        m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
        m_nerf.training.error_map.cdf_x_cond_y.resize(product(m_nerf.training.error_map.cdf_resolution) *
                                                      m_nerf.training.dataset.n_images);
        m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y *
                                               m_nerf.training.dataset.n_images);
        m_nerf.training.error_map.cdf_img.resize(m_nerf.training.dataset.n_images);

        CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(),
                                         0,
                                         m_nerf.training.error_map.cdf_x_cond_y.get_bytes(),
                                         stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(
            m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

        const dim3 threads = {16, 8, 1};
        const dim3 blocks  = {div_round_up((uint32_t) m_nerf.training.error_map.cdf_resolution.y, threads.x),
                              div_round_up((uint32_t) m_nerf.training.dataset.n_images, threads.y),
                              1};
        construct_cdf_2d<<<blocks, threads, 0, stream>>>(m_nerf.training.dataset.n_images,
                                                         m_nerf.training.error_map.cdf_resolution.y,
                                                         m_nerf.training.error_map.cdf_resolution.x,
                                                         m_nerf.training.error_map.data.data(),
                                                         m_nerf.training.error_map.cdf_x_cond_y.data(),
                                                         m_nerf.training.error_map.cdf_y.data());
        linear_kernel(construct_cdf_1d,
                      0,
                      stream,
                      m_nerf.training.dataset.n_images,
                      m_nerf.training.error_map.cdf_resolution.y,
                      m_nerf.training.error_map.cdf_y.data(),
                      m_nerf.training.error_map.cdf_img.data());

        // Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
        m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
        m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
        std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
        float cum                      = 0;
        for (float& f : cdf_img_cpu)
        {
            cum += f;
            f = cum;
        }
        float norm = 1.0f / cum;
        for (size_t i = 0; i < cdf_img_cpu.size(); ++i)
        {
            constexpr float MIN_PMF = 0.1f;
            m_nerf.training.error_map.pmf_img_cpu[i] =
                (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm +
                MIN_PMF / (float) m_nerf.training.dataset.n_images;
            cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm +
                             MIN_PMF * (float) (i + 1) / (float) m_nerf.training.dataset.n_images;
        }
        m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

        // Reset counters and decrease update rate.
        m_nerf.training.n_steps_since_error_map_update = 0;
        m_nerf.training.n_rays_since_error_map_update  = 0;
        m_nerf.training.error_map.is_cdf_valid         = true;

        m_nerf.training.n_steps_between_error_map_updates =
            (uint32_t) (m_nerf.training.n_steps_between_error_map_updates * 1.5f);
    }

    // Get extrinsics gradients
    m_nerf.training.n_steps_since_cam_update += 1;

    if (train_extra_dims)
    {
        std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
        m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);

        // Optimization step
        for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i)
        {
            std::vector<float> gradient(n_extra_dims);
            for (uint32_t j = 0; j < n_extra_dims; ++j)
            {
                gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE();
            }

            //float l2_reg = 1e-4f;
            //gradient += m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

            m_nerf.training.extra_dims_opt[i].set_learning_rate(m_optimizer->learning_rate());
            m_nerf.training.extra_dims_opt[i].step(gradient);
        }

        m_nerf.training.update_extra_dims();
    }

    bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion ||
                        m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
    if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates)
    {
        float per_camera_loss_scale = (float) m_nerf.training.n_images_for_training / LOSS_SCALE() /
                                      (float) m_nerf.training.n_steps_between_cam_updates;

        if (m_nerf.training.optimize_extrinsics)
        {
            CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(),
                                             m_nerf.training.cam_pos_gradient_gpu.data(),
                                             m_nerf.training.cam_pos_gradient_gpu.get_bytes(),
                                             cudaMemcpyDeviceToHost,
                                             stream));
            CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(),
                                             m_nerf.training.cam_rot_gradient_gpu.data(),
                                             m_nerf.training.cam_rot_gradient_gpu.get_bytes(),
                                             cudaMemcpyDeviceToHost,
                                             stream));

            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

            // Optimization step
            for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i)
            {
                vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
                vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

                float l2_reg = m_nerf.training.extrinsic_l2_reg;
                pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
                rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

                m_nerf.training.cam_pos_offset[i].set_learning_rate(
                    std::max(m_nerf.training.extrinsic_learning_rate *
                                 std::pow(0.33f, (float) (m_nerf.training.cam_pos_offset[i].step() / 128)),
                             m_optimizer->learning_rate() / 1000.0f));
                m_nerf.training.cam_rot_offset[i].set_learning_rate(
                    std::max(m_nerf.training.extrinsic_learning_rate *
                                 std::pow(0.33f, (float) (m_nerf.training.cam_rot_offset[i].step() / 128)),
                             m_optimizer->learning_rate() / 1000.0f));

                m_nerf.training.cam_pos_offset[i].step(pos_gradient);
                m_nerf.training.cam_rot_offset[i].step(rot_gradient);
            }

            m_nerf.training.update_transforms();
        }

        if (m_nerf.training.optimize_distortion)
        {
            linear_kernel(safe_divide,
                          0,
                          stream,
                          m_distortion.map->n_params(),
                          m_distortion.map->gradients(),
                          m_distortion.map->gradient_weights());
            m_distortion.trainer->optimizer_step(stream,
                                                 LOSS_SCALE() * (float) m_nerf.training.n_steps_between_cam_updates);
        }

        if (m_nerf.training.optimize_focal_length)
        {
            CUDA_CHECK_THROW(cudaMemcpyAsync(&m_nerf.training.cam_focal_length_gradient,
                                             m_nerf.training.cam_focal_length_gradient_gpu.data(),
                                             m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(),
                                             cudaMemcpyDeviceToHost,
                                             stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
            vec2 focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
            float l2_reg               = m_nerf.training.intrinsic_l2_reg;
            focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
            m_nerf.training.cam_focal_length_offset.set_learning_rate(
                std::max(1e-3f * std::pow(0.33f, (float) (m_nerf.training.cam_focal_length_offset.step() / 128)),
                         m_optimizer->learning_rate() / 1000.0f));
            m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
            m_nerf.training.dataset.update_metadata();
        }

        if (m_nerf.training.optimize_exposure)
        {
            CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(),
                                             m_nerf.training.cam_exposure_gradient_gpu.data(),
                                             m_nerf.training.cam_exposure_gradient_gpu.get_bytes(),
                                             cudaMemcpyDeviceToHost,
                                             stream));

            vec3 mean_exposure = vec3(0.0f);

            // Optimization step
            for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i)
            {
                vec3 gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

                float l2_reg = m_nerf.training.exposure_l2_reg;
                gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

                m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
                m_nerf.training.cam_exposure[i].step(gradient);

                mean_exposure += m_nerf.training.cam_exposure[i].variable();
            }

            mean_exposure /= (float) m_nerf.training.n_images_for_training;

            // Renormalize
            std::vector<vec3> cam_exposures(m_nerf.training.n_images_for_training);
            for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i)
            {
                cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
            }

            CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(),
                                             cam_exposures.data(),
                                             m_nerf.training.n_images_for_training * sizeof(vec3),
                                             cudaMemcpyHostToDevice,
                                             stream));
        }

        m_nerf.training.n_steps_since_cam_update = 0;
    }
}

void Testbed::train_nerf_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream)
{
    const uint32_t padded_output_width = m_network->padded_output_width();
    const uint32_t max_samples         = target_batch_size * 16; // Somewhat of a worst case
    const uint32_t floats_per_coord    = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
    const uint32_t extra_stride =
        m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    GPUMemoryArena::Allocation alloc;
    auto scratch = allocate_workspace_and_distribute<uint32_t,            // ray_indices
                                                     Ray,                 // rays
                                                     uint32_t,            // numsteps
                                                     float,               // coords
                                                     float,               // max_level
                                                     network_precision_t, // mlp_out
                                                     network_precision_t, // dloss_dmlp_out
                                                     float,               // coords_compacted
                                                     float,               // coords_gradient
                                                     float,               // max_level_compacted
                                                     uint32_t             // ray_counter
                                                     >(stream,
                                                       &alloc,
                                                       counters.rays_per_batch,
                                                       counters.rays_per_batch,
                                                       counters.rays_per_batch * 2,
                                                       max_samples * floats_per_coord,
                                                       max_samples,
                                                       std::max(target_batch_size, max_samples) * padded_output_width,
                                                       target_batch_size * padded_output_width,
                                                       target_batch_size * floats_per_coord,
                                                       target_batch_size * floats_per_coord,
                                                       target_batch_size,
                                                       1);

    // TODO: C++17 structured binding
    uint32_t* ray_indices               = std::get<0>(scratch);
    Ray* rays_unnormalized              = std::get<1>(scratch);
    uint32_t* numsteps                  = std::get<2>(scratch);
    float* coords                       = std::get<3>(scratch);
    float* max_level                    = std::get<4>(scratch);
    network_precision_t* mlp_out        = std::get<5>(scratch);
    network_precision_t* dloss_dmlp_out = std::get<6>(scratch);
    float* coords_compacted             = std::get<7>(scratch);
    float* coords_gradient              = std::get<8>(scratch);
    float* max_level_compacted          = std::get<9>(scratch);
    uint32_t* ray_counter               = std::get<10>(scratch);

    uint32_t max_inference;
    if (counters.measured_batch_size_before_compaction == 0)
    {
        counters.measured_batch_size_before_compaction = max_inference = max_samples;
    }
    else
    {
        max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples),
                                      BATCH_SIZE_GRANULARITY);
    }

    GPUMatrix<float> compacted_coords_matrix((float*) coords_compacted, floats_per_coord, target_batch_size);
    GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

    GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

    if (m_training_step == 0)
    {
        counters.n_rays_total = 0;
    }

    uint32_t n_rays_total = counters.n_rays_total;
    counters.n_rays_total += counters.rays_per_batch;
    m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

    // If we have an envmap, prepare its gradient buffer
    float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

    bool sample_focal_plane_proportional_to_error =
        m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
    bool sample_image_proportional_to_error =
        m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
    bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
    // This is low-overhead enough to warrant always being on.
    // It makes for useful visualizations of the training error.
    bool accumulate_error = true;

    CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));

    auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

    {
        linear_kernel(
            generate_training_samples_nerf,
            0,
            stream,
            counters.rays_per_batch,
            m_aabb,
            max_inference,
            n_rays_total,
            m_rng,
            ray_counter,
            counters.numsteps_counter.data(),
            ray_indices,
            rays_unnormalized,
            numsteps,
            PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
            m_nerf.training.n_images_for_training,
            m_nerf.training.dataset.metadata_gpu.data(),
            m_nerf.training.transforms_gpu.data(),
            m_nerf.density_grid_bitfield.data(),
            m_nerf.max_cascade,
            m_max_level_rand_training,
            max_level,
            m_nerf.training.snap_to_pixel_centers,
            m_nerf.training.train_envmap,
            m_nerf.cone_angle_constant,
            m_distortion.view(),
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
            sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
            m_nerf.training.error_map.cdf_resolution,
            m_nerf.training.extra_dims_gpu.data(),
            m_nerf_network->n_extra_dims());

        if (hg_enc)
        {
            hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
        }

        GPUMatrix<float> coords_matrix((float*) coords, floats_per_coord, max_inference);
        GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
        m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

        if (hg_enc)
        {
            hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
        }

        linear_kernel(
            compute_loss_kernel_train_nerf,
            0,
            stream,
            counters.rays_per_batch,
            m_aabb,
            n_rays_total,
            m_rng,
            target_batch_size,
            ray_counter,
            LOSS_SCALE(),
            padded_output_width,
            m_envmap.view(),
            envmap_gradient,
            m_envmap.resolution,
            m_envmap.loss_type,
            m_background_color.rgb(),
            m_color_space,
            m_nerf.training.random_bg_color,
            m_nerf.training.linear_colors,
            m_nerf.training.n_images_for_training,
            m_nerf.training.dataset.metadata_gpu.data(),
            mlp_out,
            counters.numsteps_counter_compacted.data(),
            ray_indices,
            rays_unnormalized,
            numsteps,
            PitchedPtr<const NerfCoordinate>((NerfCoordinate*) coords, 1, 0, extra_stride),
            PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_compacted, 1, 0, extra_stride),
            dloss_dmlp_out,
            m_nerf.training.loss_type,
            m_nerf.training.depth_loss_type,
            counters.loss.data(),
            m_max_level_rand_training,
            max_level_compacted,
            m_nerf.rgb_activation,
            m_nerf.density_activation,
            m_nerf.training.snap_to_pixel_centers,
            accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
            sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
            m_nerf.training.error_map.resolution,
            m_nerf.training.error_map.cdf_resolution,
            include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
            m_nerf.training.dataset.sharpness_resolution,
            m_nerf.training.sharpness_grid.data(),
            m_nerf.density_grid.data(),
            m_nerf.density_grid_mean.data(),
            m_nerf.max_cascade,
            m_nerf.training.cam_exposure_gpu.data(),
            m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
            m_nerf.training.depth_supervision_lambda,
            m_nerf.training.near_distance);
    }

    fill_rollover_and_rescale<network_precision_t>
        <<<n_blocks_linear(target_batch_size * padded_output_width), N_THREADS_LINEAR, 0, stream>>>(
            target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out);
    fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), N_THREADS_LINEAR, 0, stream>>>(
        target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*) coords_compacted);
    fill_rollover<float><<<n_blocks_linear(target_batch_size), N_THREADS_LINEAR, 0, stream>>>(
        target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted);

    bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion ||
                        m_nerf.training.optimize_focal_length;
    bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
    bool prepare_input_gradients = train_camera || train_extra_dims;
    GPUMatrix<float> coords_gradient_matrix((float*) coords_gradient, floats_per_coord, target_batch_size);

    m_trainer->training_step(stream,
                             compacted_coords_matrix,
                             {},
                             nullptr,
                             false,
                             prepare_input_gradients ? &coords_gradient_matrix : nullptr,
                             false,
                             GradientMode::Overwrite,
                             &gradient_matrix);

    if (train_extra_dims)
    {
        // Compute extra-dim gradients
        linear_kernel(compute_extra_dims_gradient_train_nerf,
                      0,
                      stream,
                      counters.rays_per_batch,
                      n_rays_total,
                      ray_counter,
                      m_nerf.training.extra_dims_gradient_gpu.data(),
                      m_nerf.training.dataset.n_extra_dims(),
                      m_nerf.training.n_images_for_training,
                      ray_indices,
                      numsteps,
                      PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_gradient, 1, 0, extra_stride),
                      sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr);
    }

    if (train_camera)
    {
        // Compute camera gradients
        linear_kernel(
            compute_cam_gradient_train_nerf,
            0,
            stream,
            counters.rays_per_batch,
            n_rays_total,
            m_rng,
            m_aabb,
            ray_counter,
            m_nerf.training.transforms_gpu.data(),
            m_nerf.training.snap_to_pixel_centers,
            m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
            m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
            m_nerf.training.n_images_for_training,
            m_nerf.training.dataset.metadata_gpu.data(),
            ray_indices,
            rays_unnormalized,
            numsteps,
            PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_compacted, 1, 0, extra_stride),
            PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords_gradient, 1, 0, extra_stride),
            m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
            m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
            m_distortion.resolution,
            m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
            sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
            sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
            m_nerf.training.error_map.cdf_resolution);
    }

    m_rng.advance();

    if (hg_enc)
    {
        hg_enc->set_max_level_gpu(nullptr);
    }
}

void Testbed::training_prep_nerf(uint32_t batch_size, cudaStream_t stream)
{
    if (m_nerf.training.n_images_for_training == 0)
    {
        return;
    }

    float alpha         = m_nerf.training.density_grid_decay;
    uint32_t n_cascades = m_nerf.max_cascade + 1;

    if (m_training_step < 256)
    {
        update_density_grid_nerf(alpha, NERF_GRID_N_CELLS() * n_cascades, 0, stream);
    }
    else
    {
        update_density_grid_nerf(
            alpha, NERF_GRID_N_CELLS() / 4 * n_cascades, NERF_GRID_N_CELLS() / 4 * n_cascades, stream);
    }
}

void Testbed::optimise_mesh_step(uint32_t n_steps)
{
    uint32_t n_verts = (uint32_t) m_mesh.verts.size();
    if (!n_verts)
    {
        return;
    }

    const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();
    const uint32_t floats_per_coord    = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
    const uint32_t extra_stride        = m_nerf_network->n_extra_dims() * sizeof(float);
    GPUMemory<float> coords(n_verts * floats_per_coord);
    GPUMemory<network_precision_t> mlp_out(n_verts * padded_output_width);

    GPUMatrix<float> positions_matrix((float*) coords.data(), floats_per_coord, n_verts);
    GPUMatrix<network_precision_t, RM> density_matrix(mlp_out.data(), padded_output_width, n_verts);

    const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

    for (uint32_t i = 0; i < n_steps; ++i)
    {
        linear_kernel(generate_nerf_network_inputs_from_positions,
                      0,
                      m_stream.get(),
                      n_verts,
                      m_aabb,
                      m_mesh.verts.data(),
                      PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords.data(), 1, 0, extra_stride),
                      extra_dims_gpu);

        // For each optimizer step, we need the density at the given pos...
        m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
        // ...as well as the input gradient w.r.t. density, which we will store in the nerf coords.
        m_nerf_network->input_gradient(m_stream.get(), 3, positions_matrix, positions_matrix);
        // and the 1ring centroid for laplacian smoothing
        compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);

        // With these, we can compute a gradient that points towards the threshold-crossing of density...
        compute_mesh_opt_gradients(m_mesh.thresh,
                                   m_mesh.verts,
                                   m_mesh.vert_normals,
                                   m_mesh.verts_smoothed,
                                   mlp_out.data(),
                                   floats_per_coord,
                                   (const float*) coords.data(),
                                   m_mesh.verts_gradient,
                                   m_mesh.smooth_amount,
                                   m_mesh.density_amount,
                                   m_mesh.inflate_amount);

        // ...that we can pass to the optimizer.
        m_mesh.verts_optimizer->step(m_stream.get(),
                                     1.0f,
                                     (float*) m_mesh.verts.data(),
                                     (float*) m_mesh.verts.data(),
                                     (float*) m_mesh.verts_gradient.data());
    }
}

void Testbed::compute_mesh_vertex_colors()
{
    uint32_t n_verts = (uint32_t) m_mesh.verts.size();
    if (!n_verts)
    {
        return;
    }

    m_mesh.vert_colors.resize(n_verts);
    m_mesh.vert_colors.memset(0);

    if (m_testbed_mode == ETestbedMode::Nerf)
    {
        const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

        const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
        const uint32_t extra_stride     = m_nerf_network->n_extra_dims() * sizeof(float);
        GPUMemory<float> coords(n_verts * floats_per_coord);
        GPUMemory<float> mlp_out(n_verts * 4);

        GPUMatrix<float> positions_matrix((float*) coords.data(), floats_per_coord, n_verts);
        GPUMatrix<float> color_matrix(mlp_out.data(), 4, n_verts);
        linear_kernel(generate_nerf_network_inputs_from_positions,
                      0,
                      m_stream.get(),
                      n_verts,
                      m_aabb,
                      m_mesh.verts.data(),
                      PitchedPtr<NerfCoordinate>((NerfCoordinate*) coords.data(), 1, 0, extra_stride),
                      extra_dims_gpu);
        m_network->inference(m_stream.get(), positions_matrix, color_matrix);
        linear_kernel(extract_srgb_with_activation,
                      0,
                      m_stream.get(),
                      n_verts * 3,
                      3,
                      mlp_out.data(),
                      (float*) m_mesh.vert_colors.data(),
                      m_nerf.rgb_activation,
                      m_nerf.training.linear_colors);
    }
}

GPUMemory<float> Testbed::get_density_on_grid(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local)
{
    const uint32_t n_elements = (res3d.x * res3d.y * res3d.z);
    GPUMemory<float> density(n_elements);

    const uint32_t batch_size = std::min(n_elements, 1u << 20);
    bool nerf_mode            = m_testbed_mode == ETestbedMode::Nerf;

    const uint32_t padded_output_width =
        nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

    GPUMemoryArena::Allocation alloc;
    auto scratch = allocate_workspace_and_distribute<NerfPosition, network_precision_t>(
        m_stream.get(), &alloc, n_elements, batch_size * padded_output_width);

    NerfPosition* positions      = std::get<0>(scratch);
    network_precision_t* mlp_out = std::get<1>(scratch);

    const dim3 threads = {16, 8, 1};
    const dim3 blocks  = {div_round_up((uint32_t) res3d.x, threads.x),
                          div_round_up((uint32_t) res3d.y, threads.y),
                          div_round_up((uint32_t) res3d.z, threads.z)};

    BoundingBox unit_cube = BoundingBox{vec3(0.0f), vec3(1.0f)};
    generate_grid_samples_nerf_uniform<<<blocks, threads, 0, m_stream.get()>>>(
        res3d, m_nerf.density_grid_ema_step, aabb, render_aabb_to_local, nerf_mode ? m_aabb : unit_cube, positions);

    // Only process 1m elements at a time
    for (uint32_t offset = 0; offset < n_elements; offset += batch_size)
    {
        uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

        GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, local_batch_size);

        GPUMatrix<float> positions_matrix(
            (float*) (positions + offset), sizeof(NerfPosition) / sizeof(float), local_batch_size);
        if (nerf_mode)
        {
            m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
        }
        else
        {
            m_network->inference_mixed_precision(m_stream.get(), positions_matrix, density_matrix);
        }
        linear_kernel(grid_samples_half_to_float,
                      0,
                      m_stream.get(),
                      local_batch_size,
                      m_aabb,
                      density.data() + offset, //+ axis_step * n_elements,
                      mlp_out,
                      m_nerf.density_activation,
                      positions + offset,
                      nerf_mode ? m_nerf.density_grid.data() : nullptr,
                      m_nerf.max_cascade);
    }

    return density;
}

__global__ void filter_with_occupancy(const uint32_t n_elements,
                                      float* pos,
                                      const uint32_t floats_per_coord,
                                      const uint8_t* density_grid_bitfield,
                                      float* rgbsigma)
{
    const uint32_t point_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (point_id >= n_elements)
        return;
    const vec3 pos_vec{
        pos[point_id * floats_per_coord], pos[point_id * floats_per_coord + 1], pos[point_id * floats_per_coord + 2]};
    const uint32_t mip = mip_from_pos(pos_vec);
    if (!density_grid_occupied_at(pos_vec, density_grid_bitfield, mip))
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        { // sigma=0 would be enough, but all to 0 compress better
            rgbsigma[point_id * 4 + i] = 0.f;
        }
    }
}

GPUMemory<vec4> Testbed::get_rgba_on_grid(ivec3 res3d,
                                          vec3 ray_dir,
                                          bool voxel_centers,
                                          float depth,
                                          bool density_as_alpha)
{
    const uint32_t n_elements = (res3d.x * res3d.y * res3d.z);
    GPUMemory<vec4> rgba(n_elements);

    const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(m_stream.get());

    const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
    const uint32_t extra_stride     = m_nerf_network->n_extra_dims() * sizeof(float);

    GPUMemory<float> positions(n_elements * floats_per_coord);

    const uint32_t batch_size = std::min(n_elements, 1u << 20);

    // generate inputs
    const dim3 threads = {16, 8, 1};
    const dim3 blocks  = {div_round_up((uint32_t) res3d.x, threads.x),
                          div_round_up((uint32_t) res3d.y, threads.y),
                          div_round_up((uint32_t) res3d.z, threads.z)};
    generate_grid_samples_nerf_uniform_dir<<<blocks, threads, 0, m_stream.get()>>>(
        res3d,
        m_nerf.density_grid_ema_step,
        m_render_aabb,
        m_render_aabb_to_local,
        m_aabb,
        ray_dir,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*) positions.data(), 1, 0, extra_stride),
        extra_dims_gpu,
        voxel_centers);

    // Only process 1m elements at a time
    for (uint32_t offset = 0; offset < n_elements; offset += batch_size)
    {
        uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

        // run network
        GPUMatrix<float> positions_matrix(
            (float*) (positions.data() + offset * floats_per_coord), floats_per_coord, local_batch_size);
        GPUMatrix<float> rgbsigma_matrix((float*) (rgba.data() + offset), 4, local_batch_size);
        m_network->inference(m_stream.get(), positions_matrix, rgbsigma_matrix);
        linear_kernel(filter_with_occupancy,
                      0,
                      m_stream.get(),
                      local_batch_size,
                      positions_matrix.data(),
                      floats_per_coord,
                      m_nerf.density_grid_bitfield.data(),
                      rgbsigma_matrix.data());

        // convert network output to RGBA (in place)
        linear_kernel(compute_nerf_rgba_kernel,
                      0,
                      m_stream.get(),
                      local_batch_size,
                      rgba.data() + offset,
                      m_nerf.rgb_activation,
                      m_nerf.density_activation,
                      depth,
                      density_as_alpha);
    }
    return rgba;
}

int Testbed::marching_cubes(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local, float thresh)
{
    res3d.x = next_multiple((unsigned int) res3d.x, 16u);
    res3d.y = next_multiple((unsigned int) res3d.y, 16u);
    res3d.z = next_multiple((unsigned int) res3d.z, 16u);

    if (thresh == std::numeric_limits<float>::max())
    {
        thresh = m_mesh.thresh;
    }

    GPUMemory<float> density = get_density_on_grid(res3d, aabb, render_aabb_to_local);
    marching_cubes_gpu(
        m_stream.get(), aabb, render_aabb_to_local, res3d, thresh, density, m_mesh.verts, m_mesh.indices);

    uint32_t n_verts = (uint32_t) m_mesh.verts.size();
    m_mesh.verts_gradient.resize(n_verts);

    m_mesh.trainable_verts = std::make_shared<TrainableBuffer<3, 1, float>>(std::array<int, 1>{{(int) n_verts}});
    m_mesh.verts_gradient.copy_from_device(
        m_mesh.verts); // Make sure the vertices don't get destroyed in the initialization

    pcg32 rnd{m_seed};
    m_mesh.trainable_verts->initialize_params(rnd, (float*) m_mesh.verts.data());
    m_mesh.trainable_verts->set_params(
        (float*) m_mesh.verts.data(), (float*) m_mesh.verts.data(), (float*) m_mesh.verts_gradient.data());
    m_mesh.verts.copy_from_device(m_mesh.verts_gradient);

    m_mesh.verts_optimizer.reset(create_optimizer<float>({
        {"otype", "Adam"},
        {"learning_rate", 1e-4},
        {"beta1", 0.9f},
        {"beta2", 0.99f},
    }));

    m_mesh.verts_optimizer->allocate(m_mesh.trainable_verts);

    compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);
    compute_mesh_vertex_colors();

    return (int) (m_mesh.indices.size() / 3);
}

uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip)
{
    return density_grid_bitfield.data() + grid_mip_offset(mip) / 8;
}

void Testbed::Nerf::reset_extra_dims(default_rng_t& rng)
{
    uint32_t n_extra_dims = training.dataset.n_extra_dims();
    std::vector<float> extra_dims_cpu(
        n_extra_dims *
        (training.dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
    float* dst = extra_dims_cpu.data();
    training.extra_dims_opt =
        std::vector<VarAdamOptimizer>(training.dataset.n_images, VarAdamOptimizer(n_extra_dims, 1e-4f));
    for (uint32_t i = 0; i < training.dataset.n_images; ++i)
    {
        vec3 light_dir = warp_direction(normalize(training.dataset.metadata[i].light_dir));
        training.extra_dims_opt[i].reset_state();
        std::vector<float>& optimzer_value = training.extra_dims_opt[i].variable();
        for (uint32_t j = 0; j < n_extra_dims; ++j)
        {
            if (training.dataset.has_light_dirs && j < 3)
            {
                dst[j] = light_dir[j];
            }
            else
            {
                dst[j] = random_val(rng) * 2.0f - 1.0f;
            }
            optimzer_value[j] = dst[j];
        }
        dst += n_extra_dims;
    }
    training.extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);

    rendering_extra_dims.resize(training.dataset.n_extra_dims());
    CUDA_CHECK_THROW(cudaMemcpy(rendering_extra_dims.data(),
                                training.extra_dims_gpu.data(),
                                rendering_extra_dims.bytes(),
                                cudaMemcpyDeviceToDevice));
}

const float* Testbed::Nerf::get_rendering_extra_dims(cudaStream_t stream) const
{
    CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

    if (training.dataset.n_extra_dims() == 0)
    {
        return nullptr;
    }

    const float* extra_dims_src =
        rendering_extra_dims_from_training_view >= 0
            ? training.extra_dims_gpu.data() + rendering_extra_dims_from_training_view * training.dataset.n_extra_dims()
            : rendering_extra_dims.data();

    if (!training.dataset.has_light_dirs)
    {
        return extra_dims_src;
    }

    // the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
    // we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
    size_t size     = training.dataset.n_extra_dims() * sizeof(float);
    float* dims_gpu = training.extra_dims_gpu.data() + training.dataset.n_images * training.dataset.n_extra_dims();
    CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
    vec3 light_dir = warp_direction(normalize(light_dir));
    CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(vec3)), cudaMemcpyHostToDevice, stream));
    return dims_gpu;
}

int Testbed::Nerf::find_closest_training_view(mat4x3 pose) const
{
    int bestimage   = training.view;
    float bestscore = std::numeric_limits<float>::infinity();
    for (int i = 0; i < training.n_images_for_training; ++i)
    {
        float score = distance(training.transforms[i].start[3], pose[3]);
        score += 0.25f * distance(training.transforms[i].start[2], pose[2]);
        if (score < bestscore)
        {
            bestscore = score;
            bestimage = i;
        }
    }

    return bestimage;
}

void Testbed::Nerf::set_rendering_extra_dims_from_training_view(int trainview)
{
    if (!training.dataset.n_extra_dims())
    {
        throw std::runtime_error{"Dataset does not have extra dims."};
    }

    if (trainview < 0 || trainview >= training.dataset.n_images)
    {
        throw std::runtime_error{"Invalid training view."};
    }

    rendering_extra_dims_from_training_view = trainview;
}

void Testbed::Nerf::set_rendering_extra_dims(const std::vector<float>& vals)
{
    CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

    if (vals.size() != training.dataset.n_extra_dims())
    {
        throw std::runtime_error{fmt::format(
            "Invalid number of extra dims. Got {} but must be {}.", vals.size(), training.dataset.n_extra_dims())};
    }

    rendering_extra_dims_from_training_view = -1;
    rendering_extra_dims.copy_from_host(vals);
}

std::vector<float> Testbed::Nerf::get_rendering_extra_dims_cpu() const
{
    CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

    if (training.dataset.n_extra_dims() == 0)
    {
        return {};
    }

    std::vector<float> extra_dims_cpu(training.dataset.n_extra_dims());
    CUDA_CHECK_THROW(cudaMemcpy(extra_dims_cpu.data(),
                                get_rendering_extra_dims(nullptr),
                                rendering_extra_dims.bytes(),
                                cudaMemcpyDeviceToHost));

    return extra_dims_cpu;
}

} // namespace ngp
