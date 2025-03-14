/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace ngp {

struct SliceColors
{
	static const int kSize = 1;

    vec4 slice[kSize];

    inline __device__ __host__ vec4& operator[](int i)
    {
        return slice[i];
    }

    inline __device__ __host__ const vec4& operator[](int i) const
    {
        return slice[i];
    }
};

struct SliceDepths
{
    static const int kSize = 1;

    float slice[kSize];

    inline __device__ __host__ float& operator[](int i)
    {
        return slice[i];
    }

    inline __device__ __host__ const float& operator[](int i) const
    {
        return slice[i];
    }
};

struct RaysNerfSoa
{
    using ColorType = SliceColors;
    using DepthType = SliceDepths;

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
    void copy_from_other_async(int slice_count, const RaysNerfSoa& other, cudaStream_t stream)
    {
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(rgba, other.rgba, slice_count * size * sizeof(vec4), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(depth, other.depth, slice_count * size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(payload, other.payload, size * sizeof(NerfPayload), cudaMemcpyDeviceToDevice, stream));
    }
#endif

    void set(ColorType* rgba, DepthType* depth, NerfPayload* payload, size_t size, int slice_count)
    {
        this->rgba        = rgba;
        this->depth       = depth;
        this->payload     = payload;
        this->size        = size;
    }

    ColorType* rgba;
    DepthType* depth;
    NerfPayload* payload;
    size_t size;
};

} // namespace ngp
