#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <limits>
//function to get the best cost branch to attach  to
int GetBestCost(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int *dev_Q, cudaTextureObject_t dev_v, int *dev_section, cudaTextureObject_t dev_root, const float x, const float y, const float z,
	const int sctn, const float qPower, const int count, float *outCostz);
//GPU kernel to get the best cost branch to attach  to
__global__ void GetBestCostKernel(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, cudaTextureObject_t dev_v, const int *dev_section, cudaTextureObject_t dev_root, const float x, const float y, const float z,
	const int sctn, const float qPower, const int count, float *outCost);
//function to check if attached a new endpoint to a branch causes an intersection
void checkIntersectionGPU(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, const float xcentroid, const float ycentroid, const float zcentroid,
	const float r_con, const float r_pos1, const float r_pos2, const float gamma, const float x, const float y, const float z, const int count, const int root,
	const float Qscale, const float scale, const int sister_branch, const float xstart, const float ystart, const float zstart, const float xstop, const float ystop, const float zstop,
	const int pos, const int child1, const int child2, bool *foundInt);
//function to update the gpu arrays while generated centerline tree
void updateGPUmemory(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int count, const int best_idx, const float xcentroid, const float ycentroid, const float zcentroid,
	const float x, const float y, const float z, int *dev_section, const int sctn);
//function to do the final intersection check on the gpu (pushing branches)
void pushBranchesGPU(const float *dev_xstart, const  float *dev_xstop, const  float *dev_ystart, const  float *dev_ystop,
	const float *dev_zstart, const float *dev_zstop, cudaTextureObject_t dev_root, const int *dev_Qs, const  int *dev_section, const int count, const float Qscale,
	const float gamma, const float scale, const int terminal_pts, const int *dev_children, int *dev_index, const float xstart, const float ystart,
	const float zstart, const float xstop, const float ystop, const float zstop, const float r_j, const float *dev_r, const int j, const int sister,
	const int parent, const bool flag);
//GPU kernel to check for an intersection and return the index of the branch in which an intersection was detected
__global__ void checkIntersectionWithIndexKernel(const float *dev_xstart, const  float *dev_xstop, const  float *dev_ystart, const  float *dev_ystop,
	const float *dev_zstart, const float *dev_zstop, cudaTextureObject_t dev_root, const int *dev_Qs, const  int *dev_section, const int count, const float Qscale,
	const float gamma, const float scale, const int terminal_pts, const int *dev_children, int *dev_index, const float xstart, const float ystart,
	const float zstart, const float xstop, const float ystop, const float zstop, const float r_j, const float *dev_r, const int j, const int sister,
	const int parent, const bool flag);
//function to update gpu memory durring final intersection check
__global__ void updateIntersectionMem(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int count, const int best_idx, const float xcentroid, const float ycentroid, const float zcentroid,
	const float x, const float y, const float z, int *dev_section, const int sctn);
//GPU kernel to check if attaching a new endpoints to a branch causes an intersection
__global__ void checkIntersectionKernel(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, const float xcentroid, const float ycentroid, const float zcentroid,
	const float r_con, const float r_pos1, const float r_pos2, const float gamma, const float x, const float y, const float z, const int count, const int root,
	const float Qscale, const float scale, const int sister_branch, const float xstart, const float ystart, const float zstart, const float xstop, const float ystop, const float zstop,
	const int pos, const int child1, const int child2, bool *foundInt);
//device function to find the distance between two arrays
__device__ float dist3DSegmentToSegmentBinaryGPU(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z,
	float s2x, float s2y, float s2z, float e2x, float e2y, float e2z);