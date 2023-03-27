/*
* This may read as a bunch of "errors" since
* the compiler does not recognize that his will be
* compiled using CUDA
*/

#pragma once


#include "kernel.cuh"
#include "cuda_runtime.h"
#include "cuda.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

using namespace std;

//function to get the best cost on the GPU
int GetBestCost(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int *dev_Q, cudaTextureObject_t dev_v, int *dev_section, cudaTextureObject_t dev_root, const float x, const float y, const float z,
	const int sctn, const float qPower, const int count, float *outCost) {
	int blocks = ceil(count / 1024.0f);
	GetBestCostKernel <<< blocks, 1024 >>> (dev_xstart, dev_xstop, dev_ystart, dev_ystop, dev_zstart,
		dev_zstop, dev_Q, dev_v, dev_section, dev_root, x, y, z, sctn, qPower, count, outCost);
#if _DEBUG
	cudaError_t er = cudaDeviceSynchronize();
#endif
	thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(outCost);
	thrust::device_vector<float>::iterator iter = thrust::min_element(d_ptr, d_ptr + count);
	int pos = thrust::device_pointer_cast(&(iter[0])) - d_ptr;
	return pos;
}


//function to check for intersections on the GPU
void checkIntersectionGPU(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, const float xcentroid, const float ycentroid, const float zcentroid, const float r_con, const float r_pos1,
	const float r_pos2, const float gamma, const float x, const float y, const float z, const int count, const int root, const float Qscale, const float scale,
	const int sister_branch, const float xstart, const float ystart, const float zstart, const float xstop, const float ystop, const float zstop,
	const int pos, const int child1, const int child2, bool *foundInt) {
	int blocks = ceil(count / 1024.0f);
	checkIntersectionKernel <<< blocks, 1024 >>> (dev_xstart, dev_xstop, dev_ystart, dev_ystop, dev_zstart, dev_zstop,
		dev_Q, xcentroid, ycentroid, zcentroid, r_con, r_pos1, r_pos2, gamma, x, y, z, count, root, Qscale, scale, sister_branch,
		xstart, ystart, zstart, xstop, ystop, zstop, pos, child1, child2, foundInt);
#if _DEBUG
	cudaError_t er = cudaDeviceSynchronize();
#endif
}

void updateGPUmemory(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int count, const int best_idx, const float xcentroid, const float ycentroid, const float zcentroid,
	const float x, const float y, const float z, int *dev_section, const int sctn) {
	updateIntersectionMem <<< 1, 1 >>> (dev_xstart, dev_xstop, dev_ystart, dev_ystop, dev_zstart,
		dev_zstop, count, best_idx, xcentroid, ycentroid, zcentroid, x, y, z, dev_section, sctn);
}

void pushBranchesGPU(const float *dev_xstart, const  float *dev_xstop, const  float *dev_ystart, const  float *dev_ystop,
	const float *dev_zstart, const float *dev_zstop, cudaTextureObject_t dev_root, const int *dev_Qs, const  int *dev_section, const int count, const float Qscale,
	const float gamma, const float scale, const int terminal_pts, const int *dev_children, int *dev_index, const float xstart, const float ystart,
	const float zstart, const float xstop, const float ystop, const float zstop, const float r_j, const float *dev_r, const int j, const int sister,
	const int parent, const bool flag) {

	int blocks;
	if (flag) {
		blocks = ceil(count / 1024.0f);
	}
	else {
		blocks = ceil((count - j) / 1024.0f);
	}
	checkIntersectionWithIndexKernel <<< blocks, 1024 >>> (dev_xstart, dev_xstop, dev_ystart, dev_ystop, dev_zstart,
		dev_zstop, dev_root, dev_Qs, dev_section, count, Qscale, gamma, scale, terminal_pts, dev_children, dev_index,
		xstart, ystart, zstart, xstop, ystop, zstop, r_j, dev_r, j, sister, parent, flag);
}


__global__ void __launch_bounds__(1024) checkIntersectionWithIndexKernel(const float *dev_xstart, const  float *dev_xstop, const  float *dev_ystart, const  float *dev_ystop,
	const float *dev_zstart, const float *dev_zstop, cudaTextureObject_t dev_root, const int *dev_Qs, const  int *dev_section, const int count, const float Qscale,
	const float gamma, const float scale, const int terminal_pts, const int *dev_children, int *dev_index, const float xstart, const float ystart,
	const float zstart, const float xstop, const float ystop, const float zstop, const float r_j, const float *dev_r, const int j, const int sister,
	const int parent, const bool flag) {

	int k;
	if (flag) {
		k = blockIdx.x * blockDim.x + threadIdx.x;//we are searching the whole array
	}
	else {
		k = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;//we are only searching the elements after j
	}
	if (k >= count) {
		return;
	}
	//skip parents, children, same branch, and sister
	if (dev_children[2 * j] != k && dev_children[(2 * j) + 1] != k && parent != k && sister != k && j != k) {
		float d = dist3DSegmentToSegmentBinaryGPU(dev_xstart[k], dev_ystart[k], dev_zstart[k], dev_xstop[k], dev_ystop[k], dev_zstop[k],
			xstart, ystart, zstart, xstop, ystop, zstop);
		//check for intersection
		if (d < (dev_r[k] + r_j)) {
			//check if the intersection occurs on the same main branch by traveling up the k branch
			int parent2 = tex1Dfetch<int>(dev_root, k);
			while (parent2 != j && parent2 != -1) {
				parent2 = tex1Dfetch<int>(dev_root, parent2);
			}
			if (parent2 == -1) {
				//check if the intersection occurs on the same main branch by traveling up the j branch
				int parent1 = tex1Dfetch<int>(dev_root, j);
				while (parent1 != k && parent1 != -1) {
					parent1 = tex1Dfetch<int>(dev_root, parent1);
				}
				//They were on different branches
				if (parent1 == -1) {
					*dev_index = k;
				}
			}
		}
	}
}


//kernel to compute connection costs of each branch and return lowest cost
__global__ void __launch_bounds__(1024) GetBestCostKernel(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, cudaTextureObject_t dev_v, const int *dev_section, cudaTextureObject_t dev_root, const float x, const float y, const float z,
	const int sctn, const float qPower, const int count, float *outCost)
{
	float costs;
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= count) {
		return;
	}
	if (sctn != dev_section[pos]) {// || tree(pos).split > max_split) {
		costs = INFINITY;
	}
	else {
		float Q = dev_Q[pos];
		float wTrunk = powf(Q + 1.0f, qPower);
		float wBranch = powf(Q, qPower);
		//centroid
		float norm = 1.0f / (1.0f + wTrunk + wBranch);
		float xcentroid = norm * (x + wTrunk * dev_xstart[pos] + wBranch * dev_xstop[pos]);
		float ycentroid = norm * (y + wTrunk * dev_ystart[pos] + wBranch * dev_ystop[pos]);
		float zcentroid = norm * (z + wTrunk * dev_zstart[pos] + wBranch * dev_zstop[pos]);

		//compute cost
		float old_length = powf(dev_xstart[pos] - dev_xstop[pos], 2.0f) + powf(dev_ystart[pos] - dev_ystop[pos], 2.0f) + powf(dev_zstart[pos] - dev_zstop[pos], 2.0f);
		float old_branch_l = powf(xcentroid - dev_xstop[pos], 2.0f) + powf(ycentroid - dev_ystop[pos], 2.0f) + powf(zcentroid - dev_zstop[pos], 2.0f);//length of old branch
		float new_branch_l = powf(x - xcentroid, 2.0f) + powf(y - ycentroid, 2.0f) + powf(z - zcentroid, 2.0f);//length of new branch
		float old_root_l = powf(dev_xstart[pos] - xcentroid, 2.0f) + powf(dev_ystart[pos] - ycentroid, 2.0f) + powf(dev_zstart[pos] - zcentroid, 2.0f);//length of old parent
		old_branch_l = powf(old_branch_l, 0.5f);
		new_branch_l = powf(new_branch_l, 0.5f);
		old_root_l = powf(old_root_l, 0.5f);
		old_length = powf(old_length, 0.5f);
		//sum cost
		float distance_cost = -(old_length * powf(Q, qPower));//starting negative cost since it is removing this branch from the tree
		distance_cost += old_branch_l * wBranch + new_branch_l + old_root_l * wTrunk;

		// Add new flow from perfusion point
		int loc = tex1Dfetch<int>(dev_root, pos);
		while (loc != -1) {
			distance_cost += tex1Dfetch<float>(dev_v, loc);
			loc = tex1Dfetch<int>(dev_root, loc);
		}
		costs = distance_cost;
	}
	outCost[pos] = costs;
}


//kernel to check for interesctions
__global__ void __launch_bounds__(1024) checkIntersectionKernel(const float *dev_xstart, const float *dev_xstop, const float *dev_ystart, const float *dev_ystop, const float *dev_zstart,
	const float *dev_zstop, const int *dev_Q, const float xcentroid, const float ycentroid, const float zcentroid, const float r_con, const float r_pos1,
	const float r_pos2, const float gamma, const float x, const float y, const float z, const int count, const int root, const float Qscale, const float scale, const int sister_branch,
	const float xstart, const float ystart, const float zstart, const float xstop, const float ystop, const float zstop, const int pos, const int child1, const int child2, bool *foundInt)
{
	int int_check = blockIdx.x * blockDim.x + threadIdx.x;
	if (int_check >= count || pos == int_check) {
		return;
	}


	float d_sc = INFINITY;
	if (root != int_check && sister_branch != int_check) {
		d_sc = dist3DSegmentToSegmentBinaryGPU(xstart, ystart, zstart, xcentroid, ycentroid, zcentroid,
			dev_xstart[int_check], dev_ystart[int_check], dev_zstart[int_check], dev_xstop[int_check], dev_ystop[int_check], dev_zstop[int_check]);
	}

	float d_cs = INFINITY;
	if (child1 != int_check && child2 != int_check) {
		d_cs = dist3DSegmentToSegmentBinaryGPU(xcentroid, ycentroid, zcentroid, xstop, ystop, zstop,
			dev_xstart[int_check], dev_ystart[int_check], dev_zstart[int_check], dev_xstop[int_check], dev_ystop[int_check], dev_zstop[int_check]);
	}

	float d_cx = dist3DSegmentToSegmentBinaryGPU(xcentroid, ycentroid, zcentroid, x, y, z,
		dev_xstart[int_check], dev_ystart[int_check], dev_zstart[int_check], dev_xstop[int_check], dev_ystop[int_check], dev_zstop[int_check]);

	float r_check = scale * Qscale * powf(dev_Q[int_check], 1.0f / gamma);//radii of possible intersecting branch
	//check if the shortest distance is smaller than the combined radii
	if (d_sc < (r_pos2 + r_check) || d_cs < (r_pos1 + r_check) || d_cx < (r_con + r_check)) {
		*foundInt = 1;
	}
}


// get the 3D minimum distance between 2 segments
__device__ float dist3DSegmentToSegmentBinaryGPU(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z, float s2x, float s2y, float s2z, float e2x, float e2y, float e2z) {
	float u[3], v[3], w[3], dP[3];
	u[0] = e1x - s1x;
	u[1] = e1y - s1y;
	u[2] = e1z - s1z;

	v[0] = e2x - s2x;
	v[1] = e2y - s2y;
	v[2] = e2z - s2z;

	w[0] = s1x - s2x;
	w[1] = s1y - s2y;
	w[2] = s1z - s2z;

	float a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
	float b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
	float c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	float d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
	float e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
	float   D = a * c - b * b;        // always >= 0
	float   sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
	float   tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0
	float	SMALL_NUM = .0000001f;

	// compute the line parameters of the two closest points
	if (D < SMALL_NUM) { // the lines are almost parallel
		sN = 0.0f;         // force using point P0 on segment S1
		sD = 1.0f;         // to prevent possible division by 0.0 later
		tN = e;
		tD = c;
	}
	else {                 // get the closest points on the infinite lines
		sN = (b*e - c * d);
		tN = (a*e - b * d);
		if (sN < 0.0f) {        // sc < 0 => the s=0 edge is visible
			sN = 0.0f;
			tN = e;
			tD = c;
		}
		else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
			sN = sD;
			tN = e + b;
			tD = c;
		}
	}

	if (tN < 0.0f) {            // tc < 0 => the t=0 edge is visible
		tN = 0.0f;
		// recompute sc for this edge
		if (-d < 0.0f) {
			sN = 0.0f;
		}
		else if (-d > a) {
			sN = sD;
		}
		else {
			sN = -d;
			sD = a;
		}
	}
	else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
		tN = tD;
		// recompute sc for this edge
		if ((-d + b) < 0.0f) {
			sN = 0.0f;
		}
		else if ((-d + b) > a) {
			sN = sD;
		}
		else {
			sN = (-d + b);
			sD = a;
		}
	}
	// finally do the division to get sc and tc
	sc = (abs(sN) < SMALL_NUM ? 0.0f : sN / sD);
	tc = (abs(tN) < SMALL_NUM ? 0.0f : tN / tD);

	// get the difference of the two closest points
	dP[0] = w[0] + (sc * u[0]) - (tc * v[0]);  // =  S1(sc) - S2(tc)
	dP[1] = w[1] + (sc * u[1]) - (tc * v[1]);  // =  S1(sc) - S2(tc)
	dP[2] = w[2] + (sc * u[2]) - (tc * v[2]);  // =  S1(sc) - S2(tc)

	float distance = pow(pow(dP[0], 2.0f) + pow(dP[1], 2.0f) + pow(dP[2], 2.0f), 0.5f);
	return distance;   // return the closest distance
}

//Kernel to update arrays after intersection check
__global__ void __launch_bounds__(1024) updateIntersectionMem(float *dev_xstart, float *dev_xstop, float *dev_ystart, float *dev_ystop, float *dev_zstart,
	float *dev_zstop, const int count, const int pos, const float xcentroid, const float ycentroid, const float zcentroid,
	const float x, const float y, const float z, int *dev_section, const int sctn) {
	dev_xstart[count] = xcentroid;
	dev_ystart[count] = ycentroid;
	dev_zstart[count] = zcentroid;
	dev_xstop[count] = dev_xstop[pos];
	dev_ystop[count] = dev_ystop[pos];
	dev_zstop[count] = dev_zstop[pos];

	//update branch: centroid -> new point	
	dev_xstart[count + 1] = xcentroid;
	dev_ystart[count + 1] = ycentroid;
	dev_zstart[count + 1] = zcentroid;
	dev_xstop[count + 1] = x;
	dev_ystop[count + 1] = y;
	dev_zstop[count + 1] = z;

	//update branch: old branch startpoint -> centroid
	dev_xstop[pos] = xcentroid;
	dev_ystop[pos] = ycentroid;
	dev_zstop[pos] = zcentroid;

	//update branch: centroid -> old branch endpoint
	dev_section[count] = dev_section[pos];

	//update branch: centroid -> new point			
	dev_section[count + 1] = sctn;
}