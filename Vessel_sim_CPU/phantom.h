#pragma once
using namespace std;
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <limits>
#include <omp.h>
#include <queue>
#include <array>
#include <numeric>
#include <chrono>
#include <fstream>
#include <random>
#include <direct.h>
#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Core>


/**
*  Class to generate hepatic arterial phantom
*/
class Liver {
public:
	//main function to build the hepatic tree
	int build_tree(const int Nx, const int Ny, const int Nz, int terminal_pts, int tree_number, float scale, unsigned int seed, float outResolutionScaleFactor);//builds vessel tree
	
private:
	//way to store  x,y,z points in a vector
	struct xyz {
		float x, y, z;
	};
	//function to create centerline hepatic arterial tree
	void create_tree(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[],
		float zstop[], int root[], int section[], float length[], int Qs[], queue<array<float, 4>> region1, queue<array<float, 4>> region2, queue<array<float, 4>> region3,
		queue<array<float, 4>> region4, queue<array<float, 4>> region5, queue<array<float, 4>> region6, queue<array<float, 4>> region7,
		queue<array<float, 4>> region8, float gamma, int terminal_pts, int children[], int *count, float scale,
		const int Nx, const int Ny, const int Nz, unsigned __int8 perf[], mt19937 mt, int arrayAlloc,unsigned int dims[4]);
	
	//function to get the Caunoud section a point is in
	int get_section(int x, int y, int z, unsigned __int8 perf[], mt19937 mt,unsigned int dims[4]);//get section the endpoint is in
	
	//function to get the closest points on two intersecting arrays
	static float dist3D_Segment_to_Segment(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z, float s2x, float s2y, float s2z, float e2x, float e2y, float e2z, float P1[3], float P2[3]);
	
	//function to test if two arrays intersect
	static float dist3DSegmentToSegmentBinary(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z, float s2x, float s2y, float s2z, float e2x, float e2y, float e2z);
	
	// function to do the final intersection check on the CPU (by pushing branches)
	int push_branches(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[], float zstop[], int root[],
		int Qs[], int section[], int *count, float Qscale, float gamma_radii, float scale, int terminal_pts, int children[], float r[]);
	
	//function to check if two arrays are equal to two decimal places
	static bool isequal(float sx, float sy, float sz, float ex, float ey, float ez);
	
	// function to check if a new branch may go outside of the liver mask
	static bool isOutMask(float x1, float y1, float z1, float x2, float y2, float z2, unsigned __int8 perf[],unsigned int dims[4]);
	
	//function to get the best index from a cost array
	int getBestCostIndx( float all_costs[], int count);
	
	//function to perform polynomial interpolation of all branches
	void polyInterp(float xstart, float ystart, float zstart, float xstop, float ystop, float zstop, float startSlopeX,
		float startSlopeY, float startSlopeZ, float endSlopeX, float endSlopeY, float endSlopeZ, float linearPointX,
		float linearPointY, float linearPointZ,vector <xyz> *interpBranches, int i, float outResolutionScaleFactor);
	
	//function to perform polynomial interpolation of branches that were broken into multiple segments
	void polyInterpWithControlPoints(vector <xyz> controlPts, vector <xyz> *interpBranches, int i, float outResolutionScaleFactor);
	
	//function to optimize the bifurcation angles durring polynomial interpolation
	void optimizeAngles(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[], float zstop[], int root[],
		int Qs[], int *count, float gamma, float scale, int children[], float r[], int prePushCount, vector <xyz> *interpBranches,
		float outResolutionScaleFactor);

	//function to get the linear index of a a 4D array (Column Major Order)
	static void getLinearInd(unsigned int y, unsigned int x, unsigned int z, unsigned int l, unsigned int dims[4], int *index);
};

