/*
Title: Hepatic Vessel Phantom Simulation
Summary: This code exists to create hepatic phantom data
*/
#pragma once
#include "phantom.h"

#define constant_seed 0//turn on to make a constant PRNG seed phantom
#define Intersection_check 1//turn on to assure no intersecting vessel branches in the centerline tree

using namespace std;


/*
* Main function to run liver class
*/
int main() {

	// Variables 
	int endpoints;//number of endpoints to simulate
	int tree_number;//number of trees
	int Nx, Ny, Nz;//number of voxels in x,y,z
	float phantom_res, desiredRes; //resolution of the phantom [mm] and the output vessel phantom [mm]

	// Get inputs
	cout << "Number of trees to build: ";
	if (cin >> tree_number) {}
	else { cerr << "Error: Please input numbers of trees as an int, returning..." << endl; return 1; }

	cout << "Number of Endpoints per tree: ";
	if(cin >> endpoints){}
	else { cerr << "Error: Please input numbers of endpoints as an int, returning..." << endl; return 1; }

	cout << "Size of X Y and Z dimension of the blood demand map [voxels] (Example: 512 512 512): ";
	if (cin >> Nx >> Ny >> Nz) {}
	else { cerr << "Error: Please input numbers as X Y Z, returning..." << endl; return 1; }

	cout << "Blood demand map isotropic voxel resolution [mm] (Example: 0.77): ";
	if (cin >> phantom_res) {}
	else { cerr << "Error: Please input phantom resolution as a float, returning..." << endl; return 1; }

	cout << "Desired output resolution of vessels [mm] (Example: 0.3): ";
	if (cin >> desiredRes) {}
	else { cerr << "Error: Please input desired output resolution as a float, returning..." << endl; return 1; }

	// initialize random number tree
	int t = time(NULL);
	mt19937 mt;
	mt.seed(t);
	
	//Build required directory
	_mkdir("../Vessel_sim_CPU/CLines");

	// Build requested number of trees
	for (int i = 1; i <= tree_number; i++) {
		
		//initialize liver object
		Liver tree;


		//initialize PRNG with a random seed and write out seed number
		unsigned int seed = mt();
#if constant_seed
		seed = 1522380473;
		mt.seed(seed);
#endif
		string dir = "../Vessel_sim_CPU/CLines/seed_" + to_string(i) + ".txt";
		ofstream file;
		file.open(dir);
		file << seed;
		file.close();

		//random value for radii of common hepatic
		float initialResoltion = 1.54f;//initial resoluion in [mm]
		float scale = 1.3f + 1.0f * (mt() / (double)mt.max());//what to scale the final radius by [4,7]mm is what is set
		scale *= initialResoltion / phantom_res;//scale radius by resolution change 
		float outResolutionScaleFactor = phantom_res / desiredRes;
		int check = tree.build_tree(Nx, Ny, Nz, endpoints, i, scale, seed, outResolutionScaleFactor);//generate vasculature


		//check if we got stuck building the tree
		if (check != 0) {
			//try again until we do not get stuck
			while (check != 0) {
				cerr << "We got stuck somewhere creating that tree...trying again" << endl;
				check = tree.build_tree(Nx, Ny, Nz, endpoints, i, scale, seed, outResolutionScaleFactor);
			}
		}
	}
	cout << "Successfully generated " << tree_number << " vessel trees." << endl;
	return 0;
}

//function to create the centerline hepatic vessel tree object
void Liver::create_tree(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[], float zstop[], int root[], int section[],
	float length[], int Qs[], queue<array<float, 4>> region1, queue<array<float, 4>> region2, queue<array<float, 4>> region3,
	queue<array<float, 4>> region4, queue<array<float, 4>> region5, queue<array<float, 4>> region6, queue<array<float, 4>> region7, queue<array<float, 4>> region8,
	float gamma, int terminal_pts, int children[], int *count_pointer, float scale, const int Nx, const int Ny, const int Nz, unsigned __int8 perf[],
	mt19937 mt, int arrayAlloc,unsigned int dims[4]) {
	//	x,y,z start/stop: arrays to hold x,y,z start/stop points of all branches
	//	root: array to hold list of all parents for each branch
	//	Qs: arras to hold perfusion of each branch
	//	section: array to hold section each branch is in
	//	section: array to hold length of each branch
	//	count_pointer: total number of branches in tree 
	//	Qscale: normalize the perfusion
	//	gamma: gamma for Murrays Law
	//	scale: what to scale the resolution by
	//	terminal_pts: total number of terminal_pts
	//	children: array of the children for each branch
	//	arrayAlloc: size of arrays for memory allocation
	//	r: radius of each branch
	//	Nx, Ny, Nz: x,y,z dimension of blood demand map
	//	perf: blood demand map
	//	mt: PRNG
	//  regions: queues to store the <x,y,z,number of tries> for each counaud segment (8 of them)
	//	return: centerline hepatic arterial tree

	// User defined Variables
	omp_set_num_threads(omp_get_max_threads());
	int max_tries = 2;//maximum number of times a point can be tried to attach the tree before it is removed and resampled

	//function variables
	int offset = 0;//amount of points added back into the pool
	int removed = 0;//amount of points removed from the pool
	float qPower = 2.0f / gamma;
	float *all_costs = new float[arrayAlloc];
	int count = 1;//number of branches in the current tree
	int region = 1;//start at one since the first point has already been selected in segment 4
	int sctn = 0;//section of current endpoint 
	int order[] = { 4, 8, 2, 5, 3, 7, 6, 1 };//order to pull points from what section
	float *V = new float[arrayAlloc];//volumetric cost of increasing the number of endpoints
	int index;//subscripts converted to linear index

	//check if the initial segment is within the mask
	bool outMask = isOutMask(xstart[0], ystart[0], zstart[0], xstop[0], ystop[0], zstop[0], perf, dims);
	//iterate until initial segment is within the mask
	while (outMask) {
		getLinearInd((int)xstart[0], (int)ystart[0], (int)zstart[0], dims[3]-1, dims, &index);
		perf[index] = 1;
		getLinearInd((int)xstop[0], (int)ystop[0], (int)zstop[0], dims[3]-1, dims, &index);
		perf[index] = 1;
		float k, j, i, val;
		double mxInt = mt.max();
		int intialPtSections[2] = { 9, 3 };//segments to choose the initial points
		for (int pos = 0; pos < 2; pos++) {
			int found = 0;
			while (found == 0) {
				k = (mt() / mxInt) * Nz - 0.5f;
				j = (mt() / mxInt) * Ny - 0.5f;
				i = (mt() / mxInt) * Nx - 0.5f;
				val = 255.0f * (mt() / mxInt);
				getLinearInd((int)i, (int)j, (int)k, intialPtSections[pos], dims, &index);
				if (val < (int)perf[index]) {
					if (pos == 0) {
						xstart[pos] = i;
						ystart[pos] = j;
						zstart[pos] = k;
						perf[index] = -1;// dont allow multiple selections of the same point
					}
					else {
						xstop[pos - 1] = i;
						ystop[pos - 1] = j;
						zstop[pos - 1] = k;
						perf[index] = -1;// dont allow multiple selections of the same point
					}
					//perf((int)i, (int)j, (int)k, 10) = -1;// dont allow multiple selections of the same point
					found = 1;
				}
			}
		}
		outMask = isOutMask(xstart[0], ystart[0], zstart[0], xstop[0], ystop[0], zstop[0], perf, dims);
	}

	// Generate Initial Tree
	children[0] = -1;
	children[1] = -1;
	root[0] = -1;
	section[0] = 4;
	Qs[0] = 1;
	length[0] = pow(xstart[0] - xstop[0], 2.0f) + pow(ystart[0] - ystop[0], 2.0f) + pow(zstart[0] - zstop[0], 2.0f);
	length[0] = pow(length[0], 0.5f);
	V[0] = 0;


	// Loop to generate points
	for (int n = 0; n < terminal_pts + offset + removed; n++) {
		//all regions empty, resample
		if (region1.empty() && region2.empty() && region3.empty() && region4.empty() && region5.empty() && region6.empty() && region7.empty() && region8.empty()) {
			int cur_pts = (count - 1) / 2;
			cout << "Resampling " << terminal_pts - cur_pts << " endpoints." << endl;
			int counter = cur_pts;
			int sec;
			float i, j, k, val;
			double mxInt = mt.max();
			for (int pos = cur_pts; pos < terminal_pts; pos++) {
				int found = 0;
				while (found == 0) {
					k = (mt() / mxInt) * Nz - 0.5f;
					j = (mt() / mxInt) * Ny - 0.5f;
					i = (mt() / mxInt) * Nx - 0.5f;
					val = 255.0f * (mt() / mxInt);
					getLinearInd((int)i, (int)j, (int)k, dims[3]-1, dims, &index);
					if (val < perf[index]) {
						array<float, 4> point = { i, j, k, 0 };
						sec = get_section(i, j, k, perf, mt, dims);
						if (sec == 0) {
							pos -= 1;
						}
						switch (sec) {
						case 1: region1.push(point); break;
						case 2: region2.push(point); break;
						case 3: region3.push(point); break;
						case 4: region4.push(point); break;
						case 5: region5.push(point); break;
						case 6: region6.push(point); break;
						case 7: region7.push(point); break;
						case 8: region8.push(point); break;
						}
						counter++;
						perf[index] = -1; //dont allow the selection of the same point
						found = 1;
					}
				}
			}
		}
		if (n % 5000 == 0) {
			cout << "Starting branch: " << n - offset - removed << " of " << terminal_pts << endl;
		}
		// Location of Perfusion Point
		bool found = 1;
		array<float, 4> point;
		while (found) {
			switch (order[(region) % 8]) {
			case 1:
				if (!region1.empty()) {
					point = region1.front();
					region1.pop();
					found = 0;
				}
				break;
			case 2:
				if (!region2.empty()) {
					point = region2.front();
					region2.pop();
					found = 0;
				}
				break;
			case 3:
				if (!region3.empty()) {
					point = region3.front();
					region3.pop();
					found = 0;
				}
				break;
			case 4:
				if (!region4.empty()) {
					point = region4.front();
					region4.pop();
					found = 0;
				}
				break;
			case 5:
				if (!region5.empty()) {
					point = region5.front();
					region5.pop();
					found = 0;
				}
				break;
			case 6:
				if (!region6.empty()) {
					point = region6.front();
					region6.pop();
					found = 0;
				}
				break;
			case 7:
				if (!region7.empty()) {
					point = region7.front();
					region7.pop();
					found = 0;
				}
				break;
			case 8:
				if (!region8.empty()) {
					point = region8.front();
					region8.pop();
					found = 0;
				}
				break;
			}
			region++;
		}
		sctn = order[(region - 1) % 8];
		//let the initial tree form before we start counting point attempts
		if (count > 50) {
			point[3] += 1;
		}

		///////////Search Loop for best connection///////////////////////
		float best_cost = INFINITY;
		int best_idx = -1;


			//compute the cost of connecting to each segment in the current vessel tree
#pragma omp parallel for
			for (int pos = 0; pos < count; pos++) {
				float distance_cost;// cost of connecting to the current segment
				//Check regions
				//Do not connect to branches not in the correct half for the first 8 endpoints (allows a main branch to form to each Couinaud segment)
				if (count < 15) {
					if ((sctn >= 5 && section[pos] < 5 && section[pos] > 1) || (sctn < 5 && section[pos] >= 5 && sctn > 1)) {
						// allow the left and right hepatic to form
						if (count > 3) {
							all_costs[pos] = INT_MAX;
							continue;
						}
					}
				}
				// Do not allow connection to branches in the wrong section
				else {
					if (sctn != section[pos]) {
						all_costs[pos] = INT_MAX;
						continue;
					}
				}

				float wTrunk = pow(Qs[pos] + 1.0f, qPower);
				float wBranch = pow(Qs[pos], qPower);
				float centroid[3];
				float norm = 1.0f / (1.0f + wTrunk + wBranch);
				float xcentroid = norm * (point[0] + wTrunk * xstart[pos] + wBranch * xstop[pos]);
				float ycentroid = norm * (point[1] + wTrunk * ystart[pos] + wBranch * ystop[pos]);
				float zcentroid = norm * (point[2] + wTrunk * zstart[pos] + wBranch * zstop[pos]);

				//compute cost
				float old_length = pow(xstart[pos] - xstop[pos], 2.0f) + pow(ystart[pos] - ystop[pos], 2.0f) + pow(zstart[pos] - zstop[pos], 2.0f);
				float old_branch_l = pow(xcentroid - xstop[pos], 2.0f) + pow(ycentroid - ystop[pos], 2.0f) + pow(zcentroid - zstop[pos], 2.0f);//length of old branch
				float new_branch_l = pow(point[0] - xcentroid, 2.0f) + pow(point[1] - ycentroid, 2.0f) + pow(point[2] - zcentroid, 2.0f);//length of new branch
				float old_root_l = pow(xstart[pos] - xcentroid, 2.0f) + pow(ystart[pos] - ycentroid, 2.0f) + pow(zstart[pos] - zcentroid, 2.0f);//length of old parent
				old_branch_l = pow(old_branch_l, 0.5f);
				new_branch_l = pow(new_branch_l, 0.5f);
				old_root_l = pow(old_root_l, 0.5f);
				old_length = pow(old_length, 0.5f);
				//sum cost
				distance_cost = -(old_length * pow(Qs[pos], qPower));//starting negative cost since it is removing this branch from the tree
				distance_cost += old_branch_l * wBranch + new_branch_l + old_root_l * wTrunk;
				// Add new flow from perfusion point
				int loc = root[pos];
				float volumnChange = 0;
				while (loc != -1) {
					volumnChange += V[loc];
					loc = root[loc];
				}
				all_costs[pos] = distance_cost + volumnChange;
			}

			//Try best costs up to 1.05*best cost until a viable connection is found
			//Non-viable connections are connections with branches outside of the liver mask or if intersection check is on,
			//branches that cause an intersection
			int pos = 0;
			float Qscale = 1.0f / pow((float)Qs[0], 1.0f / gamma);
			float r_con = scale * Qscale * pow(1.0f, 1.0f / gamma);//radii of connecting branch
			float curBestCost, max_cost = INFINITY;
			//loop through best costs in order of minimal costs
			for (int i = 0; i < 50; i++) {
				pos = getBestCostIndx(all_costs, count);//gets the next minimum cost index
				//save this as the max_cost, allows a +5% increase in cost
				if (i == 0) { max_cost = all_costs[pos] * 1.05f; }
				curBestCost = all_costs[pos];
				all_costs[pos] = INFINITY;
				if (curBestCost > max_cost || pos == -1) { break; }//don't connect if we are larger than max_cost

				//get new centerline locations
				float wTrunk = pow(Qs[pos] + 1.0f, qPower);
				float wBranch = pow(Qs[pos], qPower);
				float norm = 1.0f / (1.0f + wTrunk + wBranch);
				float xcentroid = norm * (point[0] + wTrunk * xstart[pos] + wBranch * xstop[pos]);
				float ycentroid = norm * (point[1] + wTrunk * ystart[pos] + wBranch * ystop[pos]);
				float zcentroid = norm * (point[2] + wTrunk * zstart[pos] + wBranch * zstop[pos]);

				//check if the centerline goes outside the phantom (can happen with concave masks)
				bool outMask = isOutMask(point[0], point[1], point[2], xcentroid, ycentroid, zcentroid, perf, dims); if (outMask) { curBestCost = INFINITY;  continue; }//iterate if a segment is outside
				outMask = isOutMask(xstart[pos], ystart[pos], zstart[pos], xcentroid, ycentroid, zcentroid, perf, dims); if (outMask) { curBestCost = INFINITY; continue; }//iterate if a segment is outside
				outMask = isOutMask(xstop[pos], ystop[pos], zstop[pos], xcentroid, ycentroid, zcentroid, perf, dims); if (outMask) { curBestCost = INFINITY; continue; }//iterate if a segment is outside

#if Intersection_check
				float r_pos1 = scale * Qscale * pow(Qs[pos], 1.0f / gamma);//radii of position branch
				float r_pos2 = scale * Qscale * pow(Qs[pos] + 1, 1.0f / gamma);//radii of position branch
				int sister_branch = -1;//to get the sister_branch location
				//check if there is a parent
				if (root[pos] != -1) {
					sister_branch = children[2 * (root[pos])];
					// check if we have the correct child
					if (sister_branch == pos) {
						sister_branch = children[2 * (root[pos]) + 1];
					}
				}
					// Compute distances and check for intersection for all new segments (three of them)
					volatile bool flag = 0;//flag to tell other threads that we found an intersection
#pragma omp parallel for shared(flag)
					for (int int_check = 0; int_check < count; int_check++) {
						//check if the cost is already at infinity
						//avoid seeing if there is an intersection with the one it is connecting too
						if (flag || pos == int_check) { continue; }

						float d_sc = INFINITY;
						if (root[pos] != int_check && sister_branch != int_check) {
							d_sc = dist3DSegmentToSegmentBinary(xstart[pos], ystart[pos], zstart[pos], xcentroid, ycentroid, zcentroid,
								xstart[int_check], ystart[int_check], zstart[int_check], xstop[int_check], ystop[int_check], zstop[int_check]);
						}

						float d_cs = INFINITY;
						if (children[pos * 2] != int_check && children[pos * 2 + 1] != int_check) {
							d_cs = dist3DSegmentToSegmentBinary(xcentroid, ycentroid, zcentroid, xstop[pos], ystop[pos], zstop[pos],
								xstart[int_check], ystart[int_check], zstart[int_check], xstop[int_check], ystop[int_check], zstop[int_check]);
						}

						float d_cx = dist3DSegmentToSegmentBinary(xcentroid, ycentroid, zcentroid, point[0], point[1], point[2],
							xstart[int_check], ystart[int_check], zstart[int_check], xstop[int_check], ystop[int_check], zstop[int_check]);

						float r_check = scale * Qscale * pow(Qs[int_check], 1.0f / gamma);//radii of possible intersecting branch

						//check if the shortest distance is smaller than the combined radii
						if (d_sc < (r_pos2 + r_check) || d_cs < (r_pos1 + r_check) || d_cx < (r_con + r_check)) {
							curBestCost = INFINITY; // there is an intersection, set cost to infinity
							flag = 1;
						}
					}
#endif

				//check if we can stop searching
				if (curBestCost < INT_MAX) {
					best_idx = pos;
					best_cost = curBestCost;
					break;
				}
			}


		//only update tree if a viable option was found
		if (best_cost < INT_MAX) {
			int pos = best_idx;
			//Update tree
			float wTrunk = pow(Qs[pos] + 1.0f, qPower);
			float wBranch = pow(Qs[pos], qPower);
			float norm = 1.0f / (1.0f + wTrunk + wBranch);
			float xcentroid = norm * (point[0] + wTrunk * xstart[pos] + wBranch * xstop[pos]);
			float ycentroid = norm * (point[1] + wTrunk * ystart[pos] + wBranch * ystop[pos]);
			float zcentroid = norm * (point[2] + wTrunk * zstart[pos] + wBranch * zstop[pos]);

			//update lengths
			float old_branch_l = pow(xcentroid - xstop[pos], 2.0f) + pow(ycentroid - ystop[pos], 2.0f) + pow(zcentroid - zstop[pos], 2.0f);//length of old branch
			float new_branch_l = pow(point[0] - xcentroid, 2.0f) + pow(point[1] - ycentroid, 2.0f) + pow(point[2] - zcentroid, 2.0f);//length of new branch
			float old_root_l = pow(xstart[pos] - xcentroid, 2.0f) + pow(ystart[pos] - ycentroid, 2.0f) + pow(zstart[pos] - zcentroid, 2.0f);//length of old parent
			length[count] = pow(old_branch_l, 0.5f);
			length[count + 1] = pow(new_branch_l, 0.5f);
			length[pos] = pow(old_root_l, 0.5f);

			//update branch: centroid -> old branch endpoint
			xstart[count] = xcentroid;
			ystart[count] = ycentroid;
			zstart[count] = zcentroid;
			xstop[count] = xstop[pos];
			ystop[count] = ystop[pos];
			zstop[count] = zstop[pos];

			//update branch: centroid -> new point	
			xstart[count + 1] = xcentroid;
			ystart[count + 1] = ycentroid;
			zstart[count + 1] = zcentroid;
			xstop[count + 1] = point[0];
			ystop[count + 1] = point[1];
			zstop[count + 1] = point[2];

			//update branch: old branch startpoint -> centroid
			xstop[pos] = xcentroid;
			ystop[pos] = ycentroid;
			zstop[pos] = zcentroid;

			//update branch: centroid -> old branch endpoint
			Qs[count] = Qs[pos];
			root[count] = pos;
			children[count * 2] = children[pos * 2];
			children[(count * 2) + 1] = children[(pos * 2) + 1];
			section[count] = section[pos];

			// Update Parent for Downstream Branches
			if (children[2 * pos] != -1) {//make sure daughters exist
				root[children[2 * pos]] = count;
				root[children[(2 * pos) + 1]] = count;
			}

			//update branch: centroid -> new point			
			Qs[count + 1] = 1;
			root[count + 1] = pos;
			children[2 * (count + 1)] = -1;
			children[(2 * (count + 1)) + 1] = -1;
			section[count + 1] = sctn;

			//update branch: old branch startpoint -> centroid
			children[2 * pos] = count;
			children[(2 * pos) + 1] = count + 1;

			// Add new flow from perfusion point
			while (pos != -1) {
				Qs[pos] += 1;
				V[pos] = length[pos] * (pow(Qs[pos] + 1, qPower) - pow(Qs[pos], qPower));//update potential volumn costs
				pos = root[pos];
			}

			//Update Volumns
			V[count] = length[count] * (pow(Qs[count] + 1, qPower) - pow(Qs[count], qPower));
			V[count + 1] = length[count + 1] * (pow(Qs[count + 1] + 1, qPower) - pow(Qs[count + 1], qPower));

			//increment
			count += 2;
		}
		//this point could not be connected to the tree, put it back
		else if (point[3] < max_tries) {
			switch (sctn) {
			case 1:	region1.push(point); break;
			case 2: region2.push(point); break;
			case 3: region3.push(point); break;
			case 4: region4.push(point); break;
			case 5: region5.push(point); break;
			case 6: region6.push(point); break;
			case 7: region7.push(point); break;
			case 8: region8.push(point); break;
			}
			if (count < 15) { region--; }//only redo this section if we are forming the intial tree
			offset++;
		}
		else { removed++; }//remove this point

		//check if the desired number of endpoints have been connected
		if (count == ((terminal_pts * 2) + 1)) {
			*count_pointer = count;
			return;
		}
	}
	// Terminal Pts
	*count_pointer = count;
	return;


}


/*
Random Hepatic Tree Generation within Blood Demand Map
// Nx, Ny, Nz: x,y,z dimension of blood demand map
// terminal_pts: number of terminal points for tree
// scale: what to scale the resolution by
// seed: PSRNG seed
//	outResolutionScaleFactor: factor to scale the input resolution by
*/
int  Liver::build_tree(const int Nx, const int Ny, const int Nz, int terminal_pts, int tree_number,
	float scale, unsigned int seed, float outResolutionScaleFactor) {

	//set PRNG
	mt19937 mt;
	mt.seed(seed);
	double mxInt = mt.max();


	unsigned __int8 *perf = new unsigned __int8[10 * Nz*Ny*Nx];
	string fname2 = "../Vessel_sim_CPU/BloodDemandMap/Phantom.dat";
	FILE *fid; 
	errno_t err;
	if ((err = fopen_s(&fid, fname2.c_str(), "rb")) != 0){
		cerr << "Could not open File: " << fname2 << endl;
		cerr << "Exiting..." << endl;
		exit(1);
	}
	else {
		size_t bytesRead = fread(perf, sizeof(unsigned __int8), 10 * Nx*Nz*Ny, fid);
		//test if we read enough data
		if (bytesRead != 10 * Nx*Nz*Ny * sizeof(unsigned __int8)) {
			cerr << "Only read: " << bytesRead << " of " << 10 * Nx*Nz*Ny * sizeof(unsigned __int8) << endl;
			cerr << "Exiting...." << endl;
			exit(1);
		}
		fclose(fid);
	}



	//initialize variables 
	unsigned int dims[4] = { Nx, Ny, Nz, 10 };
	int seg_number = 7;//number of liver segments (this is 8 since c++ start counting at 0)
	float gamma = 2.8f + 0.3f * (mt() / mxInt);//for Murray's law
	int count = 0;//total number of branches formed
	float factor = 1.2f;//account for pushing of branches
	int arrayAlloc = ceil(terminal_pts * 2 * factor) + 1;
	float *xstart = new float[arrayAlloc];
	float *xstop = new float[arrayAlloc];
	float *ystart = new float[arrayAlloc];
	float *ystop = new float[arrayAlloc];
	float *zstart = new float[arrayAlloc];
	float *zstop = new float[arrayAlloc];
	int *children = new int[2 * arrayAlloc];
	int *root = new int[arrayAlloc];
	int *section = new int[arrayAlloc];
	float *length = new float[arrayAlloc];
	int *Qs = new int[arrayAlloc];
	float *r = new float[arrayAlloc];
	queue<array<float, 4>> region1, region2, region3, region4, region5, region6, region7, region8;//Queues for the different regions

	

	// Find end/start point of first branch (region one excluded from stop point)
	float k, j, i, val;
	int intialPtSections[2] = { 8, 3 };//segments to choose the initial points
	int index = 0;
	for (int pos = 0; pos < 2; pos++) {
		int found = 0;
		while (found == 0) {
			k = (mt() / mxInt) * Nz - 0.5f;
			j = (mt() / mxInt) * Ny - 0.5f;
			i = (mt() / mxInt) * Nx - 0.5f;
			val = 254.5f * (mt() / mxInt);
			getLinearInd(i, j, k, intialPtSections[pos], dims, &index);
			if (val < perf[index]) {
				if (pos == 0) {
					xstart[pos] = i;
					ystart[pos] = j;
					zstart[pos] = k;
					perf[index] = -1;// dont allow multiple selections of the same point
				}
				else {
					xstop[pos - 1] = i;
					ystop[pos - 1] = j;
					zstop[pos - 1] = k;
					perf[index] = -1;// dont allow multiple selections of the same point
				}
				//perf((int)i, (int)j, (int)k, 10) = -1;// dont allow multiple selections of the same point
				found = 1;
			}
		}
	}
	// Find start point for each segment 
	if (terminal_pts < seg_number) { seg_number = terminal_pts; }//make sure we have enough endpoints for at least an endpoint in each segment
	//loop through to find an endpoint for each segment
	for (int pos = 0; pos < seg_number; pos++) {
		int found = 0;
		while (found == 0) {
			k = (mt() / mxInt) * Nz - 0.5f;
			j = (mt() / mxInt) * Ny - 0.5f;
			i = (mt() / mxInt) * Nx - 0.5f;
			val = 254.5f * (mt() / mxInt);
			getLinearInd(i, j, k, pos, dims, &index);
			if (val < perf[index]) {
				array<float, 4> point = { i, j, k, 0 };
				switch (pos + 1) {
				case 1: region1.push(point); break;
				case 2: region2.push(point); break;
				case 3: region3.push(point); break;
				case 4: region4.push(point); break;
				case 5: region5.push(point); break;
				case 6: region6.push(point); break;
				case 7: region7.push(point); break;
				case 8: region8.push(point); break;
				}
				//perf((int)i, (int)j, (int)k, 10) = -1; //dont allow the selection of the same point
				found = 1;
			}
		}
	}

	// Find rest of endpoints to connect
	cout << "Building Tree " << tree_number << " with " << terminal_pts << " points" << endl;
	int region;
	for (int pos = seg_number; pos < terminal_pts; pos++) {
		int found = 0;
		while (found == 0) {
			k = (mt() / mxInt) * Nz - 0.5f;
			j = (mt() / mxInt) * Ny - 0.5f;
			i = (mt() / mxInt) * Nx - 0.5f;
			val = 254.5f * (mt() / mxInt);
			getLinearInd(i, j, k, dims[3]-1, dims, &index);
			if (val < perf[index]) {
				array<float, 4> point = { i, j, k, 0 };
				region = get_section(i, j, k, perf, mt, dims);
				if (region == 0) {
					pos -= 1;
				}
				switch (region) {
				case 1: region1.push(point); break;
				case 2: region2.push(point); break;
				case 3: region3.push(point); break;
				case 4: region4.push(point); break;
				case 5: region5.push(point); break;
				case 6: region6.push(point); break;
				case 7: region7.push(point); break;
				case 8: region8.push(point); break;
				}
				//perf[index] = -1; //dont allow the selection of the same point
				found = 1;
			}
		}
	}

	/* 
	*
	*    Arterial Tree
	*
	*/ 
	create_tree(xstart, ystart, zstart, xstop, ystop, zstop, root, section, length, Qs, region1,
		region2, region3, region4, region5, region6, region7, region8, gamma, terminal_pts, children, &count, scale,
		Nx, Ny, Nz, perf, mt, arrayAlloc, dims);


	//get radii of all branches
	float Qscale = 1.0f / pow((float)Qs[0], 1.0f / gamma);// Max Perfusion at Qs[0] (this is the root)
	for (unsigned int t = 0; t < count; t++) {
		r[t] = Qscale * pow(Qs[t], 1.0f / gamma) * scale;
	}

	int prePushCount = count;
#if Intersection_check
	//Correct for any branches that cause an intersection due to growth of the vessel
	int check = push_branches(xstart, ystart, zstart, xstop, ystop, zstop, root, Qs, section, &count, Qscale, gamma, scale, terminal_pts, children, r);
	//check if we got stuck in a loop
	if (check != 0) {
		return -1;
	}
#endif
	
	/* 
	*
	*    Centerline start/stop points
	*
	*
	float(*CLines)[6][2] = new float[count][6][2];//array to hold all centerline tree info
	for (unsigned int t = 0; t < count; t++) {
		// Radius & profusion
		CLines[t][3][0] = r[t];
		CLines[t][3][1] = Qs[t];
		// Parent
		CLines[t][4][0] = root[t] + 1;//add one for matlab indexing
		CLines[t][4][1] = root[t] + 1;
		// Children 
		CLines[t][5][0] = children[2 * t] + 1;//add one for matlab indexing
		CLines[t][5][1] = children[(2 * t) + 1] + 1;
		// Vessels
		CLines[t][0][0] = xstart[t] + 1;//add one for matlab indexing
		CLines[t][1][0] = ystart[t] + 1;
		CLines[t][2][0] = zstart[t] + 1;
		CLines[t][0][1] = xstop[t] + 1;
		CLines[t][1][1] = ystop[t] + 1;
		CLines[t][2][1] = zstop[t] + 1;
	}

	// Arterial tree
	cout << "Writing to files..." << endl;
	FILE* file;
	string dir1 = "../Vessel_sim_GPU/CLines/CLines" + to_string(tree_number) + ".dat";
	fopen_s(&file, dir1.c_str(), "wb");
	fwrite(CLines, sizeof(float), 2 * 6 * count, file);
	fclose(file);
	*/

	/*
	*
	*   Polynomial Interpolation
	*
	*/
	vector <xyz> *interpBranches = new vector <xyz>[count];//holds x,y,z points for all interpolated branches
	optimizeAngles(xstart, ystart, zstart, xstop, ystop, zstop, root,
		Qs, &count, gamma, scale, children, r, prePushCount, interpBranches, outResolutionScaleFactor);


	/*
	*
	*   write out data
	*
	*/
	//writing vessel tree to txt file, see readme on data format
	string dir2 = "../Vessel_sim_CPU/CLines/Tree" + to_string(tree_number) + ".txt";
	ofstream tree;
	tree.open(dir2);
	for (unsigned int t = 0; t < prePushCount; t++) {
		for (vector<xyz>::iterator i = interpBranches[t].begin(); i != interpBranches[t].end(); i++) {
			tree << (*i).x << ' ' << (*i).y << ' ' << (*i).z << ' ';
		}
		tree << endl;
	}
	tree.close();

	//writing vessel radi to txt file, see readme on data format
	dir2 = "../Vessel_sim_CPU/CLines/Radii" + to_string(tree_number) + ".txt";
	ofstream rad;
	rad.open(dir2);
	for (unsigned int t = 0; t < prePushCount; t++) {
		rad << r[t] << endl;
	}
	rad.close();
	
	delete[] xstart, xstop, ystart, ystop, zstart, zstop, length, Qs, children, section, root;
	return 0;
}


//function to find the section the point is in (if multile section overlap, samples the probabilities to find which one)
// x, y, z: chosen point
// perf: blood demand map
// mt: PRNG
// return: Counaud section the point is in
int Liver::get_section(int x, int y, int z, unsigned __int8 perf[], mt19937 mt,unsigned int dims[4]) {
	int regions[8] = { 0 };
	float probability[8] = { 0 };
	int count = 7;
	int index = 0;
	//loop through all section and get propability of being in each one
	for (int i = 0; i < 8; i++) {
		getLinearInd(x, y, z, i, dims, &index);
		if (perf[index] != 0) {
			regions[count] = i + 1;//add one since section index starts at one
			probability[count] = perf[index];
			count--;
		}
	}
	//make probability distribution
	for (int i = 0; i < 7; i++) {
		probability[i + 1] = probability[i + 1] + probability[i];
	}
	//check if only in one region
	if (probability[6] == 0) { return(regions[7]); }
	//normalize distribution
	if (probability[7] != 1) {
		for (int i = 0; i < 8; i++) {
			probability[i] = probability[i] / probability[7];
		}
	}
	// if point is in multiple sections, role for section
	float prob = 1.0f * mt() / mt.max();
	if (prob == 1) { return(regions[7]); }//handle edge case
	if (prob == 0) { return(regions[7]); }//handle edge case
	//put all sections into array, starting with the chosen section
	for (int i = 0; i < 7; i++) {
		if (i < 7) {
			if (probability[i] <= prob && probability[i + 1] > prob) {
				return(regions[i + 1]);
			}
		}
	}
}



// dist3D_Segment_to_Segment(): get the 3D minimum distance between 2 segments
//	  From : Edited with permission from https://geomalgorithms.com/a07-_distance.html
//    Input: four points: s1: start of segment one
//						  e1: end of segment one
//						  s2: start of segment two
//						  e2: end of segment two
//	  P1: vector to store the closest point on segment one
//	  P2: vector to store the closest point on segment two
//    Return: the shortest distance between two branches i, j
float Liver::dist3D_Segment_to_Segment(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z, float s2x, float s2y, float s2z, float e2x, float e2y, float e2z, float P1[3], float P2[3]) {
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

	//get the two closest points
	P1[0] = s1x + sc * u[0];//closest point on branch one
	P1[1] = s1y + sc * u[1];
	P1[2] = s1z + sc * u[2];
	P2[0] = s2x + tc * v[0];//closest point on branch two
	P2[1] = s2y + tc * v[1];
	P2[2] = s2z + tc * v[2];

	float distance = pow(pow(dP[0], 2.0f) + pow(dP[1], 2.0f) + pow(dP[2], 2.0f), 0.5f);
	return distance;   // return the closest distance
}


// dist3D_Segment_to_Segment_Binary(): get the 3D minimum distance between 2 segments
//	  From : adapated with permission from: https://geomalgorithms.com/a07-_distance.html
//    Input: four points: s1: start of segment one
//						  e1: end of segment one
//						  s2: start of segment two
//						  e2: end of segment two
//    Return: the shortest distance between two branches i, j
float Liver::dist3DSegmentToSegmentBinary(float s1x, float s1y, float s1z, float e1x, float e1y, float e1z, float s2x, float s2y, float s2z, float e2x, float e2y, float e2z) {
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





//Function to move the small branches out of the way if they cause an intersection on the CPU
// Input: Tree object with arrays
//	x,y,z start/stop: x,y,z start/stop points of all branches
//	root: list of all parents for each branch
//	Qs: perfusion of each branch
//	section: section each branch is in
//	count: total number of branches in tree 
//	Qscale: normalize the perfusion
//	gamma: gamma for Murrays Law
//	scale: what to scale the resolution by
//	terminal_pts: total number of terminal_pts
//	children: array of the children for each branch
//	r: radius of each branch
// Output: Tree with branches pushed to not cause intersection
int Liver::push_branches(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[], float zstop[], int root[],
	int Qs[], int section[], int *count, float Qscale, float gamma, float scale, int terminal_pts, int children[], float r[]) {
	cout << "Fixing any intersection that may have occured due to growth..." << endl;
	queue<int> recheck;//branches to recheck
	int max_indx = *count;
	for (int j = 0; j < max_indx; j++) {
		if (j % 5000 == 0) {
			cout << "Checking branch: " << j << " of " << max_indx << endl;
		}
		//Get Sister if she exists
		int sister = -1;
		if (root[j] != -1) {
			sister = children[2 * (root[j])];
			//Check if we have the right child
			if (sister == j) {
				sister = children[(2 * root[j]) + 1];
			}
		}
		float r_j = r[j];//radius of j branch
#if !constant_seed //this will have non-consisent behavior if the same seed point is desired
#pragma omp parallel for
#endif
		for (int k = (j + 1); k < max_indx; k++) {
			//skip parents, children, same branch, and sister
			if (children[2 * j] != k && children[(2 * j) + 1] != k && root[j] != k && sister != k) {
				float r_k = r[k];//radius of k branch
				float c1[3], c2[3];//closest point on branch one
				float d = dist3D_Segment_to_Segment(xstart[j], ystart[j], zstart[j], xstop[j], ystop[j], zstop[j],
					xstart[k], ystart[k], zstart[k], xstop[k], ystop[k], zstop[k], c1, c2);
				if (d == 0) { cerr << "Error line: 1216, possible division by 0." << endl; }
				//check for intersection
				if (d < (r_k + r_j)) {
					//check if the intersection occurs on the same main branch by traveling up the k branch
					int parent2 = root[k];
					while (parent2 != j && parent2 != -1) {
						parent2 = root[parent2];
					}
					if (parent2 == -1) {
						//check if the intersection occurs on the same main branch by traveling up the j branch
						int parent1 = root[j];
						while (parent1 != k && parent1 != -1) {
							parent1 = root[parent1];
						}
						//They were on different branches
						if (parent1 == -1) {
							//cout << "one : " << k << ", two: " << j << endl;
#if !constant_seed
#pragma omp critical//make sure this area is only ran one thread at a time
#endif
							{
								//Determine which branch is smaller so we can move it
								float closest[3], mov_vec[3];//which closest point we are moving
								int indx;//branch indx that we are moving
								//move j branch
								if (r_j < r_k) {
									for (int ind = 0; ind < 3; ind++) {
										closest[ind] = c1[ind];
										mov_vec[ind] = (c1[ind] - c2[ind]) * (.001 + r_j + r_k) / d;//.005 is a small safety factor for rounding errors
									}
									indx = j;
								}
								//move k branch
								else {
									for (int ind = 0; ind < 3; ind++) {
										closest[ind] = c2[ind];
										mov_vec[ind] = (c2[ind] - c1[ind]) * (.001 + r_j + r_k) / d;//.005 is a small safety factor for rounding errors
									}
									indx = k;
								}
								//See if we are on the start of the branch
								int parent = root[indx];
								if (isequal(closest[0], closest[1], closest[2], xstart[indx], ystart[indx], zstart[indx])) {
									//move end of parent and start of this branch
									xstop[parent] = closest[0] + mov_vec[0];
									ystop[parent] = closest[1] + mov_vec[1];
									zstop[parent] = closest[2] + mov_vec[2];

									xstart[indx] = closest[0] + mov_vec[0];
									ystart[indx] = closest[1] + mov_vec[1];
									zstart[indx] = closest[2] + mov_vec[2];
									//move start of sister as well
									int sister;
									if (children[2 * parent] != indx) {
										sister = children[2 * parent];
									}
									else {
										//other sister
										sister = children[(2 * parent) + 1];
									}
									xstart[sister] = closest[0] + mov_vec[0];
									ystart[sister] = closest[1] + mov_vec[1];
									zstart[sister] = closest[2] + mov_vec[2];
									//re-check parent
									recheck.push(parent);
									//re-check daughter
									recheck.push(sister);
									recheck.push(indx);
								}
								else if (isequal(closest[0], closest[1], closest[2], xstop[indx], ystop[indx], zstop[indx])) {
									//see if we are on the end of the branch
									//move start of daughters if they exist
									int child1 = children[2 * indx];
									int child2 = children[(2 * indx) + 1];
									if (child1 != -1) {
										xstart[child1] = closest[0] + mov_vec[0];
										ystart[child1] = closest[1] + mov_vec[1];
										zstart[child1] = closest[2] + mov_vec[2];

										xstart[child2] = closest[0] + mov_vec[0];
										ystart[child2] = closest[1] + mov_vec[1];
										zstart[child2] = closest[2] + mov_vec[2];
										//recheck daughters
										recheck.push(child1);
										recheck.push(child2);
									}
									//move end of this branch
									xstop[indx] = closest[0] + mov_vec[0];
									ystop[indx] = closest[1] + mov_vec[1];
									zstop[indx] = closest[2] + mov_vec[2];
									recheck.push(indx);
								}
								else {
									//else we are inbetween, just add a branch inbetween then
									//new branch
									xstart[*count] = closest[0] + mov_vec[0];
									ystart[*count] = closest[1] + mov_vec[1];
									zstart[*count] = closest[2] + mov_vec[2];
									xstop[*count] = xstop[indx];
									ystop[*count] = ystop[indx];
									zstop[*count] = zstop[indx];
									r[*count] = r[indx];
									children[2 * (*count)] = children[2 * indx];
									children[(2 * (*count)) + 1] = children[(2 * indx) + 1];
									root[*count] = indx;
									Qs[*count] = Qs[indx];
									section[*count] = section[indx];
									recheck.push(*count);//recheck both new branches
									recheck.push(indx);//recheck both new branches
									//move endpoint of old branch
									xstop[indx] = closest[0] + mov_vec[0];
									ystop[indx] = closest[1] + mov_vec[1];
									zstop[indx] = closest[2] + mov_vec[2];
									//tell daughters who the new parent is
									if (children[2 * (*count)] != -1) {
										root[children[2 * (*count)]] = *count;
										root[children[(2 * (*count)) + 1]] = *count;
									}
									//Update indx children
									children[2 * indx] = *count;
									children[(2 * indx) + 1] = *count;
									*count += +1;//new branch added
								}
							}
#if !constant_seed
#pragma omp flush(xstart, ystart, zstart, xstop, ystop, Qs, children, root, section) // make sure all thread have an updated view of tree variables
#endif
						}
					}
				}
			}
		}
	}
	//Due the rechecks
	int maxCheck = recheck.size() * 10;//breaks out if the recheck is stuck in a loop
	while (!recheck.empty()) {
		if (recheck.size() > maxCheck) { return -1; }
		int j = recheck.front();
		recheck.pop();
		//Get Sister if she exists
		int sister = -1;
		if (root[j] != -1) {
			sister = children[2 * root[j]];
			//Check if we have the right child
			if (sister == j) {
				sister = children[(2 * root[j]) + 1];
			}
		}
		//check all branches against this 
		float r_j = r[j];//radius of j branch
		max_indx = *count;
		for (int k = 0; k < max_indx; k++) {
			//skip parents, children, same branch, and sister
			if (j != k && children[2 * j] != k && children[(2 * j) + 1] != k && root[j] != k && sister != k) {
				float r_k = r[k];//radius of k branch
				float c1[3], c2[3];//closest points on branch one and two
				float d = dist3D_Segment_to_Segment(xstart[j], ystart[j], zstart[j], xstop[j], ystop[j], zstop[j],
					xstart[k], ystart[k], zstart[k], xstop[k], ystop[k], zstop[k], c1, c2);
				if (d == 0) { cout << "Error line: 1317, possible division by 0." << endl; }
				//check for intersection
				if (d < (r_k + r_j)) {
					//check if the intersection occurs on the same main branch by traveling up the k branch
					int parent2 = root[k];
					while (parent2 != j && parent2 != -1) {
						parent2 = root[parent2];
					}
					if (parent2 == -1) {
						//check if the intersection occurs on the same main branch by traveling up the j branch
						int parent1 = root[j];
						while (parent1 != k && parent1 != -1) {
							parent1 = root[parent1];
						}
						//They were on different branches
						if (parent1 == -1) {
							//Determine which branch is smaller so we can move it
							float closest[3], mov_vec[3];//which closest point we are moving and vector to move closest point
							int indx;//branch indx that we are moving
							//move j branch
							if (r_j < r_k) {
								for (int ind = 0; ind < 3; ind++) {
									closest[ind] = c1[ind];
									mov_vec[ind] = (c1[ind] - c2[ind]) * (.001 + r_j + r_k) / d;//.005 is a small safety factor for rounding errors
								}
								indx = j;
							}
							//move k branch
							else {
								for (int ind = 0; ind < 3; ind++) {
									closest[ind] = c2[ind];
									mov_vec[ind] = (c2[ind] - c1[ind]) * (.001 + r_j + r_k) / d;//.005 is a small safety factor for rounding errors
								}
								indx = k;
							}
							//recheck both branches
							recheck.push(j);
							recheck.push(k);
							//See if we are on the start of the branch
							int parent = root[indx];
							if (isequal(closest[0], closest[1], closest[2], xstart[indx], ystart[indx], zstart[indx])) {
								//move end of parent and start of this branch
								xstop[parent] = closest[0] + mov_vec[0];
								ystop[parent] = closest[1] + mov_vec[1];
								zstop[parent] = closest[2] + mov_vec[2];

								xstart[indx] = closest[0] + mov_vec[0];
								ystart[indx] = closest[1] + mov_vec[1];
								zstart[indx] = closest[2] + mov_vec[2];
								//move start of sister as well
								int sister = -1;
								if (children[2 * parent] != indx) {
									sister = children[2 * parent];
								}
								else if (children[(2 * parent + 1)] != indx) {
									//other sister
									sister = children[(2 * parent) + 1];
								}
								//if sister exists
								if (sister != -1) {
									xstart[sister] = closest[0] + mov_vec[0];
									ystart[sister] = closest[1] + mov_vec[1];
									zstart[sister] = closest[2] + mov_vec[2];
									recheck.push(sister);
								}
								recheck.push(parent);
							}
							else if (isequal(closest[0], closest[1], closest[2], xstop[indx], ystop[indx], zstop[indx])) {
								//see if we are on the end of the branch
								//move start of daughters if they exist
								int child1 = children[2 * indx];
								int child2 = children[(2 * indx) + 1];
								if (child1 != -1) {
									xstart[child1] = closest[0] + mov_vec[0];
									ystart[child1] = closest[1] + mov_vec[1];
									zstart[child1] = closest[2] + mov_vec[2];

									xstart[child2] = closest[0] + mov_vec[0];
									ystart[child2] = closest[1] + mov_vec[1];
									zstart[child2] = closest[2] + mov_vec[2];

									//recheck children
									recheck.push(child1);
									recheck.push(child2);
								}
								//move end of this branch
								xstop[indx] = closest[0] + mov_vec[0];
								ystop[indx] = closest[1] + mov_vec[1];
								zstop[indx] = closest[2] + mov_vec[2];
							}
							else {
								//else we are inbetween, just add a branch inbetween then
								//new branch
								xstart[*count] = closest[0] + mov_vec[0];
								ystart[*count] = closest[1] + mov_vec[1];
								zstart[*count] = closest[2] + mov_vec[2];
								xstop[*count] = xstop[indx];
								ystop[*count] = ystop[indx];
								zstop[*count] = zstop[indx];
								r[*count] = r[indx];
								children[2 * (*count)] = children[2 * indx];
								children[(2 * (*count)) + 1] = children[(2 * indx) + 1];
								root[*count] = indx;
								Qs[*count] = Qs[indx];
								section[*count] = section[indx];
								recheck.push(*count);//recheck new branch

								//move endpoint of old branch
								xstop[indx] = closest[0] + mov_vec[0];
								ystop[indx] = closest[1] + mov_vec[1];
								zstop[indx] = closest[2] + mov_vec[2];
								//tell daughters who the new parent is
								if (children[2 * (*count)] != -1) {
									root[children[2 * (*count)]] = *count;
									root[children[(2 * (*count)) + 1]] = *count;
								}
								//Update indx children
								children[2 * indx] = *count;
								children[(2 * indx) + 1] = *count;
								*count += 1;//new branch added
							}
							break;//skip remaining computations
						}
					}
				}
			}
		}
	}
	return 0;
}



//Function to determine array equality
// sx, sy, sz: array one
// ex, ey, ez: array two
//returns: 1 if arrays are equal, 0 if not
bool Liver::isequal(float sx, float sy, float sz, float ex, float ey, float ez) {
	//avoid floating point equality issues
	sx = round(sx * 100);
	sy = round(sy * 100);
	sz = round(sz * 100);
	ex = round(ex * 100);
	ey = round(ey * 100);
	ez = round(ez * 100);
	if (sx == ex && sy == ey && sz == ez) {
		return(1);
	}
	else {
		return(0);
	}
}


//Function to check if a centerline goes outside of the mask
// x1, y1, z1:start point of the branch to be tested
// x2, y2, z2:end point of the branch to be tested
//return: 1 if any point is outside of mask, else 0
bool Liver::isOutMask(float x1, float y1, float z1, float x2, float y2, float z2, unsigned __int8 perf[],unsigned int dims[4]) {
	//check end cases first
	int index;
	getLinearInd((int)x1, (int)y1, (int)z1, dims[3]-1, dims, &index);
	if (perf[index] == 0) { return(1); }
	getLinearInd((int)x2, (int)y2, (int)z2, dims[3]-1, dims, &index);
	if (perf[index] == 0) { return(1); }

	//check along the centerline
	float xvec = x2 - x1;
	float yvec = y2 - y1;
	float zvec = z2 - z1;
	float length = pow(pow(xvec, 2.0f) + pow(yvec, 2.0f) + pow(zvec, 2.0f), 0.5f);//length of old branch
	int maxIt = ceil(length);
	float tempx, tempy, tempz;
	//iterate through centerline with stepsize equal to voxel resolution
	for (int i = 0; i < maxIt; i++) {
		tempx = x1 + xvec * ((float)i / (float)maxIt);
		tempy = y1 + yvec * ((float)i / (float)maxIt);
		tempz = z1 + zvec * ((float)i / (float)maxIt);
		getLinearInd((int)tempx, (int)tempy, (int)tempz, dims[3]-1, dims, &index);
		if (perf[index] == 0) {
			return(1);
		}
	}
	return(0);//all inside the mask
}



//Function to return the index of best cost 
//all_cost: array of all costs
//count: total number of branches in the tree
int Liver::getBestCostIndx( float all_costs[], int count) {
	int best_idx = -1;
	float best_cost = INFINITY;
	//loop through all costs and get the next best cost
	for (int pos = 0; pos < count; pos++) {
		if ((all_costs[pos] < best_cost)) {
			best_idx = pos;
			best_cost = all_costs[pos];
		}
	}
	return(best_idx);
}





//Function to move the small branches out of the way if they cause an intersection
// Input: Tree object with arrays
//	x,y,z start/stop: x,y,z start/stop points of all branches
//	root: list of all parents for each branch
//	Qs: perfusion of each branch
//	count: total number of branches in tree 
//	scale: what to scale the resolution by
//	terminal_pts: total number of terminal_pts
//	children: array of the children for each branch
//	r: radius of each branch
//	prePushCount: number of branches prior to being pushed
//	outResolutionScaleFactor: factor to scale the input resolution by
//	
// Output: interpBranches, tree with branches pushed to not cause intersection
void Liver::optimizeAngles(float xstart[], float ystart[], float zstart[], float xstop[], float ystop[], float zstop[], int root[],
	int Qs[], int *count, float gamma, float scale, int children[], float r[], int prePushCount, vector <xyz> *interpBranches, float outResolutionScaleFactor) {

	//Initialize Arrays
	float *angles = new float[*count];//array to hold the ideal angle (Murrays law) for each bifurcation
	float *linearPointX = new float[*count];//array to hold the X point at which we should use a linear interpolation until
	float *linearPointY = new float[*count];//array to hold the Y point at which we should use a linear interpolation until
	float *linearPointZ = new float[*count];//array to hold the Z point at which we should use a linear interpolation unti
	float *startSlopeX = new float[*count];//array to hold the X starting slope for polynomial interp
	float *startSlopeY = new float[*count];//array to hold the Y starting slope for polynomial interp
	float *startSlopeZ = new float[*count];//array to hold the Z starting slope for polynomial interp
	float *endSlopeX = new float[*count];//array to hold the X ending slope for polynomial interp
	float *endSlopeY = new float[*count];//array to hold the Y ending slope for polynomial interp
	float *endSlopeZ = new float[*count];//array to hold the Z ending slope for polynomial interp
	vector <xyz> *controlPts = new vector <xyz>[*count];//holds x,y,z control point for split branches

	///////////////////////////
	//Get the optimal angles///
	///////////////////////////
	for (int i = 1; i < *count; i++) {//skip root
		float rParent = r[root[i]];
		float r1 = r[i];
		angles[i] = acos((pow(rParent, 4.0f) + pow(r1, 4.0f) - pow(pow(rParent, gamma) - pow(r1, gamma), 4.0f / gamma)) / (2.0f * pow(rParent, 2.0f) * pow(r1, 2.0f)));
	}

	///////////////////////////////////////////
	//Get the starting slopes of each branch///
	//////////////////////////////////////////

	//initialize root
	startSlopeX[0] = 0;
	startSlopeY[0] = 0;
	startSlopeZ[0] = 0;
	linearPointX[0] = -1;
#pragma omp parallel for
	for (int i = 1; i < *count; i++) {//skip root
		float parentVec[3], childVec[3], normVec[3], sisterVec[3], parentProjVec[3], rotVec[3];//Initialize vectors 
		int parent = root[i];
		int sister = children[2 * parent];
		if (sister == i) { sister = children[(2 * parent) + 1]; }//Check if we have the right child
		if (sister == i) { continue; }//we have a pushed branch
		if (children[2 * i] == -1) {//we have an endpoint
			endSlopeX[i] = 0;
			endSlopeY[i] = 0;
			endSlopeZ[i] = 0;
		}
		//Generate Initial Vecotors
		parentVec[0] = xstop[parent] - xstart[parent];
		parentVec[1] = ystop[parent] - ystart[parent];
		parentVec[2] = zstop[parent] - zstart[parent];

		childVec[0] = xstop[i] - xstart[i];
		childVec[1] = ystop[i] - ystart[i];
		childVec[2] = zstop[i] - zstart[i];

		sisterVec[0] = xstop[sister] - xstart[sister];
		sisterVec[1] = ystop[sister] - ystart[sister];
		sisterVec[2] = zstop[sister] - zstart[sister];
		//cross product of children branches sister -> daugther
		normVec[0] = sisterVec[1] * childVec[2] - sisterVec[2] * childVec[1];
		normVec[1] = sisterVec[2] * childVec[0] - sisterVec[0] * childVec[2];
		normVec[2] = sisterVec[0] * childVec[1] - sisterVec[1] * childVec[0];
		float length = pow(pow(normVec[0], 2.0f) + pow(normVec[1], 2.0f) + pow(normVec[2], 2.0f), 0.5f);
		normVec[0] /= length;
		normVec[1] /= length;
		normVec[2] /= length;

		//projected parent into the plane defined by children
		float dotParentNorm = 0;
		for (int j = 0; j < 3; j++) {
			dotParentNorm += parentVec[j] * normVec[j];
		}
		for (int j = 0; j < 3; j++) {
			parentProjVec[j] = parentVec[j] - normVec[j] * dotParentNorm;
		}
		//need to rotate parent about norm by murray angle(Rodriguess' rotation formula)
		float crossNormParent[3];
		crossNormParent[0] = normVec[1] * parentVec[2] - normVec[2] * parentVec[1];
		crossNormParent[1] = normVec[2] * parentVec[0] - normVec[0] * parentVec[2];
		crossNormParent[2] = normVec[0] * parentVec[1] - normVec[1] * parentVec[0];
		for (int j = 0; j < 3; j++) {
			rotVec[j] = parentProjVec[j] * cos(angles[i]) + crossNormParent[j] * sin(angles[i]) + normVec[j] * dotParentNorm * (1 - cos(angles[i]));
		}
		float lengthRotVec = pow(pow(rotVec[0], 2.0f) + pow(rotVec[1], 2.0f) + pow(rotVec[2], 2.0f), 0.5f);
		//calculate the point to form this angle when the branches are seperated by 1 * combined radii
		float seperateDist = (r[i] + r[sister]) * 1.01f;//small saftey factor
		float dist = seperateDist / pow(2.0f - 2.0f * cos(angles[i] + angles[sister]), 0.5f);
		float factor = 1.0f;
		//check if the distance is farther than the half length of the branches
		float childLength = pow(pow(childVec[0], 2.0f) + pow(childVec[1], 2.0f) + pow(childVec[2], 2.0f), 0.5f);
		if (dist > childLength / 2.0f) {
			dist = childLength / 2.0f;
			factor = 2.0f;
		}
		linearPointX[i] = xstop[parent] + (rotVec[0] / lengthRotVec) * dist;
		linearPointY[i] = ystop[parent] + (rotVec[1] / lengthRotVec) * dist;
		linearPointZ[i] = zstop[parent] + (rotVec[2] / lengthRotVec) * dist;

		//Compute start/end slopes scaled by vector length
		float lengthParentProjVec = pow(pow(parentProjVec[0], 2.0f) + pow(parentProjVec[1], 2.0f) + pow(parentProjVec[2], 2.0f), 0.5f);
		float lengthParentVec = pow(pow(parentVec[0], 2.0f) + pow(parentVec[1], 2.0f) + pow(parentVec[2], 2.0f), 0.5f);
		//starting slopes
		startSlopeX[i] = (rotVec[0] / lengthRotVec) * childLength * factor;
		startSlopeY[i] = (rotVec[1] / lengthRotVec) * childLength * factor;
		startSlopeZ[i] = (rotVec[2] / lengthRotVec) * childLength * factor;
		//ending slopes
		endSlopeX[parent] = (parentProjVec[0] / lengthParentProjVec) * lengthParentVec;
		endSlopeY[parent] = (parentProjVec[1] / lengthParentProjVec) * lengthParentVec;
		endSlopeZ[parent] = (parentProjVec[2] / lengthParentProjVec) * lengthParentVec;
	}

#if Intersection_check
	//we may have split branches from the interesction check, add the control points for each branch with multiple branches
	for (int i = prePushCount - 5; i < *count; i++) {
		int master = root[i];
		//Check if a branch has multiple segments
		if (master != -1 && children[2 * master] != 0 && children[2 * master] == children[(2 * master) + 1]) {
			linearPointX[master] = -1;//this branch has multiple segments
			controlPts[master].push_back({ xstart[master], ystart[master], zstart[master] });//initialize control points
			//branch was moved to remove an intersection, this is now a control point
			controlPts[master].push_back({ xstart[i], ystart[i], zstart[i] });
			xstop[master] = xstop[i];
			ystop[master] = ystop[i];
			zstop[master] = zstop[i];
			//check if branch was split multiple times and add all control points if it has
			int indx = children[2 * master];
			if (indx != -1) {
				while (children[indx * 2] != -1 && children[2 * root[indx]] == children[2 * root[indx] + 1]) {
					controlPts[master].push_back({ xstart[indx], ystart[indx], zstart[indx] });
					indx = children[indx * 2];
				}
				//update start/stop points and slopes
				xstop[master] = xstart[indx];
				ystop[master] = ystart[indx];
				zstop[master] = zstart[indx];
				endSlopeX[master] = endSlopeX[indx];
				endSlopeY[master] = endSlopeY[indx];
				endSlopeZ[master] = endSlopeZ[indx];
				controlPts[master].push_back({ xstop[indx], ystop[indx], zstop[indx] });//close control points
			}
			else {//no extra split branches
				endSlopeX[master] = endSlopeX[i];
				endSlopeY[master] = endSlopeY[i];
				endSlopeZ[master] = endSlopeZ[i];
				controlPts[master].push_back({ xstop[i], ystop[i], zstop[i] });//close control points
			}
		}
	}
#endif
	////////////////////////////////
	////Polynomial Interpolation////
	////////////////////////////////
	for (int i = 0; i < prePushCount; i++) {
		//check if there are control points
		if (linearPointX[i] != -1 || i == 0) {
			polyInterp(xstart[i], ystart[i], zstart[i], xstop[i], ystop[i], zstop[i], startSlopeX[i],
				startSlopeY[i], startSlopeZ[i], endSlopeX[i], endSlopeY[i], endSlopeZ[i], linearPointX[i], linearPointY[i], linearPointZ[i], interpBranches, i, outResolutionScaleFactor);
		}
		else {//this branch has control points
			polyInterpWithControlPoints(controlPts[i], interpBranches, i, outResolutionScaleFactor);
		}
	}

	delete[] angles, linearPointX, linearPointY, linearPointZ, startSlopeX, startSlopeY, startSlopeZ, endSlopeX, endSlopeY, endSlopeZ, controlPts;
	return;
}

//function to perform a polynomial interpolation to optimize branching angles
//(linear fit between start->linearPoint)
//(polynomial fit between linearPoint->stop)
//x,y,zstart/stop: x,y,z start/stop points of branch to be polynomial fitted
//startSlopeX,Y,Z: x,y,z start slope of the polynomial fit
//stopSlopeX,Y,Z: x,y,z stop slope of the polynomial fit
//linearPointX,Y,Z: x,y,z for which a linear fit is used till 
//interpBranches: vector to hold interpolation points
//kk: index of branch in the tree object
//	outResolutionScaleFactor: factor to scale the input resolution by
void Liver::polyInterp(float xstart, float ystart, float zstart, float xstop, float ystop, float zstop, float startSlopeX,
	float startSlopeY, float startSlopeZ, float endSlopeX, float endSlopeY, float endSlopeZ, float linearPointX,
	float linearPointY, float linearPointZ, vector <xyz> *interpBranches, int k, float outResolutionScaleFactor) {

	interpBranches[k].push_back({ xstart, ystart, zstart });
	//Initial linear interpolation 
	if (k != 0) {
		//get initial length of each segment
		int lengthLin = ceil(outResolutionScaleFactor * 1.5f * pow(pow(linearPointX - xstart, 2.0f) + pow(linearPointY - ystart, 2.0f) + pow(linearPointZ - zstart, 2.0f), 0.5f));
		if (lengthLin < 5) { lengthLin = 5; }

		float x, y, z;
		for (int i = 1; i < lengthLin; i++) {
			x = xstart + (float)i * (linearPointX - xstart) / (float)lengthLin;
			y = ystart + (float)i * (linearPointY - ystart) / (float)lengthLin;
			z = zstart + (float)i * (linearPointZ - zstart) / (float)lengthLin;
			interpBranches[k].push_back({ x, y, z });
		}
		interpBranches[k].push_back({ linearPointX, linearPointY, linearPointZ });
	}
	//this is the root
	else {
		linearPointX = xstart;
		linearPointY = ystart;
		linearPointZ = zstart;
	}

	//Polynomial interp
	int lengthPoly = ceil(outResolutionScaleFactor * 1.5f * pow(pow(xstop - linearPointX, 2.0f) + pow(ystop - linearPointY, 2.0f) + pow(zstop - linearPointZ, 2.0f), 0.5f));
	if (lengthPoly < 5) { lengthPoly = 5; }


	//find initial slopes for interp
	float stPts[3] = { startSlopeX , startSlopeY, startSlopeZ };
	float linPts[3] = { linearPointX , linearPointY, linearPointZ };
	float divdif[3] = { xstop - linPts[0] , ystop - linPts[1], zstop - linPts[2] };
	float dzzdx[3] = { divdif[0] - stPts[0] , divdif[1] - stPts[1], divdif[2] - stPts[2] };
	float dzdxdx[3] = { endSlopeX - divdif[0], endSlopeY - divdif[1], endSlopeZ - divdif[2] };
	float dif_dzzdx_dzdxdx[3] = {2 * dzzdx[0] - dzdxdx[0], 2 * dzzdx[1] - dzdxdx[1], 2 * dzzdx[2] - dzdxdx[2] };
	float dif_dzdxdx_dzzdx[3] = { dzdxdx[0] - dzzdx[0], dzdxdx[1] - dzzdx[1], dzdxdx[2] - dzzdx[2] };

	//find quarery point values
	float pt[3];
	for (int i = 1; i < lengthPoly; i++) {
		float l = ((float)i / (float)lengthPoly);
		for (int j = 0; j < 3; j++) {
			pt[j] = l * dif_dzdxdx_dzzdx[j] + dif_dzzdx_dzdxdx[j];
			pt[j] = l * pt[j] + stPts[j];
			pt[j] = l * pt[j] + linPts[j];
		}
		interpBranches[k].push_back({ pt[0], pt[1], pt[2] });
	}
	interpBranches[k].push_back({ xstop, ystop, zstop });
}

//function to perform a polynomial interpolation with a split branch to optimize branching angles
//controlPts: points for the spline to go through
//interpBranches: vector to hold interpolation points
//kk: index of branch in the tree object
//	outResolutionScaleFactor: factor to scale the input resolution by
void Liver::polyInterpWithControlPoints(vector <xyz> controlPts, vector <xyz> *interpBranches, int kk, float outResolutionScaleFactor) {

	//Initialize
	int k = 4;//order
	int d = 3;//dim
	int n = controlPts.size();
	Eigen::MatrixXd c = Eigen::MatrixXd::Zero(n, n);
	Eigen::MatrixXd yi(n, d);
	Eigen::MatrixXd c1(d, (n - 1) * k);
	Eigen::MatrixXd b = Eigen::MatrixXd::Ones(n, d);
	Eigen::MatrixXd divdif(n - 1, d);
	Eigen::MatrixXd s(n, d);
	Eigen::MatrixXd c4(n - 1, d);
	Eigen::MatrixXd c3(n - 1, d);

	//Perform cubic polynomial interpolation with control points
	//initialize values
	for (int i = 0; i < controlPts.size() - 1; i++) {
		divdif(i, 0) = controlPts.at(i + 1).x - controlPts.at(i).x;
		divdif(i, 1) = controlPts.at(i + 1).y - controlPts.at(i).y;
		divdif(i, 2) = controlPts.at(i + 1).z - controlPts.at(i).z;
		yi(i, 0) = controlPts.at(i).x;
		yi(i, 1) = controlPts.at(i).y;
		yi(i, 2) = controlPts.at(i).z;
	}
	yi(n - 1, 0) = controlPts.at(n - 1).x;
	yi(n - 1, 1) = controlPts.at(n - 1).y;
	yi(n - 1, 2) = controlPts.at(n - 1).z;
	//build linear system
	for (int i = 1; i < n - 1; i++) {
		c(i, i) = 4;
		c(i, i - 1) = 1;
		c(i, i + 1) = 1;
		divdif.row(i);
		divdif.row(i - 1) + divdif.row(i);
		b.row(i) = divdif.row(i - 1) + divdif.row(i);
	}
	c(0, 0) = 1;
	c(n - 1, n - 1) = 1;

	//solve linear system
	s = c.colPivHouseholderQr().solve(b);
	for (int i = 0; i < n - 1; i++) {
		c4.row(i) = s.row(i) + s.row(i + 1) - 2 * divdif.row(i);
		c3.row(i) = (divdif.row(i) - s.row(i)) - c4.row(i);
	}
	c1 << c4.transpose(), c3.transpose(), s.topRows(n - 1).transpose(), yi.topRows(n - 1).transpose();
	c1 = c1.reshaped((n - 1) * d, 4).eval();
	//sample spline curve
	int l;
	float q, x, y, z;
	//loop through each segment
	interpBranches[kk].push_back({ controlPts.at(0).x, controlPts.at(0).y, controlPts.at(0).z });
	for (int i = 0; i < n - 1; i++) {
		l = ceil(outResolutionScaleFactor * pow(pow(yi(i + 1, 0) - yi(i, 0), 2.0f) + pow(yi(i + 1, 1) - yi(i, 1), 2.0f) + pow(yi(i + 1, 2) - yi(i, 2), 2.0f), 0.5f));
		if (l < 5) { l = 5; }
		int index = 0 + 3 * i;
		//loop through query points
		for (int j = 1; j < l + 1; j++) {
			q = (float)j / (float)l;
			x = c1(index, 0);
			y = c1(index + 1, 0);
			z = c1(index + 2, 0);
			//apply nested multiplication for each order of the polynomial
			for (int order = 1; order < k; order++) {
				x = q * x + c1(index, order);
				y = q * y + c1(index + 1, order);
				z = q * z + c1(index + 2, order);
			}
			interpBranches[kk].push_back({ x, y, z });
		}
	}
}



//function to convert to subscripts to linear indices (COLUMN MAJOR ORDER)
// x, y, z, l: the POSITIONS of the 1,2,3, and 4 dimensions
// dims = the SIZE of the 1,2,3, and 4 dimensions
void Liver::getLinearInd(unsigned int y,unsigned int x,unsigned int z,unsigned int l,unsigned int dims[4], int *index) {
	*index = y + dims[0] * x + dims[1]*dims[0] * z + dims[2]*dims[1]*dims[0] * l;
	return;
}




