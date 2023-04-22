/*
 * CPU implementation of AIDW interpolation algorithm
 *
 * By Dr.Gang Mei
 *
 * Created on 2015.08.10, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * Revised on 2015.12.23, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * 
 *********************************************************************
 * Related publications:
 *  1) "An adaptive inverse-distance weighting spatial interpolation technique"
 *      George Y. Lu1, David W. Wong, Computers and Geosciences
 *     http://www.sciencedirect.com/science/article/pii/S0098300408000721
 *********************************************************************
 * Compiled with VS2010
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
using namespace std;

#define a1 1.5
#define a2 2.0
#define a3 2.5
#define a4 3.0
#define a5 3.5
#define R_min 0
#define R_max 2
#define kNN 15

//#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
    #define REAL float
#else
    #define REAL double
#endif

typedef struct CVert                                              
{                                                                       
	REAL x, y, z;
	CVert() {};                                                         
} CVert;

typedef struct CTrgl 
{
    int pt0, pt1, pt2; 
    CTrgl() {};
} CTrgl;

//CPU version
void AIDW(CVert * dpts, int  ndp, // Data points, known
	      CVert * ipts, int  idp, // Interplated points, unknown         
		  REAL  AREA)             // Area of planar region
{	
	REAL d = 0, t = 0, sum = 0, dist[kNN], alpha;

	for(int i = 0; i < idp; i++) {
        for(int j = 0; j < kNN; j++) {  // Distance of the first kNN points
			dist[j]= (ipts[i].x - dpts[j].x) * (ipts[i].x - dpts[j].x) +
                     (ipts[i].y - dpts[j].y) * (ipts[i].y - dpts[j].y);
		}	 
        for(int ii = 0; ii < kNN - 1; ii++)  // Sort in ascending order
            for (int jj = 0; jj < kNN - 1 - ii; jj++) 
                if(dist[jj] > dist[jj + 1]) {
                    d = dist[jj];  dist[jj] = dist[jj + 1];  dist[jj + 1] = d;
                }

        for(int j = kNN; j < ndp; j++) {  // All distances
            d =(ipts[i].x - dpts[j].x) * (ipts[i].x - dpts[j].x) +
               (ipts[i].y - dpts[j].y) * (ipts[i].y - dpts[j].y);
            if(d < dist[kNN-1]) {  //Potential nearest neighbor
                dist[kNN-1] = d;   //Replace the last distance
                for(int jj = kNN - 1; jj > 0; jj--) {  //Sort again by swapping
                    if(dist[jj] < dist[jj - 1]) {
                        d = dist[jj];  dist[jj] = dist[jj - 1];  dist[jj - 1] = d;
                    }
				}
            }
        }		
		
        for(int j = 0; j < kNN; j++)   sum += sqrt(dist[j]);
        REAL r_obs = sum / kNN;                            // The observed average nearest neighbor distance                 
        REAL r_exp = 1.0 / (2 * sqrt(ndp / AREA));         // The expected nearest neighbor distance for a random pattern
        REAL R_S0  = r_obs / r_exp;                        // The nearest neighbor statistic
		
       // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
		REAL u_R = 0;
		if(R_S0 >= R_min) u_R = 0.5-0.5 * cos(3.1415926 / R_max * (R_S0-R_min));
        if(R_S0 >= R_max) u_R = 1.0;
		 
		// Determine the appropriate distance-decay parameter alpha by a triangular membership function
		// Adaptive power parameter: a (alpha)
        if(u_R>= 0 && u_R<=0.1)  alpha = a1; 
        if(u_R>0.1 && u_R<=0.3)  alpha = a1*(1-5*(u_R-0.1)) + a2*5*(u_R-0.1);
        if(u_R>0.3 && u_R<=0.5)  alpha = a3*5*(u_R-0.3) + a1*(1-5*(u_R-0.3));
        if(u_R>0.5 && u_R<=0.7)  alpha = a3*(1-5*(u_R-0.5)) + a4*5*(u_R-0.5);
        if(u_R>0.7 && u_R<=0.9)  alpha = a5*5*(u_R-0.7) + a4*(1-5*(u_R-0.7));
	    if(u_R>0.9 && u_R<=1.0)  alpha = a5;		
		alpha *= 0.5; // Half of the power	

		REAL z = 0;	
		REAL t =0;
	    d = 0, sum = 0;
		for (int j = 0; j < ndp; j++) {		
			    d = (ipts[i].x - dpts[j].x) * (ipts[i].x - dpts[j].x) + 
					(ipts[i].y - dpts[j].y) * (ipts[i].y - dpts[j].y) ;
			    t = 1 /( pow(d, alpha));
				sum += t;
				z += dpts[j].z * t;							
		}
		ipts[i].z = z / sum;
	}		
}


// Simple example
int main(void)
{
	string tStr;   // The name of  the input file
	REAL x, y, z;  // The coordinates of the points
	int nVert, nTrgl, nEdge;  
	CVert * verts;                          
    CTrgl * trgls;                   
              
	// Input set of known points
	CVert * pts;

	char str1[20];
	cout << "\nPlease enter the file name of the known points:  ";
	cin >> str1;                                                 
	ifstream  fin(str1); 
	if (!fin) {
		cout << "\nCannot Open File ! " << str1 << endl;
		exit(1);
	}
	fin >> tStr >> nVert >> nTrgl >> nEdge;                       

	int nDataPoint = nVert;  // Number of known points            
	pts = new CVert[nVert];  //The coordinates of the known points
	for (int i = 0; i < nVert; i++)	{
		fin >> x >> y >> z;
		pts[i].x = x;
		pts[i].y = y;
		pts[i].z = z;		
	}
	fin.close();
		
	nDataPoint = nVert;

	cout << "\nPlease enter the file name of the unknown points:  ";
	cin >> str1;                                                               
	ifstream  finOFF(str1);	
	if (!finOFF) {
		cout << "\nCannot Open File ! " << str1 << endl;
		exit(1);
	}

	finOFF >> tStr >> nVert >> nTrgl >> nEdge;

	int nInplPoint = nVert; // Number of unknown points  

	verts = new CVert[nVert + 4];                                       
	trgls = new CTrgl[nVert * 3];

	for (int i = 0; i < nVert; i++) {
		finOFF >> x >> y >> z;
		verts[i].x = x;
		verts[i].y = y;
		verts[i].z = z;
	}
	int Num, ID0, ID1, ID2;
	for(int i = 0; i < nTrgl; i++) {
	      finOFF >> Num >> ID0 >> ID1 >> ID2;         
	      trgls[i].pt0 = ID0;                        
	      trgls[i].pt1 = ID1;                          
	      trgls[i].pt2 = ID2;                         
	 }
    finOFF.close();
	cout << endl;
	// End input
		
	// Area of planar region
	REAL width = 350 , height =250;
    REAL A = width * height;

    cout << "AIDW is on running..." << endl;
	AIDW(pts, nDataPoint, verts, nInplPoint, A);
	cout << "AIDW is completed" << endl;
	
	// Output the result
	char str2[20];
	cout << "\nSave the interpolation results:  ";
	cin >> str2;                                              
	ofstream fout(str2);
	if (!fout) {
		cout << "\nCannot Save File ! " << str2 << endl;
		exit(1);
	}
	fout << "OFF" << endl;
	fout << nVert << "  " << nTrgl << "  " << 0 << endl;
	for (int i = 0; i < nVert; i++) {
		x = verts[i].x;
		y = verts[i].y;
		z = verts[i].z;
		fout << x << "   " << y << "   " << z << endl;
	}
	for(int i = 0; i < nTrgl; i++) {
         fout <<"3    "<< trgls[i].pt0 <<"   "<< trgls[i].pt1 <<"   "<< trgls[i].pt2 << endl;
     }
    fout.close();
	cout << endl;
	fout.close();
	cout << endl;
	// End the output

	// Free up memory
	delete[] pts;
	delete[] verts;
	delete[] trgls;	
	system("pause");
	return 0;

}

// Experimental test
// int main(int argc, char *argv[])
// {
// 	printf("CPU version:\n");

// 	int numk = 500;
// 	int dnum = numk * 1024;
// 	CVert * dpts = new CVert[dnum];
// 	int inum = dnum;
// 	CVert * ipts = new CVert[inum];;
	
// 	// Area
//     double width = 2000 , height =2000;
//     double A = width * height;

// 	/* initialize random seed: */
//     srand (time(NULL));
	
// 	for(int i = 0; i < dnum; i++) {
// 		dpts[i].x = rand()/(double)RAND_MAX * 1000;
// 		dpts[i].y = rand()/(double)RAND_MAX * 1000;
// 		dpts[i].z = rand()/(double)RAND_MAX * 1000;
// 	}

// 	for(int i = 0; i < inum; i++) {
// 		ipts[i].x = rand()/(double)RAND_MAX * 1000;
// 		ipts[i].y = rand()/(double)RAND_MAX * 1000;
// 		ipts[i].z = 0.0f;
// 	}
	
// 	printf("dnum = : %d\ninum = : %d\n\n", dnum, inum);

// 	// CPU AIDW
// 	clock_t start, end;
// 	float cpuTime;
// 	start = clock();

// 	AIDW(dpts, dnum, ipts, inum, A);

// 	end = clock();
// 	cpuTime = (end - start) / (CLOCKS_PER_SEC);
// 	std::cout << "it took " << cpuTime << " s of CPU to execute this\n";
// 	std::cout << "it took " << (end - start) << " ms of CPU to execute this\n";

// 	delete [] dpts;  delete [] ipts;

// 	system("pause");

// 	return 0;

// }