#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
using namespace std;

#define D_SIZE 512
#define I_SIZE 512
// long long int D_size, I_size;

#define BLOCK__SIZE 1024

//#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
#define REAL float
#else
#define REAL double
#endif

// Variation function theoretical model, Type of variogram to use
enum Model
{
    Linear,
    LinearWithoutIntercept,
    Spherical,
    Exponential,
    Gaussian,
    Wave,
    RationalQuadratic,
    Circular
};

// Data Model
struct VariogramModel
{
    Model M;
    double Nugget; // NUGGET C0
    double Sill;   // SILL (C0+C)
    double Range;  // RANGE (Max distance to consider v(Range)=SILL)
};

// coordinate of point (x, y) and value z(x, y)
typedef struct __align__(16) Point
{
    REAL x, y, z;
    Point(){};
}
Point;

typedef struct CTrgl
{
    int pt0, pt1, pt2;
    CTrgl(){};
} CTrgl;

// executes  same row transformation for matrix_A and matrix_B
__device__ void Row_Transform_Matrix(REAL matrix_A[][D_SIZE], REAL matrix_B[][D_SIZE], int target_row, int source_row, int matrix_size)
{
    REAL value = matrix_A[target_row][source_row];
    if (target_row == source_row)
    {
        for (size_t i = 0; i < matrix_size; i++)
        {
            matrix_A[target_row][i] = matrix_A[target_row][i] / value;
            matrix_B[target_row][i] = matrix_B[target_row][i] / value;
        }
    }
    else
    {
        for (size_t i = 0; i < matrix_size; i++)
        {
            matrix_A[target_row][i] = matrix_A[target_row][i] - matrix_A[source_row][i] * value;
            matrix_B[target_row][i] = matrix_B[target_row][i] - matrix_B[source_row][i] * value;
        }
    }
}

__device__ REAL Calculate_semivariance(VariogramModel vm, REAL distance)
{
    REAL semivariance = 0.0;

    // Linear Model does't have sill, so it is impossible to calculate
    switch (vm.M)
    {
    case Linear: // None
    case LinearWithoutIntercept:
        break;
    case Spherical:
        if (distance == 0.0)
            semivariance = vm.Sill;
        else if (distance < vm.Range)
            semivariance = vm.Sill * (1 - (1.5 * (distance / vm.Range) - 0.5 * (pow(distance / vm.Range, 3.0))));
        else
            semivariance = vm.Sill;
        break;
    case Exponential:
        if (distance == 0.0)
            semivariance = vm.Sill;
        else
            semivariance = (vm.Sill - vm.Nugget) * (exp(-distance / vm.Range));
        break;
    case Gaussian:
        if (distance == 0.0)
            semivariance = vm.Sill;
        else
            semivariance = (vm.Sill - vm.Nugget) * (exp(-pow(distance / vm.Range, 2.0)));
        break;
    case Wave:
        if (distance == 0.0)
            semivariance = vm.Sill;
        else
            semivariance = vm.Nugget + ((vm.Sill - vm.Nugget) * (1 - (sin(distance / vm.Range) / (distance / vm.Range))));
        break;
    case RationalQuadratic:
        if (distance == 0.0)
            semivariance = vm.Sill;
        else
            semivariance = vm.Nugget + ((vm.Sill - vm.Nugget) * (pow(distance / vm.Range, 2.0) / (1 + pow(distance / vm.Range, 2.0))));
        break;
    case Circular:
        if (distance == 0.0 || distance > vm.Range)
            semivariance = vm.Sill;
        else
            semivariance = vm.Nugget + ((vm.Sill - vm.Nugget) * (1 - (2 / M_PI) * acos(distance / vm.Range) + (2 / M_PI) * (distance / vm.Range) * sqrt(1 - pow(distance / vm.Range, 2.0))));
    default:
        break;
    }

    return semivariance;
}

// Naive version SOA
__global__ void Naive_Kriging_Kernel_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                                         REAL *ix, REAL *iy, REAL *iz, int inum,
                                         VariogramModel vm, REAL AREA)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    REAL A_semivariance_matrix[D_SIZE][D_SIZE];
    REAL inversed_A[D_SIZE][D_SIZE];
    REAL B_semivariance_matrix[D_SIZE];
    REAL weight_martix[D_SIZE][D_SIZE];
    REAL distance = 0.0;

    if (tid < dnum)
    {
        for (size_t i = 0; i < dnum; i++)
        {
            for (size_t j = i; j < dnum; j++)
            {
                distance = sqrt(pow(dx[i] - dx[j], 2.0) + pow(dy[i] - dy[j], 2.0));
                A_semivariance_matrix[i][j] = Calculate_semivariance(vm, distance);
                if (j != i)
                    A_semivariance_matrix[j][i] = A_semivariance_matrix[i][j];
            }
        }

        for (size_t i = 0; i < dnum; i++)
            inversed_A[i][i] = 1.0;

        for (size_t i = 0; i < dnum; i++)
        {
            // converts original_matrix[i][i] to 1.0, does the same transformation to inversed_matrix
            Row_Transform_Matrix(A_semivariance_matrix, inversed_A, i, i, dnum);
            // converts original_matrix[j][i] (i≠j) to 0.0 using row-transformation, does the same transformation to inversed_matrix
            for (size_t j = 0; j < dnum; j++)
            {
                if (j != i)
                    Row_Transform_Matrix(A_semivariance_matrix, inversed_A, j, i, dnum);
            }
        }
    }
    if (tid < inum)
    {

        for (size_t j = 0; j < dnum; j++)
        {
            distance = sqrt(pow(ix[tid] - dx[j], 2.0) + pow(iy[tid] - dy[j], 2.0));
            B_semivariance_matrix[j] = Calculate_semivariance(vm, distance);
        }

        for (size_t i = 0; i < dnum; i++)
        {
            weight_martix[tid][i] = 0.0;
            for (size_t k = 0; k < dnum; k++)
            {
                weight_martix[tid][i] += (inversed_A[i][k] * B_semivariance_matrix[k]);
            }
        }

        iz[tid] = 0.0;
        for (size_t k = 0; k < dnum; k++)
        {
            iz[tid] += weight_martix[tid][k] * dz[k];
        }
    }
}
void Cuda_Naive_Kriging_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                            REAL *ix, REAL *iy, REAL *iz, int inum,
                            VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    REAL *dev_dx, *dev_dy, *dev_dz;
    REAL *dev_ix, *dev_iy, *dev_iz;

    cudaMalloc((void **)&dev_dx, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dy, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dz, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_ix, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iy, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iz, inum * sizeof(REAL));

    cudaMemcpy(dev_dx, dx, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dy, dy, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dz, dz, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ix, ix, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iy, iy, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iz, iz, inum * sizeof(REAL), cudaMemcpyHostToDevice);

    // 待定
    int threadsPerBlock = 4;
    int blocksPerGrid = 4;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    Naive_Kriging_Kernel_SOA<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dev_dz, dnum,
                                                                 dev_ix, dev_iy, dev_iz, inum,
                                                                 vm, AREA);

    cudaDeviceSynchronize();

    cudaMemcpy(iz, dev_iz, inum * sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(dev_dx);
    cudaFree(dev_dy);
    cudaFree(dev_dz);
    cudaFree(dev_ix);
    cudaFree(dev_iy);
    cudaFree(dev_iz);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("Cuda_Naive_Kriging_SOA\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}
/**
// Tiled version SOA
__global__ void Tiled_Kriging_Kernel_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                                         REAL *ix, REAL *iy, REAL *iz, int inum,
                                         VariogramModel vm, REAL AREA)
{
    int tid = blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ REAL A_semivariance_matrix[D_SIZE][D_SIZE];
    __shared__ REAL inversed_A[D_SIZE][D_SIZE];
    REAL B_semivariance_matrix[D_SIZE];
    REAL weight_martix[D_SIZE][D_SIZE];
    REAL distance = 0.0;

    int BLOCK_SIZE = 2;
    int part = (dnum-1)/BLOCK_SIZE;

    for (int i = 0; i < part; i++)
    {
        for (int j = i; j < part; j++)
        {
            distance = sqrt(pow(dx[threadIdx.x+i*BLOCK_SIZE] - dx[threadIdx.y+j*BLOCK_SIZE], 2.0) + pow(dy[threadIdx.x+i*BLOCK_SIZE] - dy[threadIdx.y+j*BLOCK_SIZE], 2.0));
            A_semivariance_matrix[threadIdx.x+i*BLOCK_SIZE][threadIdx.y+j*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
            A_semivariance_matrix[threadIdx.y+j*BLOCK_SIZE][threadIdx.x+i*BLOCK_SIZE] = A_semivariance_matrix[threadIdx.x+i*BLOCK_SIZE][threadIdx.y+j*BLOCK_SIZE];
        }
        if (threadIdx.x <(dnum-part*BLOCK_SIZE))
        {
            distance = sqrt(pow(dx[threadIdx.x+part*BLOCK_SIZE] - dx[threadIdx.y+i*BLOCK_SIZE], 2.0) + pow(dy[threadIdx.x+part*BLOCK_SIZE] - dy[threadIdx.y+i*BLOCK_SIZE], 2.0));
            A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+i*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
            A_semivariance_matrix[threadIdx.y+i*BLOCK_SIZE][threadIdx.x+part*BLOCK_SIZE] = A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+i*BLOCK_SIZE];
        }
    }

    if(threadIdx.x <(dnum-part*BLOCK_SIZE)&&threadIdx.y <(dnum-part*BLOCK_SIZE))
    {
        distance = sqrt(pow(dx[threadIdx.x+part*BLOCK_SIZE] - dx[threadIdx.y+part*BLOCK_SIZE], 2.0) + pow(dy[threadIdx.x+part*BLOCK_SIZE] - dy[threadIdx.y+part*BLOCK_SIZE], 2.0));
        A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+part*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
    }


    __syncthreads();

    for (int i = 0; i < part; i++)
    {
        inversed_A[threadIdx.x+i*BLOCK_SIZE][threadIdx.x+i*BLOCK_SIZE]=1.0;
    }
    if (threadIdx.x<(dnum-part*BLOCK_SIZE))
    {
        inversed_A[threadIdx.x+part*BLOCK_SIZE][threadIdx.x+part*BLOCK_SIZE]=1.0;
    }

    __syncthreads();


    for (size_t i = 0; i < dnum; i++)
    {
        // converts original_matrix[i][i] to 1.0, does the same transformation to inversed_matrix
        Row_Transform_Matrix(A_semivariance_matrix, inversed_A, i, i, dnum);
        // converts original_matrix[j][i] (i≠j) to 0.0 using row-transformation, does the same transformation to inversed_matrix
        for (size_t j = 0; j < dnum; j++)
        {
            if (j != i)
                Row_Transform_Matrix(A_semivariance_matrix, inversed_A, j, i, dnum);
        }
    }

    if (tid<inum)
    {
        for (size_t j = 0; j < dnum; j++)
        {
            distance = sqrt(pow(ix[tid] - dx[j], 2.0) + pow(iy[tid] - dy[j], 2.0));
            B_semivariance_matrix[j] = Calculate_semivariance(vm, distance);
        }

        for (size_t i = 0; i < dnum; i++)
        {
            weight_martix[tid][i] = 0.0;
            for (size_t k = 0; k < dnum; k++)
            {
                weight_martix[tid][i] += (inversed_A[i][k] * B_semivariance_matrix[k]);
            }
        }
        iz[tid] = 0.0;
        for (size_t k = 0; k < dnum; k++)
        {
            iz[tid] += weight_martix[tid][k] * dz[k];
        }
    }
}
void Cuda_Tiled_Kriging_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                            REAL *ix, REAL *iy, REAL *iz, int inum,
                            VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    REAL *dev_dx, *dev_dy, *dev_dz;
    REAL *dev_ix, *dev_iy, *dev_iz;

    cudaMalloc((void **)&dev_dx, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dy, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dz, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_ix, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iy, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iz, inum * sizeof(REAL));

    cudaMemcpy(dev_dx, dx, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dy, dy, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dz, dz, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ix, ix, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iy, iy, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iz, iz, inum * sizeof(REAL), cudaMemcpyHostToDevice);

    // 待定
    int blocksPerGrid = 2;
    dim3 threadsPerBlock(2, 2);

    printf("CUDA kernel launch with %d blocks of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock.x, threadsPerBlock.y);

    Tiled_Kriging_Kernel_SOA<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dev_dz, dnum,
                                                                 dev_ix, dev_iy, dev_iz, inum,
                                                                 vm, AREA);

    cudaDeviceSynchronize();

    cudaMemcpy(iz, dev_iz, inum * sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(dev_dx);
    cudaFree(dev_dy);
    cudaFree(dev_dz);
    cudaFree(dev_ix);
    cudaFree(dev_iy);
    cudaFree(dev_iz);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("Tiled_Kriging_Kernel_SOA\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}
**/

__device__ REAL A_semivariance_matrix[D_SIZE][D_SIZE];
__device__ REAL inversed_A[D_SIZE][D_SIZE];
// __device__ REAL weight[D_SIZE];

// __device__ REAL B_semivariance_matrix[D_SIZE];
// __device__ REAL weight_martix[D_SIZE][D_SIZE];
// Inverts a Matrix using gauss-jordan reduction method
__device__ void Inverse_Matrix()
{

    for (size_t i = 0; i < D_SIZE; i++)
        inversed_A[i][i] = 1.0;

    // converts original_matrix to Identity matrix, gets the result matrix: inversed_matrix
    for (size_t i = 0; i < D_SIZE; i++)
    {
        // converts original_matrix[i][i] to 1.0, does the same transformation to inversed_matrix
        Row_Transform_Matrix(A_semivariance_matrix, inversed_A, i, i, D_SIZE);
        // converts original_matrix[j][i] (i≠j) to 0.0 using row-transformation, does the same transformation to inversed_matrix
        for (size_t j = 0; j < D_SIZE; j++)
        {
            if (j != i)
                Row_Transform_Matrix(A_semivariance_matrix, inversed_A, j, i, D_SIZE);
        }
    }
}
// Naive version AOAS
__global__ void Naive_Kriging_Kernel_AOAS(Point *dpts, int ndp,
                                          Point *ipts, int idp,
                                          VariogramModel vm, REAL AREA
                                          , REAL weight_martix[D_SIZE][D_SIZE])
//   , REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE])
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    REAL distance = 0.0;
    if (tid < ndp)
    {
        // generate A_semivariance_matrix
        printf("===1-%d==", tid);
        for (size_t j = tid; j < ndp; j++)
        {
            distance = sqrt(pow(dpts[tid].x - dpts[j].x, 2.0) + pow(dpts[tid].y - dpts[j].y, 2.0));
            A_semivariance_matrix[tid][j] = Calculate_semivariance(vm, distance);
            if (j != tid)
                A_semivariance_matrix[j][tid] = A_semivariance_matrix[tid][j];
        }
        inversed_A[tid][tid] = 1.0;
        printf("===2-%d==", tid);
        __syncthreads();
        if (tid == 100)
        {

            Inverse_Matrix();
            printf("===321-%d==\n", tid);
        }
        printf("===3-%d==", tid);

        __syncthreads();
        // if (tid < idp)
        // {
        printf("===4-%d==", tid);

        for (size_t i = 0; i < ndp; i++)
        {
            weight_martix[tid][i] = 0.0;
            for (size_t k = 0; k < ndp; k++)
            {
                distance = sqrt(pow(ipts[tid].x - dpts[k].x, 2.0) + pow(ipts[tid].y - dpts[k].y, 2.0));
                weight_martix[tid][i] += (inversed_A[i][k] * Calculate_semivariance(vm, distance));
            }
            ipts[tid].z += weight_martix[tid][i] * dpts[i].z;
        }
        printf("===5-%d==", tid);
        // }
    }
}
void Cuda_Naive_Kriging_AOAS(Point *dpts, int ndp,
                             Point *ipts, int idp,
                             VariogramModel vm, REAL AREA, REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE], REAL weight_martix[][D_SIZE])
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    Point *dev_dpts;
    Point *dev_ipts;

    REAL(*A)
    [D_SIZE];
    REAL(*IA)
    [D_SIZE];
    REAL(*w)[D_SIZE];
    // REAL *w;

    cudaMalloc((void **)&dev_dpts, ndp * sizeof(Point));
    cudaMalloc((void **)&dev_ipts, idp * sizeof(Point));
    cudaMalloc((void **)&A, D_SIZE * D_SIZE * sizeof(REAL));
    cudaMalloc((void **)&IA, D_SIZE * D_SIZE * sizeof(REAL));
    cudaMalloc((void **)&w, I_SIZE * D_SIZE * sizeof(REAL));
    // cudaMalloc((void **)&w,  D_SIZE * sizeof(REAL));

    cudaMemcpy(dev_dpts, dpts, ndp * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ipts, ipts, idp * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(A, A_semivariance_matrix, D_SIZE * D_SIZE * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(IA, inversed_A, D_SIZE * D_SIZE * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(w, weight_martix, I_SIZE * D_SIZE * sizeof(REAL), cudaMemcpyHostToDevice);
    // cudaMemcpy(w, weight_martix,  D_SIZE * sizeof(REAL), cudaMemcpyHostToDevice);

    // 待定
    int threadsPerBlock = BLOCK__SIZE;
    int blocksPerGrid = (idp + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    Naive_Kriging_Kernel_AOAS<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp,
                                                                  dev_ipts, idp,
                                                                  vm, AREA, w);
    //   , A, IA);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    cudaDeviceSynchronize();
    cudaMemcpy(ipts, dev_ipts, idp * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(dev_dpts);
    cudaFree(dev_ipts);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("Naive_Kriging_Kernel_AOAS\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}
/**
// Tiled version AOAS
__global__ void Tiled_Kriging_Kernel_AOAS(Point *dpts, int ndp,
                                          Point *ipts, int idp,
                                          VariogramModel vm, REAL AREA)
{
    int tid = blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ REAL A_semivariance_matrix[D_SIZE][D_SIZE];
    __shared__ REAL inversed_A[D_SIZE][D_SIZE];
    REAL B_semivariance_matrix[D_SIZE];
    REAL weight_martix[D_SIZE][D_SIZE];
    REAL distance = 0.0;

    int BLOCK_SIZE = 2;
    int part = (ndp-1)/BLOCK_SIZE;

    for (int i = 0; i < part; i++)
    {
        for (int j = i; j < part; j++)
        {
            distance = sqrt(pow(dpts[threadIdx.x+i*BLOCK_SIZE].x - dpts[threadIdx.y+j*BLOCK_SIZE].x, 2.0) + pow(dpts[threadIdx.x+i*BLOCK_SIZE].y - dpts[threadIdx.y+j*BLOCK_SIZE].y, 2.0));
            A_semivariance_matrix[threadIdx.x+i*BLOCK_SIZE][threadIdx.y+j*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
            A_semivariance_matrix[threadIdx.y+j*BLOCK_SIZE][threadIdx.x+i*BLOCK_SIZE] = A_semivariance_matrix[threadIdx.x+i*BLOCK_SIZE][threadIdx.y+j*BLOCK_SIZE];
        }
        if (threadIdx.x <(ndp-part*BLOCK_SIZE))
        {
            distance = sqrt(pow(dpts[threadIdx.x+part*BLOCK_SIZE].x - dpts[threadIdx.y+i*BLOCK_SIZE].x, 2.0) + pow(dpts[threadIdx.x+part*BLOCK_SIZE].y - dpts[threadIdx.y+i*BLOCK_SIZE].y, 2.0));
            A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+i*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
            A_semivariance_matrix[threadIdx.y+i*BLOCK_SIZE][threadIdx.x+part*BLOCK_SIZE] = A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+i*BLOCK_SIZE];
        }
    }

    if(threadIdx.x <(ndp-part*BLOCK_SIZE)&&threadIdx.y <(ndp-part*BLOCK_SIZE))
    {
        distance = sqrt(pow(dpts[threadIdx.x+part*BLOCK_SIZE].x - dpts[threadIdx.y+part*BLOCK_SIZE].x, 2.0) + pow(dpts[threadIdx.x+part*BLOCK_SIZE].y - dpts[threadIdx.y+part*BLOCK_SIZE].y, 2.0));
        A_semivariance_matrix[threadIdx.x+part*BLOCK_SIZE][threadIdx.y+part*BLOCK_SIZE] = Calculate_semivariance(vm, distance);
    }

    __syncthreads();

    for (int i = 0; i < part; i++)
    {
        inversed_A[threadIdx.x+i*BLOCK_SIZE][threadIdx.x+i*BLOCK_SIZE]=1.0;
    }
    if (threadIdx.x<(ndp-part*BLOCK_SIZE))
    {
        inversed_A[threadIdx.x+part*BLOCK_SIZE][threadIdx.x+part*BLOCK_SIZE]=1.0;
    }

    __syncthreads();

    for (size_t i = 0; i < ndp; i++)
    {
        // converts original_matrix[i][i] to 1.0, does the same transformation to inversed_matrix
        Row_Transform_Matrix(A_semivariance_matrix, inversed_A, i, i, ndp);
        // converts original_matrix[j][i] (i≠j) to 0.0 using row-transformation, does the same transformation to inversed_matrix
        for (size_t j = 0; j < ndp; j++)
        {
            if (j != i)
                Row_Transform_Matrix(A_semivariance_matrix, inversed_A, j, i, ndp);
        }
    }
    if (tid<idp)
    {
        for (size_t j = 0; j < idp; j++)
        {
            distance = sqrt(pow(ipts[tid].x - dpts[j].x, 2.0) + pow(ipts[tid].y - dpts[j].y, 2.0));
            B_semivariance_matrix[j] = Calculate_semivariance(vm, distance);
        }
        for (size_t i = 0; i < idp; i++)
        {
            weight_martix[tid][i] = 0.0;
            for (size_t k = 0; k < idp; k++)
            {
                weight_martix[tid][i] += (inversed_A[i][k] * B_semivariance_matrix[k]);
            }
        }
        ipts[tid].z = 0.0;
        for (size_t k = 0; k < idp; k++)
        {
            ipts[tid].z += weight_martix[tid][k] * dpts[k].z;
        }
    }
}
void Cuda_Tiled_Kriging_AOAS(Point *dpts, int ndp,
                             Point *ipts, int idp,
                             VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    Point *dev_dpts;
    Point *dev_ipts;

    cudaMalloc((void **)&dev_dpts, ndp * sizeof(Point));
    cudaMalloc((void **)&dev_ipts, idp * sizeof(Point));

    cudaMemcpy(dev_dpts, dpts, ndp * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ipts, ipts, idp * sizeof(Point), cudaMemcpyHostToDevice);

    // 待定
    int blocksPerGrid = 2;
    dim3 threadsPerBlock(2, 2);

    printf("CUDA kernel launch with %d blocks of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock.x, threadsPerBlock.y);

    Tiled_Kriging_Kernel_AOAS<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp,
                                                                  dev_ipts, idp,
                                                                  vm, AREA);

    cudaDeviceSynchronize();

    cudaMemcpy(ipts, dev_ipts, idp * sizeof(Point), cudaMemcpyDeviceToHost);

    cudaFree(dev_dpts);
    cudaFree(dev_ipts);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("Tiled_Kriging_Kernel_AOAS\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}
**/
// test
/**
int main()
{
    VariogramModel v;
    v.Nugget = 0.1;
    v.Sill = 8.5;
    v.Range = 25;
    v.M = Gaussian;
    cudaSetDevice(1);

    REAL width = 350, height = 250;
    REAL A = width * height;

    // The coordinates of the Aoas version for known points
    Point *dpts = new Point[D_SIZE];
    Point *ipts = new Point[I_SIZE];

    dpts[0].x = 1;
    dpts[0].y = 2;
    dpts[0].z = 3;
    dpts[1].x = 4;
    dpts[1].y = 5;
    dpts[1].z = 6;
    dpts[2].x = 14;
    dpts[2].y = 15;
    dpts[2].z = 16;

    ipts[0].x = 7;
    ipts[0].y = 8;
    ipts[1].x = 10;
    ipts[1].y = 11;
    ipts[2].x = 20;
    ipts[2].y = 21;

    // The coordinates of the SoA version for known points
    REAL *dx = new REAL[D_SIZE];
    REAL *dy = new REAL[D_SIZE];
    REAL *dz = new REAL[D_SIZE];

    REAL *ix = new REAL[I_SIZE];
    REAL *iy = new REAL[I_SIZE];
    REAL *iz = new REAL[I_SIZE];

    dx[0] = 1;
    dy[0] = 2;
    dz[0] = 3;
    dx[1] = 4;
    dy[1] = 5;
    dz[1] = 6;
    dx[2] = 14;
    dy[2] = 15;
    dz[2] = 16;

    ix[0] = 7;
    iy[0] = 8;
    ix[1] = 10;
    iy[1] = 11;
    ix[2] = 20;
    iy[2] = 21;
/**
    // test execute
    Cuda_Naive_Kriging_SOA(dx, dy, dz, D_SIZE, ix, iy, iz, I_SIZE, v, A);
    cout << "Naive SOA: ";
    for (int i = 0; i < I_SIZE; i++)
    {
        cout << iz[i] << " ";
    }
    cout << endl;

    Cuda_Tiled_Kriging_SOA(dx, dy, dz, D_SIZE, ix, iy, iz, I_SIZE, v, A);
    cout << "Tiled SOA: ";
    for (int i = 0; i < I_SIZE; i++)
    {
        cout << iz[i] << " ";
    }
    cout << endl;

    Cuda_Naive_Kriging_AOAS(dpts, D_SIZE, ipts, I_SIZE, v, A);
    cout << "Naive AOAS: ";
    for (int i = 0; i < I_SIZE; i++)
    {
        cout << ipts[i].z << " ";
    }
    cout << endl;

    // Cuda_Tiled_Kriging_AOAS(dpts, D_SIZE, ipts, I_SIZE, v, A);
    // cout << "Tiled AOAS: ";
    // for (int i = 0; i < I_SIZE; i++)
    // {
    //     cout << ipts[i].z << " ";
    // }
    // cout << endl;

    return 0;
}
**/

int main()
{
    Point *dpts;
    Point *ipts;
    CTrgl *trgls;

    string data_root = "./data/";
    string inBase, inPoint;
    string flag;
    int point, face, line;

    cout << endl
         << "Known points file: ";
    cin >> inBase;
    ifstream fin(data_root + inBase);
    if (!fin)
    {
        cout << "\nCannot Open File ! " << inBase << endl;
        exit(1);
    }
    fin >> flag >> point >> face >> line;
    // D_size=point;
    dpts = new Point[D_SIZE];
    for (size_t i = 0; i < D_SIZE; i++)
    {
        fin >> dpts[i].x >> dpts[i].y >> dpts[i].z;
    }
    fin.close();

    cout << endl
         << "Unknown points file: ";
    cin >> inPoint;
    ifstream fin2(data_root + inPoint);
    if (!fin2)
    {
        cout << "\nCannot Open File ! " << inPoint << endl;
        exit(1);
    }
    fin2 >> flag >> point >> face >> line;
    // I_size= point;
    ipts = new Point[I_SIZE];
    trgls = new CTrgl[face];
    for (size_t i = 0; i < I_SIZE; i++)
    {
        fin2 >> ipts[i].x >> ipts[i].y >> ipts[i].z;
    }
    int num;
    for (size_t i = 0; i < face; i++)
    {
        fin2 >> num >> trgls[i].pt0 >> trgls[i].pt1 >> trgls[i].pt2;
    }
    fin2.close();
    cout << endl;

    VariogramModel v;
    v.Nugget = 0.1;
    v.Sill = 8.5;
    v.Range = 25;
    v.M = Gaussian;
    REAL width = 350, height = 250;
    REAL A = width * height;

    REAL A_semivariance_matrix[D_SIZE][D_SIZE];
    REAL inversed_A[D_SIZE][D_SIZE];
    REAL weight_martix[I_SIZE][D_SIZE];

    cout << "Kriging is on running..." << endl;

    Cuda_Naive_Kriging_AOAS(dpts, D_SIZE, ipts, I_SIZE, v, A, A_semivariance_matrix, inversed_A, weight_martix);

    string result_root = "./result/";
    string result;
    cout << "save the result in: ";
    cin >> result;

    ofstream fout(result_root + result);
    if (!fout)
    {
        cout << "\nCannot Save File ! " << result << endl;
        exit(1);
    }
    fout << "OFF" << endl;
    fout << point << "  " << face << "  " << 0 << endl;
    for (size_t i = 0; i < point; i++)
    {
        fout << ipts[i].x << "   " << ipts[i].y << "   " << ipts[i].z << endl;
    }
    for (size_t i = 0; i < face; i++)
    {
        fout << "3    " << trgls[i].pt0 << "   " << trgls[i].pt1 << "   " << trgls[i].pt2 << endl;
    }
    fout.close();
    cout << endl;

    delete[] dpts;
    delete[] ipts;
    delete[] trgls;

    return 0;
}
