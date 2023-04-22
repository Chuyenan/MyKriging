#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
using namespace std;

#define K 1024
#define D_SIZE 20 * K
#define I_SIZE 20 * K
// long long int D_size, I_size;

#define BLOCK_SIZE 1024
#define DIM2_BLOCK_SIZE 16

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

// Model based on http://spatial-analyst.net/ILWIS/htm/ilwisapp/sec/semivar_models_sec.htm
// Calculate semivariance
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

/** Inverts a Matrix using gauss-jordan reduction method **/
// nodiagonal_normalize
__global__ void nodiag_normalize(REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE], int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ REAL value;
    value = A_semivariance_matrix[i][i];

    if (x < n && y < n)
        if (x == i && x != y)
        {
            inversed_A[x][y] /= value;
            A_semivariance_matrix[x][y] /= value;
        }
}
// diagonal_normalize
__global__ void diag_normalize(REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE], int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ REAL value;
    value = A_semivariance_matrix[i][i];

    if (x < n && y < n)
        if (x == y && x == i)
        {
            inversed_A[x][y] /= value;
            A_semivariance_matrix[x][y] /= value;
        }
}
// gaussjordan
__global__ void gaussjordan(REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE], int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
    {
        if (x != i)
        {
            inversed_A[x][y] -= inversed_A[i][y] * A_semivariance_matrix[x][i];
            if (y != i)
            {
                A_semivariance_matrix[x][y] -= A_semivariance_matrix[i][y] * A_semivariance_matrix[x][i];
            }
        }
    }
}
// set_zero
__global__ void set_zero(REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE], int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
    {
        if (x != i)
        {
            if (y == i)
            {
                A_semivariance_matrix[x][y] = 0;
            }
        }
    }
}

// Naive AOAS generate semivariance matrix A
__global__ void Naive_Kriging_Kernel_AOAS_Generate_A(Point *dpts, int ndp, VariogramModel vm,
                                                     REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    REAL distance = 0.0;
    if (tid < ndp)
    {
        for (size_t j = tid; j < ndp; j++)
        {
            distance = sqrt(pow(dpts[tid].x - dpts[j].x, 2.0) + pow(dpts[tid].y - dpts[j].y, 2.0));
            A_semivariance_matrix[tid][j] = Calculate_semivariance(vm, distance);
            if (j != tid)
                A_semivariance_matrix[j][tid] = A_semivariance_matrix[tid][j];
        }
        inversed_A[tid][tid] = 1.0;
    }
}

// Tiled AOAS generate semivariance matrix A
__global__ void Tiled_Kriging_Kernel_AOAS_Generate_A(Point *dpts, int ndp, VariogramModel vm,
                                                     REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const int bsize = BLOCK_SIZE;

    __shared__ REAL SM_dx[bsize];
    __shared__ REAL SM_dy[bsize];
    int part = (ndp - 1) / bsize;

    REAL distance = 0.0;

    if (tid<ndp)
    {
        inversed_A[tid][tid] = 1.0;
    }
    for (size_t p = 0; p < part; p++)
    {
        SM_dx[threadIdx.x] = dpts[threadIdx.x + bsize * p].x;
        SM_dy[threadIdx.x] = dpts[threadIdx.x + bsize * p].y;
        __syncthreads();
        if (tid < ndp)
        {
            for (size_t bs = 0; bs < bsize; bs++)
            {
                distance = sqrt(pow(dpts[tid].x - SM_dx[bs], 2.0) + pow(dpts[tid].y - SM_dy[bs], 2.0));
                A_semivariance_matrix[tid][bs + bsize * p] = Calculate_semivariance(vm, distance);
                if ((bs + bsize * p) != tid)
                    A_semivariance_matrix[bs + bsize * p][tid] = A_semivariance_matrix[tid][bs + bsize * p];
            }
            // inversed_A[tid][tid] = 1.0;
        }
        __syncthreads();
    }

    if (threadIdx.x < (ndp - bsize * part))
    {
        SM_dx[threadIdx.x] = dpts[threadIdx.x + bsize * part].x;
        SM_dy[threadIdx.x] = dpts[threadIdx.x + bsize * part].y;
        __syncthreads();
        if (tid < ndp)
        {
            for (size_t bs = 0; bs < (ndp - bsize * part); bs++)
            {
                distance = sqrt(pow(dpts[tid].x - SM_dx[bs], 2.0) + pow(dpts[tid].y - SM_dy[bs], 2.0));
                A_semivariance_matrix[tid][bs + bsize * part] = Calculate_semivariance(vm, distance);
                if (bs + bsize * part != tid)
                    A_semivariance_matrix[bs + bsize * part][tid] = A_semivariance_matrix[tid][bs + bsize * part];
            }
            // inversed_A[tid][tid] = 1.0;
        }
        __syncthreads();
    }
    
    
}

// Naive SOA generate semivariance matrix A
__global__ void Naive_Kriging_Kernel_SOA_Generate_A(REAL *dx, REAL *dy, int dnum, REAL *ix, REAL *iy, int inum, VariogramModel vm,
                                                    REAL A_semivariance_matrix[][D_SIZE], REAL inversed_A[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    REAL distance = 0.0;
    if (tid < dnum)
    {
        for (size_t j = tid; j < dnum; j++)
        {
            distance = sqrt(pow(dx[tid] - dx[j], 2.0) + pow(dy[tid] - dy[j], 2.0));
            A_semivariance_matrix[tid][j] = Calculate_semivariance(vm, distance);
            if (j != tid)
                A_semivariance_matrix[j][tid] = A_semivariance_matrix[tid][j];
        }
        inversed_A[tid][tid] = 1.0;
    }
}

// Naive SOA generate matrix B
__global__ void Naive_Kriging_Kernel_SOA_Generate_B_Matrix(REAL *dx, REAL *dy, int dnum, REAL *ix, REAL *iy, int inum,
                                                           VariogramModel vm, REAL B_semivariance_matrix[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    REAL distance = 0.0;
    if (tid < dnum)
    {
        for (size_t j = 0; j < dnum; j++)
        {
            distance = sqrt(pow(ix[tid] - dx[j], 2.0) + pow(iy[tid] - dy[j], 2.0));
            B_semivariance_matrix[j][tid] = Calculate_semivariance(vm, distance);
        }
    }
}

// Naive AOAS generate matrix B
__global__ void Naive_Kriging_Kernel_AOAS_Generate_B_Matrix(Point *dpts, int ndp, Point *ipts, int idp,
                                                            VariogramModel vm, REAL B_semivariance_matrix[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    REAL distance = 0.0;
    if (tid < ndp)
    {
        for (size_t j = 0; j < ndp; j++)
        {
            distance = sqrt(pow(ipts[tid].x - dpts[j].x, 2.0) + pow(ipts[tid].y - dpts[j].y, 2.0));
            B_semivariance_matrix[j][tid] = Calculate_semivariance(vm, distance);
        }
    }
}

// Tiled AOAS generate matrix B
__global__ void Tiled_Kriging_Kernel_AOAS_Generate_B_Matrix(Point *dpts, int ndp, Point *ipts, int idp,
                                                            VariogramModel vm, REAL B_semivariance_matrix[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ REAL SM_dx[BLOCK_SIZE];
    __shared__ REAL SM_dy[BLOCK_SIZE];
    int part = (ndp - 1) / BLOCK_SIZE;

    REAL distance = 0.0;

    for (size_t p = 0; p < part; p++)
    {
        SM_dx[threadIdx.x] = dpts[threadIdx.x + BLOCK_SIZE * p].x;
        SM_dy[threadIdx.x] = dpts[threadIdx.x + BLOCK_SIZE * p].y;
        __syncthreads();
        if (tid < ndp)
        {
            for (size_t bs = 0; bs < BLOCK_SIZE; bs++)
            {
                distance = sqrt(pow(ipts[tid].x - SM_dx[bs], 2.0) + pow(ipts[tid].y - SM_dy[bs], 2.0));
                B_semivariance_matrix[bs + BLOCK_SIZE * p][tid] = Calculate_semivariance(vm, distance);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < (ndp - BLOCK_SIZE * part))
    {
        SM_dx[threadIdx.x] = dpts[threadIdx.x + BLOCK_SIZE * part].x;
        SM_dy[threadIdx.x] = dpts[threadIdx.x + BLOCK_SIZE * part].y;
        __syncthreads();
        if (tid < ndp)
        {
            for (size_t bs = 0; bs < (ndp - BLOCK_SIZE * part); bs++)
            {
                distance = sqrt(pow(ipts[tid].x - SM_dx[bs], 2.0) + pow(ipts[tid].y - SM_dy[bs], 2.0));
                B_semivariance_matrix[bs + BLOCK_SIZE * part][tid] = Calculate_semivariance(vm, distance);
            }
        }
        __syncthreads();
    }
}

// naive generate weight matrix
__global__ void Naive_Generate_Weight_Matrix(REAL inversed_A[][D_SIZE], REAL B_semivariance_matrix[][D_SIZE], int ndp, REAL weight_martix[][D_SIZE])
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < ndp && tx < ndp)
    {
        REAL c = 0;
        for (int i = 0; i < ndp; ++i)
        {
            c += inversed_A[ty][i] * B_semivariance_matrix[i][tx];
        }
        weight_martix[ty][tx] = c;
    }
}

// tiled generate weight matrix
__global__ void Tiled_Generate_Weight_Matrix(REAL inversed_A[][D_SIZE], REAL B_semivariance_matrix[][D_SIZE], int ndp, REAL weight_martix[][D_SIZE])
{
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;

    // int gx = blockIdx.x * blockDim.x + tx;
    // int gy = blockIdx.y * blockDim.y + ty;

    // REAL subC = 0;
    // __shared__ REAL subA[16][16];
    // __shared__ REAL subB[16][16];

    // for (int m = 0; m < ndp; m += 16)
    // {

    //     subA[tx][ty] = inversed_A[gy][m + tx];
    //     subB[tx][ty] = B_semivariance_matrix[m + ty][gx];
    //     __syncthreads();

    //     for (int i = 0; i < 16; i++)
    //     {
    //         subC += subA[i][ty] * subB[tx][i];
    //     }
    //     __syncthreads();
    // }
    // weight_martix[gy][gx] = subC;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < ndp && tx < ndp)
    {
        REAL c = 0;
        for (int i = 0; i < ndp; ++i)
        {
            c += inversed_A[ty][i] * B_semivariance_matrix[i][tx];
        }
        weight_martix[ty][tx] = c;
    }
}

// Naive AOAS generate ipts.z
__global__ void Naive_Kriging_Kernel_AOAS_Generate_Ipts_Z(Point *dpts, Point *ipts, int idp, REAL weight_martix[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < idp)
    {
        for (size_t i = 0; i < idp; i++)
        {
            ipts[tid].z += weight_martix[i][tid] * dpts[i].z;
        }
    }
}

// Naive SOA generate iz
__global__ void Naive_Kriging_Kernel_SOA_Generate_IZ(REAL *dz, REAL *iz, int inum, REAL weight_martix[][D_SIZE])
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < inum)
    {
        for (size_t i = 0; i < inum; i++)
        {
            iz[tid] += weight_martix[i][tid] * dz[i];
        }
    }
}

/** ------------------Naive version AOAS------------------ **/
void Cuda_Naive_Kriging_AOAS(Point *dpts, int ndp,
                             Point *ipts, int idp,
                             VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end0, end1, end2, end3, end4, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);
    cudaEventCreate(&end3);
    cudaEventCreate(&end4);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    Point *dev_dpts;
    Point *dev_ipts;

    REAL(*A)
    [D_SIZE];
    REAL(*IA)
    [D_SIZE];
    REAL(*B)
    [D_SIZE];
    REAL(*w)
    [D_SIZE];

    cudaMalloc((void **)&dev_dpts, ndp * sizeof(Point));
    cudaMalloc((void **)&dev_ipts, idp * sizeof(Point));
    cudaMalloc((void **)&A, ndp * ndp * sizeof(REAL));
    cudaMalloc((void **)&IA, ndp * ndp * sizeof(REAL));
    cudaMalloc((void **)&B, idp * ndp * sizeof(REAL));
    cudaMalloc((void **)&w, idp * ndp * sizeof(REAL));

    cudaMemcpy(dev_dpts, dpts, ndp * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ipts, ipts, idp * sizeof(Point), cudaMemcpyHostToDevice);

    // one-dimension block and one-dimension grid
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (ndp + threadsPerBlock - 1) / threadsPerBlock;
    // two-dimension block and two-dimension grid
    dim3 threadsPerBlock1(DIM2_BLOCK_SIZE, DIM2_BLOCK_SIZE);
    dim3 numBlocks((ndp + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE, (ndp + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE);

    printf("CUDA kernel launch with %d blocks of %d threads OR block (%d, %d) of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock, numBlocks.x, numBlocks.y, threadsPerBlock1.x, threadsPerBlock1.y);

    Naive_Kriging_Kernel_AOAS_Generate_A<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp, vm, A, IA);
    cudaEventRecord(end0, 0);
    cudaEventSynchronize(end0);
    float elapsedTime0;
    cudaEventElapsedTime(&elapsedTime0, start, end0);
    printf("Time to generate A: %3.3f ms\n", elapsedTime0);

    for (int i = 0; i < ndp; i++)
    {
        nodiag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
        diag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
        gaussjordan<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
        set_zero<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
    }
    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, end0, end1);
    printf("Time to generate IA: %3.3f ms\n", elapsedTime1);

    Naive_Kriging_Kernel_AOAS_Generate_B_Matrix<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp, dev_ipts, idp, vm, B);
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, end1, end2);
    printf("Time to generate B: %3.3f ms\n", elapsedTime2);

    Naive_Generate_Weight_Matrix<<<numBlocks, threadsPerBlock1>>>(IA, B, ndp, w);
    cudaEventRecord(end3, 0);
    cudaEventSynchronize(end3);
    float elapsedTime3;
    cudaEventElapsedTime(&elapsedTime3, end2, end3);
    printf("Time to generate Weight: %3.3f ms\n", elapsedTime3);

    Naive_Kriging_Kernel_AOAS_Generate_Ipts_Z<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, dev_ipts, idp, w);
    cudaEventRecord(end4, 0);
    cudaEventSynchronize(end4);
    float elapsedTime4;
    cudaEventElapsedTime(&elapsedTime4, end3, end4);
    printf("Time to generate Z: %3.3f ms\n", elapsedTime4);

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
    printf("Naive_Kriging_AOAS\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end0);
    cudaEventDestroy(end1);
    cudaEventDestroy(end2);
    cudaEventDestroy(end3);
    cudaEventDestroy(end4);
    cudaEventDestroy(end);

    cudaDeviceReset();
}

/** ------------------Naive version SOA------------------ **/
void Cuda_Naive_Kriging_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                            REAL *ix, REAL *iy, REAL *iz, int inum,
                            VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end0, end1, end2, end3, end4, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);
    cudaEventCreate(&end3);
    cudaEventCreate(&end4);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    REAL *dev_dx, *dev_dy, *dev_dz;
    REAL *dev_ix, *dev_iy, *dev_iz;

    REAL(*A)
    [D_SIZE];
    REAL(*IA)
    [D_SIZE];
    REAL(*B)
    [D_SIZE];
    REAL(*w)
    [D_SIZE];

    cudaMalloc((void **)&dev_dx, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dy, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dz, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_ix, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iy, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iz, inum * sizeof(REAL));

    cudaMalloc((void **)&A, dnum * dnum * sizeof(REAL));
    cudaMalloc((void **)&IA, dnum * dnum * sizeof(REAL));
    cudaMalloc((void **)&B, inum * dnum * sizeof(REAL));
    cudaMalloc((void **)&w, inum * dnum * sizeof(REAL));

    cudaMemcpy(dev_dx, dx, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dy, dy, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dz, dz, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ix, ix, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iy, iy, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iz, iz, inum * sizeof(REAL), cudaMemcpyHostToDevice);

    // one-dimension block and one-dimension grid
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (dnum + threadsPerBlock - 1) / threadsPerBlock;
    // two-dimension block and two-dimension grid
    dim3 threadsPerBlock1(DIM2_BLOCK_SIZE, DIM2_BLOCK_SIZE);
    dim3 numBlocks((dnum + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE, (dnum + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE);

    printf("CUDA kernel launch with %d blocks of %d threads OR block (%d, %d) of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock, numBlocks.x, numBlocks.y, threadsPerBlock1.x, threadsPerBlock1.y);

    Naive_Kriging_Kernel_SOA_Generate_A<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dnum, dev_ix, dev_iy, inum, vm, A, IA);
    cudaEventRecord(end0, 0);
    cudaEventSynchronize(end0);
    float elapsedTime0;
    cudaEventElapsedTime(&elapsedTime0, start, end0);
    printf("Time to generate A: %3.3f ms\n", elapsedTime0);

    for (int i = 0; i < dnum; i++)
    {
        nodiag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        diag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        gaussjordan<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        set_zero<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
    }
    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, end0, end1);
    printf("Time to generate IA: %3.3f ms\n", elapsedTime1);

    Naive_Kriging_Kernel_SOA_Generate_B_Matrix<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dnum, dev_ix, dev_iy, inum, vm, B);
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, end1, end2);
    printf("Time to generate B: %3.3f ms\n", elapsedTime2);

    Naive_Generate_Weight_Matrix<<<numBlocks, threadsPerBlock1>>>(IA, B, dnum, w);
    cudaEventRecord(end3, 0);
    cudaEventSynchronize(end3);
    float elapsedTime3;
    cudaEventElapsedTime(&elapsedTime3, end2, end3);
    printf("Time to generate Weight: %3.3f ms\n", elapsedTime3);

    Naive_Kriging_Kernel_SOA_Generate_IZ<<<blocksPerGrid, threadsPerBlock>>>(dev_dz, dev_iz, inum, w);
    cudaEventRecord(end4, 0);
    cudaEventSynchronize(end4);
    float elapsedTime4;
    cudaEventElapsedTime(&elapsedTime4, end3, end4);
    printf("Time to generate Z: %3.3f ms\n", elapsedTime4);

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
    printf("Naive_Kriging_SOA\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}

/** ------------------Tiled version AOAS------------------ **/
void Cuda_Tiled_Kriging_AOAS(Point *dpts, int ndp,
                             Point *ipts, int idp,
                             VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end0, end1, end2, end3, end4, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);
    cudaEventCreate(&end3);
    cudaEventCreate(&end4);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    Point *dev_dpts;
    Point *dev_ipts;

    REAL(*A)
    [D_SIZE];
    REAL(*IA)
    [D_SIZE];
    REAL(*B)
    [D_SIZE];
    REAL(*w)
    [D_SIZE];

    cudaMalloc((void **)&dev_dpts, ndp * sizeof(Point));
    cudaMalloc((void **)&dev_ipts, idp * sizeof(Point));
    cudaMalloc((void **)&A, ndp * ndp * sizeof(REAL));
    cudaMalloc((void **)&IA, ndp * ndp * sizeof(REAL));
    cudaMalloc((void **)&B, idp * ndp * sizeof(REAL));
    cudaMalloc((void **)&w, idp * ndp * sizeof(REAL));

    cudaMemcpy(dev_dpts, dpts, ndp * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ipts, ipts, idp * sizeof(Point), cudaMemcpyHostToDevice);

    // one-dimension block and one-dimension grid
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (ndp + threadsPerBlock - 1) / threadsPerBlock;
    // two-dimension block and two-dimension grid
    // dim3 threadsPerBlock1(DIM2_BLOCK_SIZE, DIM2_BLOCK_SIZE);
    // dim3 numBlocks((ndp + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE, (ndp + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE);

    // printf("CUDA kernel launch with %d blocks of %d threads OR block (%d, %d) of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock, numBlocks.x, numBlocks.y, threadsPerBlock1.x, threadsPerBlock1.y);

    Tiled_Kriging_Kernel_AOAS_Generate_A<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp, vm, A, IA);
    cudaEventRecord(end0, 0);
    cudaEventSynchronize(end0);
    float elapsedTime0;
    cudaEventElapsedTime(&elapsedTime0, start, end0);
    printf("Time to generate A: %3.3f ms\n", elapsedTime0);

    // for (int i = 0; i < ndp; i++)
    // {
    //     nodiag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
    //     diag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
    //     gaussjordan<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
    //     set_zero<<<numBlocks, threadsPerBlock1>>>(A, IA, ndp, i);
    // }
    // cudaEventRecord(end1, 0);
    // cudaEventSynchronize(end1);
    // float elapsedTime1;
    // cudaEventElapsedTime(&elapsedTime1, end0, end1);
    // printf("Time to generate IA: %3.3f ms\n", elapsedTime1);

    // Tiled_Kriging_Kernel_AOAS_Generate_B_Matrix<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, ndp, dev_ipts, idp, vm, B);
    // cudaEventRecord(end2, 0);
    // cudaEventSynchronize(end2);
    // float elapsedTime2;
    // cudaEventElapsedTime(&elapsedTime2, end1, end2);
    // printf("Time to generate B: %3.3f ms\n", elapsedTime2);

    // Tiled_Generate_Weight_Matrix<<<numBlocks, threadsPerBlock1>>>(IA, B, ndp, w);
    // cudaEventRecord(end3, 0);
    // cudaEventSynchronize(end3);
    // float elapsedTime3;
    // cudaEventElapsedTime(&elapsedTime3, end2, end3);
    // printf("Time to generate Weight: %3.3f ms\n", elapsedTime3);

    // Naive_Kriging_Kernel_AOAS_Generate_Ipts_Z<<<blocksPerGrid, threadsPerBlock>>>(dev_dpts, dev_ipts, idp, w);
    // cudaEventRecord(end4, 0);
    // cudaEventSynchronize(end4);
    // float elapsedTime4;
    // cudaEventElapsedTime(&elapsedTime4, end3, end4);
    // printf("Time to generate Z: %3.3f ms\n", elapsedTime4);

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
    printf("Tiled_Kriging_AOAS\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end0);
    cudaEventDestroy(end1);
    cudaEventDestroy(end2);
    cudaEventDestroy(end3);
    cudaEventDestroy(end4);
    cudaEventDestroy(end);

    cudaDeviceReset();
}

/** ------------------Tiled version SOA------------------ **/
void Cuda_Tiled_Kriging_SOA(REAL *dx, REAL *dy, REAL *dz, int dnum,
                            REAL *ix, REAL *iy, REAL *iz, int inum,
                            VariogramModel vm, REAL AREA)
{
    cudaEvent_t start, end0, end1, end2, end3, end4, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventCreate(&end1);
    cudaEventCreate(&end2);
    cudaEventCreate(&end3);
    cudaEventCreate(&end4);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    REAL *dev_dx, *dev_dy, *dev_dz;
    REAL *dev_ix, *dev_iy, *dev_iz;

    REAL(*A)
    [D_SIZE];
    REAL(*IA)
    [D_SIZE];
    REAL(*B)
    [D_SIZE];
    REAL(*w)
    [D_SIZE];

    cudaMalloc((void **)&dev_dx, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dy, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_dz, dnum * sizeof(REAL));
    cudaMalloc((void **)&dev_ix, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iy, inum * sizeof(REAL));
    cudaMalloc((void **)&dev_iz, inum * sizeof(REAL));

    cudaMalloc((void **)&A, dnum * dnum * sizeof(REAL));
    cudaMalloc((void **)&IA, dnum * dnum * sizeof(REAL));
    cudaMalloc((void **)&B, inum * dnum * sizeof(REAL));
    cudaMalloc((void **)&w, inum * dnum * sizeof(REAL));

    cudaMemcpy(dev_dx, dx, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dy, dy, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dz, dz, dnum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ix, ix, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iy, iy, inum * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iz, iz, inum * sizeof(REAL), cudaMemcpyHostToDevice);

    // one-dimension block and one-dimension grid
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (dnum + threadsPerBlock - 1) / threadsPerBlock;
    // two-dimension block and two-dimension grid
    dim3 threadsPerBlock1(DIM2_BLOCK_SIZE, DIM2_BLOCK_SIZE);
    dim3 numBlocks((dnum + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE, (dnum + DIM2_BLOCK_SIZE - 1) / DIM2_BLOCK_SIZE);

    printf("CUDA kernel launch with %d blocks of %d threads OR block (%d, %d) of thread (%d, %d)\n", blocksPerGrid, threadsPerBlock, numBlocks.x, numBlocks.y, threadsPerBlock1.x, threadsPerBlock1.y);

    Naive_Kriging_Kernel_SOA_Generate_A<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dnum, dev_ix, dev_iy, inum, vm, A, IA);
    cudaEventRecord(end0, 0);
    cudaEventSynchronize(end0);
    float elapsedTime0;
    cudaEventElapsedTime(&elapsedTime0, start, end0);
    printf("Time to generate A: %3.3f ms\n", elapsedTime0);

    for (int i = 0; i < dnum; i++)
    {
        nodiag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        diag_normalize<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        gaussjordan<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
        set_zero<<<numBlocks, threadsPerBlock1>>>(A, IA, dnum, i);
    }
    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    float elapsedTime1;
    cudaEventElapsedTime(&elapsedTime1, end0, end1);
    printf("Time to generate IA: %3.3f ms\n", elapsedTime1);

    Naive_Kriging_Kernel_SOA_Generate_B_Matrix<<<blocksPerGrid, threadsPerBlock>>>(dev_dx, dev_dy, dnum, dev_ix, dev_iy, inum, vm, B);
    cudaEventRecord(end2, 0);
    cudaEventSynchronize(end2);
    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, end1, end2);
    printf("Time to generate B: %3.3f ms\n", elapsedTime2);

    Tiled_Generate_Weight_Matrix<<<numBlocks, threadsPerBlock1>>>(IA, B, dnum, w);
    cudaEventRecord(end3, 0);
    cudaEventSynchronize(end3);
    float elapsedTime3;
    cudaEventElapsedTime(&elapsedTime3, end2, end3);
    printf("Time to generate Weight: %3.3f ms\n", elapsedTime3);

    Naive_Kriging_Kernel_SOA_Generate_IZ<<<blocksPerGrid, threadsPerBlock>>>(dev_dz, dev_iz, inum, w);
    cudaEventRecord(end4, 0);
    cudaEventSynchronize(end4);
    float elapsedTime4;
    cudaEventElapsedTime(&elapsedTime4, end3, end4);
    printf("Time to generate Z: %3.3f ms\n", elapsedTime4);

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
    printf("Tiled_Kriging_SOA\nTime to generate: %3.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaDeviceReset();
}

int main()
{
    VariogramModel v;
    v.Nugget = 0.1;
    v.Sill = 8.5;
    v.Range = 25;
    v.M = Gaussian;
    REAL width = 350, height = 250;
    REAL A = width * height;

    Point *dpts = new Point[D_SIZE];
    Point *ipts = new Point[I_SIZE];
    CTrgl *trgls;

    REAL *dx = new REAL[D_SIZE];
    REAL *dy = new REAL[D_SIZE];
    REAL *dz = new REAL[D_SIZE];

    REAL *ix = new REAL[I_SIZE];
    REAL *iy = new REAL[I_SIZE];
    REAL *iz = new REAL[I_SIZE];

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
    for (size_t i = 0; i < D_SIZE; i++)
    {
        REAL x, y, z;
        fin >> x >> y >> z;
        dpts[i].x = x;
        dx[i] = x;
        dpts[i].y = y;
        dy[i] = y;
        dpts[i].z = z;
        dz[i] = z;
        // fin >> dpts[i].x >> dpts[i].y >> dpts[i].z;
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
    trgls = new CTrgl[face];
    for (size_t i = 0; i < I_SIZE; i++)
    {
        REAL x, y, z;
        fin2 >> x >> y >> z;
        ipts[i].x = x;
        ix[i] = x;
        ipts[i].y = y;
        iy[i] = y;
        ipts[i].z = z;
        iz[i] = z;
        // fin2 >> ipts[i].x >> ipts[i].y >> ipts[i].z;
    }
    int num;
    for (size_t i = 0; i < face; i++)
    {
        fin2 >> num >> trgls[i].pt0 >> trgls[i].pt1 >> trgls[i].pt2;
    }
    fin2.close();
    cout << endl;

    cout << "Kriging is on running..." << endl;

    Cuda_Naive_Kriging_AOAS(dpts, D_SIZE, ipts, I_SIZE, v, A);
    // Cuda_Naive_Kriging_SOA(dx, dy, dz, D_SIZE, ix, iy, iz, I_SIZE, v, A);
    // Cuda_Tiled_Kriging_AOAS(dpts, D_SIZE, ipts, I_SIZE, v, A);

    string result_root = "./result/";
    string result;
    cout << "save the result in: ";
    cin >> result;

    /****/
    ofstream fout_base_AOAS(result_root + result);
    if (!fout_base_AOAS)
    {
        cout << "\nCannot Save File ! " << result << endl;
        exit(1);
    }
    fout_base_AOAS << "OFF" << endl;
    fout_base_AOAS << point << "  " << face << "  " << 0 << endl;
    for (size_t i = 0; i < point; i++)
    {
        fout_base_AOAS << ipts[i].x << "   " << ipts[i].y << "   " << ipts[i].z << endl;
    }
    for (size_t i = 0; i < face; i++)
    {
        fout_base_AOAS << "3    " << trgls[i].pt0 << "   " << trgls[i].pt1 << "   " << trgls[i].pt2 << endl;
    }
    fout_base_AOAS.close();
    cout << endl;

    /**
    ofstream fout_base_SOA(result_root + result);
    if (!fout_base_SOA)
    {
        cout << "\nCannot Save File ! " << result << endl;
        exit(1);
    }
    fout_base_SOA << "OFF" << endl;
    fout_base_SOA << point << "  " << face << "  " << 0 << endl;
    for (size_t i = 0; i < point; i++)
    {
        fout_base_SOA << ix[i] << "   " << iy[i] << "   " << iz[i] << endl;
    }
    for (size_t i = 0; i < face; i++)
    {
        fout_base_SOA << "3    " << trgls[i].pt0 << "   " << trgls[i].pt1 << "   " << trgls[i].pt2 << endl;
    }
    fout_base_SOA.close();
    cout << endl;
    **/

    delete[] dpts;
    delete[] ipts;
    delete[] trgls;
    delete[] dx;
    delete[] dy;
    delete[] dz;
    delete[] ix;
    delete[] iy;
    delete[] iz;

    return 0;
}
