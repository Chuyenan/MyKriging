#include <iostream>
#include <vector>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#define SINGLE_PRECISION
#ifdef SINGLE_PRECISION
#define REAL float
#else
#define REAL double
#endif

// #define D_SIZE 1024
// #define I_SIZE 1024
long long int D_SIZE, I_SIZE;

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
typedef struct CVert
{
    REAL x, y, z;
    CVert(){};
} CVert;

typedef struct CTrgl
{
    int pt0, pt1, pt2;
    CTrgl(){};
} CTrgl;

// executes  same row transformation for matrix_A and matrix_B
void Row_Transform_Matrix(vector<vector<REAL>> &matrix_A, vector<vector<REAL>> &matrix_B, int target_row, int source_row, int matrix_size)
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

// Inverts a Matrix using gauss-jordan reduction method
vector<vector<REAL>> Inverse_Matrix(vector<vector<REAL>> original_matrix, int matrix_size)
{
    vector<vector<REAL>> inversed_matrix(matrix_size, vector<REAL>(matrix_size, 0));

    REAL row = matrix_size, col = matrix_size;

    for (size_t i = 0; i < matrix_size; i++)
        inversed_matrix[i][i] = 1.0;

    // converts original_matrix to Identity matrix, gets the result matrix: inversed_matrix
    for (size_t i = 0; i < matrix_size; i++)
    {
        // converts original_matrix[i][i] to 1.0, does the same transformation to inversed_matrix
        Row_Transform_Matrix(original_matrix, inversed_matrix, i, i, matrix_size);
        // converts original_matrix[j][i] (i≠j) to 0.0 using row-transformation, does the same transformation to inversed_matrix
        for (size_t j = 0; j < matrix_size; j++)
        {
            if (j != i)
                Row_Transform_Matrix(original_matrix, inversed_matrix, j, i, matrix_size);
        }
    }

    return inversed_matrix;
}

// multiplies matrix A with matrix B, A × B
vector<REAL> Multiply_Matrix(vector<vector<REAL>> martix_A, int A_row, int A_col, vector<REAL> martix_B, int B_row, int B_col)
{
    vector<REAL> product_matrix = vector<REAL>(A_row, 0);

    for (size_t i = 0; i < A_row; i++)
    {
        product_matrix[i] = 0.0;
        for (size_t k = 0; k < A_col; k++)
        {
            product_matrix[i] += (martix_A[i][k] * martix_B[k]);
        }
    }

    return product_matrix;
}

// Model based on http://spatial-analyst.net/ILWIS/htm/ilwisapp/sec/semivar_models_sec.htm
// Calculate semivariance
REAL Calculate_semivariance(VariogramModel vm, REAL distance)
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

// Kriging Algorithm
void Ordinary_Kriging(CVert *dpts, int ndp, // Data points , known
                      CVert *ipts, int idp, // Interpolation point, unknown
                      VariogramModel vm,    // Data Model,
                      REAL AREA)            // Area of planar region
{
    // initialize Semivariance matrix for all known points
    vector<vector<REAL>> A_semivariance_matrix(ndp, vector<REAL>(ndp, 0));
    // initialize inverse matrix A
    vector<vector<REAL>> inversed_A(ndp, vector<REAL>(ndp, 0));
    // initialize Semivariance matrix for all unknown points
    vector<REAL> B_semivariance_matrix(ndp, 0);
    // initialize weight matrix (x) for all unknown points
    vector<REAL> weight_martix(ndp, 0);
    // A*x=b, x=inversed_A*b

    REAL distance = 0.0, semivariance = 0.0;

    // generate A_semivariance_matrix
    for (size_t i = 0; i < ndp; i++)
    {
        for (size_t j = 0; j < ndp; j++)
        {
            distance = sqrt(pow(dpts[i].x - dpts[j].x, 2.0) + pow(dpts[i].y - dpts[j].y, 2.0));
            A_semivariance_matrix[i][j] = Calculate_semivariance(vm, distance);
        }
    }

    // Inverse Matrix A_semivariance_matrix
    inversed_A = Inverse_Matrix(A_semivariance_matrix, ndp);

    // generate semivariance matrix for each unknown point, separately
    // calculate weight martix (x) for each unknown point, separately
    // calculate value z for each unknown point, separately
    for (size_t i = 0; i < idp; i++)
    {
        for (size_t j = 0; j < ndp; j++)
        {
            distance = sqrt(pow(ipts[i].x - dpts[j].x, 2.0) + pow(ipts[i].y - dpts[j].y, 2.0));
            B_semivariance_matrix[j] = Calculate_semivariance(vm, distance);
        }
        weight_martix = Multiply_Matrix(inversed_A, ndp, ndp, B_semivariance_matrix, ndp, 1);

        ipts[i].z = 0.0;
        for (size_t k = 0; k < ndp; k++)
        {
            ipts[i].z += (weight_martix[k] * dpts[k].z);
        }
    }
}

// test
/*
int main(void)
{
    VariogramModel v;
    v.Nugget = 0.1;
    v.Sill = 8.5;
    v.Range = 25;
    v.M = Gaussian;
    CVert *dpts = new CVert[D_SIZE];
    CVert *ipts = new CVert[I_SIZE];
    REAL width = 350, height = 250;
    REAL A = width * height;

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
    clock_t start;
    double during;
    start = clock();
    Ordinary_Kriging(dpts, D_SIZE, ipts, I_SIZE, v, A);
    during = double(clock() - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU Kriging\nTime to generate: %3.3f ms\n", during);
    // cout << ipts[0].z << " " << ipts[1].z << " " << ipts[2].z << endl;
    for (int i = 0; i < I_SIZE; i++)
    {
        cout<< ipts[i].z<<" ";
    }
    cout << endl;

    // vector<vector<REAL>> a = vector<vector<REAL>>(3, vector<REAL>(2, 1));
    // vector<vector<REAL>> b = vector<vector<REAL>>(2, vector<REAL>(2, 1));
    // vector<REAL> c = vector<REAL>(2, 1);
    // a[0][0] = 1;
    // a[0][1] = 2;
    // a[1][0] = 3;
    // a[1][1] = 4;
    // a[2][0] = 5;
    // a[2][1] = 6;
    // // cout << v.M;
    // // a = Inverse_Matrix(a, 2);
    // c = Multiply_Matrix(a, 3, 2, c, 2, 1);
    // // Row_Transform_Matrix(a, b, 1, 0, 2);
    // for (int i = 0; i < 3; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // for (int i = 0; i < 3; i++)
    // {
    //     cout << c[i] << endl;
    // }

    return 0;
}
*/

int main()
{

    CVert *dpts;
    CVert *ipts;
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
    D_SIZE = point;
    dpts = new CVert[D_SIZE];
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
    I_SIZE = point;
    ipts = new CVert[I_SIZE];
    trgls = new CTrgl[face];
    for (size_t i = 0; i < D_SIZE; i++)
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
    REAL width = 1000, height = 1000;
    REAL A = width * height;

    clock_t start, end;
    float cpuTime;
    cout << "Kriging is on running..." << endl;
    start = clock();
    Ordinary_Kriging(dpts, D_SIZE, ipts, I_SIZE, v, A);
    end = clock();
    cpuTime = double(end - start) / CLOCKS_PER_SEC * 1000;
    printf("CPU Kriging\nTime to generate: %3.3f ms\n", cpuTime);

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
    fout << I_SIZE << "  " << face << "  " << 0 << endl;
    for (size_t i = 0; i < I_SIZE; i++)
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