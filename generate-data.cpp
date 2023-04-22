#include <iostream>
#include <cstring>
#include <fstream>

#define K 1024

using namespace std;

int main()
{
    string root = "./data/";
    string file, path;
    int n;
    int face, dataSize;

    cout << "generate data points in which file:";
    cin >> file;
    path = root + file;
    cout << "data points size:";
    cin >> n;
    dataSize = (int)(n * K);
    // cin>>dataSize;
    face = dataSize * 2;
    cout << "num of faces:" << face << endl;
    // cin >> face;

    cout << "data points:" << endl;
    cout << "data file path is:" << path << endl;
    cout << "data points size is(K,1K=1024):" << n << " K," << dataSize << endl;
    cout << "num of faces is:" << face << endl;

    ofstream fout(path);
    if (!fout)
    {
        cout << "\nCannot Save File ! " << path << endl;
        exit(1);
    }
    fout << "OFF" << endl;
    fout << dataSize << "  " << face << "  " << 0 << endl;

    cout << "generating..." << endl;
    clock_t start, end;
    float cpuTime;
    start = clock();
    srand(time(NULL));
    for (int i = 0; i < dataSize; i++)
    {
        fout << rand() / (float)RAND_MAX * 1000 << "  " << rand() / (float)RAND_MAX * 1000 << "  " << 0.0f << endl;
    }
    for (int i = 0; i < face; i++)
    {
        fout << "3    " << rand() % (dataSize + 1) << "  " << rand() % (dataSize + 1) << "  " << rand() % (dataSize + 1) << endl;
    }
    end = clock();
    cout << "done!" << endl;
    cpuTime = (end - start) / (CLOCKS_PER_SEC);
    cout << "it took " << cpuTime << " s of CPU to execute this.\n";
    cout << "it took " << (end - start) << " ms of CPU to execute this.\n";

    return 0;
}