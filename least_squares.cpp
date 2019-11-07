// #include "pch.h"
#include <iostream>

// #include <windows.h>
#include <vector>   
using namespace std;
#include "math.h"   

double test1[24][3]
{
{-735, -312, 193},
{ -252    ,-298    ,179 },
{ 290    ,-264    ,164 },
{ 657    ,-252    ,154 },
{ -753    ,82     ,193 },
{ -92    ,27        ,175 },
{ 656    ,31        ,153 },
{ -726    ,389    ,193 },
{ -27    ,355    ,173 },
{ 652    ,413    ,153 },
{ -922    ,-306    ,199 },
{ -915    ,-114    ,199 },
{ -791    ,87        ,195 },
{ -729    ,390    ,194 },
{ 24        ,422    ,172 },
{ 1        ,273    ,173 },
{ 10        ,168    ,172 },
{ -2        ,3        ,172 },
{ -16    ,-130    ,173 },
{ -22    ,-292    ,173 },
{ 615    ,-342    ,155 },
{ 724    ,-137    ,152 },
{ 712    ,64        ,152 },
{ 728    ,359    ,151 },
};

double test[24][3] =
{
{-735    ,-312    ,6},
{-252    ,-298    ,6},
{290    ,-264    ,7},
{657    ,-252    ,7},
{-753    ,82     ,15 },
{-92    ,27        ,15 },
{656    ,31        ,14 },
{-726    ,389    ,24 },
{-27    ,355    ,25 },
{652    ,413    ,26 },
{-922    ,-306    ,4},
{-915    ,-114    ,10},
{-791    ,87        ,14 },
{-729    ,390    ,22 },
{24        ,422    ,25 },
{1        ,273    ,21 },
{10        ,168    ,18 },
{-2        ,3        ,13 },
{-16    ,-130    ,11},
{-22    ,-292    ,6},
{615    ,-342    ,5},
{724    ,-137    ,10},
{712    ,64        ,15},
{728    ,359    ,21},

};

struct Point3D {
    double x;
    double y;
    double z;
};


void column_principle_gauss(int N, double **a)
{
    int k = 0, i = 0, r = 0, j = 0;
    double t;
    for (k = 0; k < N - 1; k++)
    {
        for (i = k; i < N; i++)
        {
            r = i;
            t = (double)fabs(a[r][k]);
            if (fabs(a[i][k]) > t)
            {
                r = i;
            }
        }
        if (a[r][k] == 0)
        {
            break;
        }
        for (j = k; j < N + 1; j++)
        {
            t = a[r][j];
            a[r][j] = a[k][j];
            a[k][j] = t;
        }
        for (i = k + 1; i < N; i++)
        {
            for (j = k + 1; j < N + 1; j++)
            {
                a[i][j] = a[i][j] - a[i][k] / a[k][k] * a[k][j];
            }
        }
    }

    double he = 0;
    for (k = N - 1; k >= 0; k--)
    {
        he = 0;
        for (j = k + 1; j < N; j++)
        {
            he = he + a[k][j] * a[j][N];
        }
        a[k][N] = (a[k][N] - he) / a[k][k];
    }
}


void Least_squares(vector<Point3D>&v_Point, double M[3])
{
    double **c = NULL;
    c = new double*[3];
    for (int i = 0; i < 3; i++)
    {
        c[i] = new double[4];
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            c[i][j] = 0;
        }
    }
    c[0][0] = v_Point.size();
    for (int i = 0; i < v_Point.size(); i++)
    {
        c[0][1] = c[0][1] + v_Point.at(i).x;
        c[0][2] = c[0][2] + v_Point.at(i).y;
        c[0][3] = c[0][3] + v_Point.at(i).z;
        c[1][1] = c[1][1] + v_Point.at(i).x*v_Point.at(i).x;
        c[1][2] = c[1][2] + v_Point.at(i).x*v_Point.at(i).y;
        c[1][3] = c[1][3] + v_Point.at(i).x*v_Point.at(i).z;
        c[2][2] = c[2][2] + v_Point.at(i).y*v_Point.at(i).y;
        c[2][3] = c[2][3] + v_Point.at(i).y*v_Point.at(i).z;
    }
    c[1][0] = c[0][1];
    c[2][0] = c[0][2];
    c[2][1] = c[1][2];

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%f", c[i][j]);
        }
        printf("\n");
    }
    column_principle_gauss(3, c);

    for (int i = 0; i < 3; i++)
    {
        M[i] = c[i][3];
    }

    for (int i = 0; i < 3; i++)
    {
        delete[]c[i];
        c[i] = NULL;
    }
    delete[]c;
    c = NULL;
}

int  main()
{
    Point3D temp[23];
    for (int i = 0; i < 24; i++)
    {
        
            temp[i].x = test[i][0];
            temp[i].y = test[i][1];
            temp[i].z = test[i][2];
    }

    vector<Point3D>v_Point;
    double M[3];
    for (int i = 0; i < 24; i++)
    {
        v_Point.push_back(temp[i]);
    }

    Least_squares(v_Point, M);

    for (int i = 0; i < 3; i++)
    {
        printf("M%d = %lf\n", i, M[i]);
    }

    Sleep(100000);
    return 0;
}