#include <stdio.h>

void quickSort(int arr[], int low, int high)
{


    int first = low;
    int last  = high;
    int key = arr[first];
    if(low >= high)
        return;
    while(first < last)
    {
        while(first < last && arr[last] < key)
        {
            last--;
        }
        arr[first] = arr[last];
        while(first < last && arr[first] > key)
        {
            first++;
        }
        arr[last] = arr[first];
    }
    arr[first] = key;

    quickSort(arr, low, first-1);
    quickSort(arr, first+1, high);
}

// int Division(double list[], int left, int right, int mindexArr[])
// {
//     double baseNum = list[left];
//     int baseIndex = mindexArr[left];
//     while(left < right)
//     {
//         while(left < right && list[right] <= baseNum)
//         {
//             right = right -1;
//         }
//         list[left] = list[right];
//         mindexArr[left] = mindexArr[right];
//         while(left < right && list[left] >= baseNum)
//         {
//             left = left +1;
//         }
//         list[right] = list[left];
//         mindexArr[right] = mindexArr[left];
//     }
//     list[left] = baseNum;
//     mindexArr[left] = baseIndex;
//     return left;
// }

int Division(double list[], int left, int right, int mindexArr[])
{
    double baseNum = list[left];
    int baseIndex = mindexArr[left];
    while(left < right)
    {
        while(left < right && list[right] <= baseNum)
        {
            right = right -1;
        }
        list[left] = list[right];
        mindexArr[left] = mindexArr[right];
        while(left < right && list[left] >= baseNum)
        {
            left = left +1;
        }
        list[right] = list[left];
        mindexArr[right] = mindexArr[left];
    }
    list[left] = baseNum;
    mindexArr[left] = baseIndex;
    return left;
}



void QuickSort(double list[], int left, int right, int mindexArr[])
{
    if(left < right)
    {
        int i = Division(list, left, right, mindexArr);
        QuickSort(list, left, i-1, mindexArr);
        QuickSort(list , i+1, right, mindexArr);
    }
}


int main()
{
    int i;
    int a[10] = {3, 1, 111, 5, 8, 2, 0, 9, 103, 81};
    int index[10]= {0,1,2,3,4,5,6,7,8,9};


    for(i = 0; i < 10; i++)
        printf("%d ", a[i]);
    printf("\n");

    quickSort(a, 0, 9);

    for(i = 0; i < 10; i++)
        printf("%d , index = %d\n", a[i], index[i]);
    printf("\n");

    double b[10] = {3.0, 1.0, 111.0, 5.0, 8.0, 2.0, 0.0, 9.0, 103.0, 81.0};
    QuickSort(b, 0, 9, index);
    for(i = 0; i < 10; i++)
    {
        printf("QuickSort after index = %d, b[i] = %f\n", index[i], b[i]);
    }

    return 0;
}

