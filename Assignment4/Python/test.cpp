#include <iostream>
#include <stdlib.h>

union test
{
    int a;
    char *ptr;
};


int main()
{
    union test a ;
    a.a=10;
    char p='J';
    a.ptr=(char*)malloc(sizeof(char)*10);

    std::cout<<a.a;
    return 0;
}