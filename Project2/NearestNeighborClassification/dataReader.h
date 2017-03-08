#ifndef __DATAREADER_H__
#define __DATAREADER_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include "Instance.h"
#include <cstdlib>
#include <cmath>

using namespace std;

class dataReader {
    private:
        fstream inputstream;
    public:
        dataReader(string inputfilename);
        ~dataReader();
        Instance lineToItem(string s);
        vector<Instance> read();
};
#endif
