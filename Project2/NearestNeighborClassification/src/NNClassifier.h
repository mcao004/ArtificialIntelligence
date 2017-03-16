#ifndef __NNCLASS_H__
#define __NNCLASS_H__

#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include "Instance.h"

using namespace std;

class NNClassifier {
    public:
        // returns the class the newData would belong to based on training data
        double classify(const vector<Instance> &training,const Instance &newData,const vector<int> &subset) const;        
        // returns the Euclidean distance of the newData from the training data
        double dist(const Instance &i1,const Instance &i2,const vector<int> &subset) const;
};
#endif
