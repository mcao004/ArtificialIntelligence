
#ifndef __LOOVALIDATOR_H__
#define __LOOVALIDATOR_H__

#include "dataReader.h"
#include "NNClassifier.h"
#include "Instance.h"

// Leave-One-Out Validator
// only has one function
class LOOValidator{
    public:
        double validate(vector<Instance> & training,const NNClassifier &nnc,const vector<int> &subset) const;        
};    
#endif
