
#include <iostream>
#include "NNClassifier.h"

float NNClassifier::classify(const vector<Instance> &training, const Instance &newData,const vector<int> &subset) const {
    if (training.empty()) {
        cout << "This training set is empty" << endl;
        return -1;
    }
    unsigned i = 0;
    float closestDist = FLT_MAX;
    float d = FLT_MAX;
    // only have this in case I want it to return the closest Instance
    unsigned closestInstance = -1;
    
    for (i = 0; i < training.size(); i++) {
        // for each training point, find dist
        d = dist(training.at(i), newData, subset);
        if (d < closestDist) {
            // if better distance, save this point
            closestDist = d;
            closestInstance = i;
        }
        // d = 0; // don't really need
    }

    // if for some reason the currentclass is 0
    if (training.at(closestInstance).getClass() == 0) {
        cout << "Error: did not find a class" << endl;
        return -1;
    }

    return training.at(closestInstance).getClass();
}

// Euclidean
float NNClassifier::dist(const Instance &i1, const Instance &i2, const vector<int> &subset) const {
    // if same number of features, go ahead
    float sum = 0.0;
    unsigned i;
    for(i = 0; i < subset.size(); i++) {
        // (x1 - x2)^2
        sum += (i1.at(subset.at(i)) - i2.at(subset.at(i))) * (i1.at(subset.at(i)) - i2.at(subset.at(i)));
    }
    return sqrt(sum);
}

