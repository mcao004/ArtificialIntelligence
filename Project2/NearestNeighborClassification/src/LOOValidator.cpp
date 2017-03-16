
#include "LOOValidator.h"

// takes in training data, feature subset, and the classifier
// returns accuracy
double LOOValidator::validate(vector<Instance> & training, const NNClassifier &nnc, const vector<int> & subset) const {
    double correct = 0.0;
    // loop to leave one out and classify
    for (unsigned i = 0; i < training.size(); i++) {
        vector<Instance>::iterator it = training.begin();

        // temporarily remove the ith instance
        Instance currentInst = training.at(i);
        it = training.begin() + i;
        it = training.erase(it);

        // use the rest of the data to classify this instance
        double classType = nnc.classify(training, currentInst, subset);
        
        if (classType == currentInst.getClass()) {
            correct++;
        }

        it = training.insert(it, currentInst);
    }
    //cout << "Number of Correct Classifications: " << correct << " out of " << training.size() << endl;
    correct /= training.size();
    return correct;
}
