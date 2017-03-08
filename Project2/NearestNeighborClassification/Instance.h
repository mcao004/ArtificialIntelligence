#ifndef __INSTANCE_H__
#define __INSTANCE_H__

#include <iostream>
#include <vector>

using namespace std;

// object to hold the currently relevant features
// features {1,3,5} can be held at features.at(0, 1 , or 2)
class Instance {
    private:
        vector<float> features;
    public: 
        // instance with single double of value 0 (class 0)
        Instance() :features(1,0) {}
        Instance(vector<float> features)
            :features(features) {}
        // returns the number of features this instance has
        unsigned numFeatures() const {
            return features.size();
        }
        float at(unsigned i) const {
            return features.at(i);
        }
        int getClass() const{
            return features.at(0);
        }
        void print() const{
            for (int i = 0 ; i < features.size(); i++) {
                cout << features.at(i) << " ";
            } cout << endl;
        }
};

#endif
