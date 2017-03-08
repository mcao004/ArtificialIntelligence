
#include <ctime>
#include <cstdlib>

#include "NNClassifier.h"
#include "dataReader.h"

using namespace std;

float LO = -256;
float HI = 256;

float rando();

void printInstance(Instance i) {
    cout << "Instance" << endl;
    cout << "\tClass: " << i.getClass() << endl;
    cout << "\tFeatures: ";
    for (unsigned j = 1; j < i.numFeatures(); j++) {
        cout << i.at(j) << ", ";
    }
    cout << endl;
    cout << "End Instance" << endl;
}   

void test1() {
    cout << "Test 1" << endl;
    NNClassifier nnc;
    vector<Instance> v;
    vector<float> features;
    vector<int> subset;

    subset.push_back(1);
    subset.push_back(2);

    features.push_back(1);
    features.push_back(12);
    features.push_back(5);
    v.push_back(Instance(features));
    
    printInstance(v.at(0));
    
    features.at(0) = 2;
    features.at(1) = 0;
    features.at(2) = 0;
    v.push_back(Instance(features));

    printInstance(v.at(1));


    cout << "Distance: " << nnc.dist(v.at(0), v.at(1), subset) << endl;
    cout << endl;
}

void test2() {
    cout << "Test 2" << endl;
    NNClassifier nnc;
    vector<Instance> v;
    vector<float> features;
    vector<int> subset;

    subset.push_back(1);
    subset.push_back(2);
    features.push_back(1);
    features.push_back(-90);
    features.push_back(42);
    v.push_back(Instance(features));

    printInstance(v.at(0));

    features.at(0) = 2;
    features.at(1) = 777;
    features.at(2) = 164;
    v.push_back(Instance(features));

    printInstance(v.at(1));

    cout << "Distance: " << nnc.dist(v.at(0), v.at(1),subset) << endl;
    cout << endl;
}

void test3() {
    cout << "Test 3" << endl;
    NNClassifier nnc;
    vector<Instance> v;
    vector<float> features;
    vector<int> subset;

    subset.push_back(1);
    subset.push_back(2);
    features.push_back(1);
    features.push_back(rando());
    features.push_back(rando());

    v.push_back(Instance(features));

    printInstance(v.at(0));

    features.at(0) = 2;
    features.at(1) = 0; 
    features.at(2) = 0;
    v.push_back(Instance(features));

    printInstance(v.at(1));

    cout << "Distance: " << nnc.dist(v.at(0), v.at(1), subset) << endl;
    cout << endl;
}

float rando() {
    return LO + static_cast <float> ( rand()) / (static_cast <float> (RAND_MAX/(HI-LO)));
}

void test4() {
    cout << endl;
    cout << "Small Dataset test" << endl;
    
    string inputfilename = "Datasets/cs_170_small80.txt";
    
    NNClassifier nnc;
    vector<Instance> allData;
    dataReader dr(inputfilename);
    
    allData = dr.read();

    // now have all of my data premade from the input file
    // initialize subset
    vector<int> subset;
    subset.push_back(7);
    subset.push_back(5);
    subset.push_back(3);

    float correct = 0.0; // stores how many we got correct 
    for (int i = 0; i < allData.size(); i++) {
        //cout << i << endl;
        vector<Instance>::iterator it;
        
        Instance currentInst = allData.at(i);
        it = allData.begin() + i;
        //currentInst = *it;
        it = allData.erase(it);
        
        // use the copy as training data, and currentInst as the newData to compare it to
        float classType = nnc.classify(allData, currentInst, subset);

        if (classType == currentInst.getClass()) {
            correct++;
        }

        // insert the currentInst back into allData
        it = allData.insert(it, currentInst);
    }
    cout << "Number of Correct Classifications: " << correct << endl;
    correct /= (allData.size()-1); // since not compared to itself
    cout << "Accuracy: " << correct << endl;

}

void test5() {
    cout << endl;
    cout << "Large Dataset test" << endl;
    
    string inputfilename = "Datasets/cs_170_large80.txt";
    
    NNClassifier nnc;
    vector<Instance> allData;
    dataReader dr(inputfilename);
    
    allData = dr.read();


    // now have all of my data premade from the input file
    // initialize subset
    vector<int> subset;
    subset.push_back(27);
    subset.push_back(15);
    subset.push_back(1);

    float correct = 0.0; // stores how many we got correct 
    for (int i = 0; i < allData.size(); i++) {
        //cout << i << endl;
        vector<Instance>::iterator it = allData.begin();
        
        Instance currentInst = allData.at(i);
        it = allData.begin() + i;
        //currentInst = *it;
        it = allData.erase(it);
        
        // use the copy as training data, and currentInst as the newData to compare it to
        float classType = nnc.classify(allData, currentInst, subset);

        if (classType == currentInst.getClass()) {
            correct++;
        }

        // insert the currentInst back into allData
        it = allData.insert(it, currentInst);
    }
    cout << "Number of Correct Classifications: " << correct << endl;
    correct /= (allData.size()-1); // since not compared to itself
    cout << "Accuracy: " << correct << endl;

}

int main() {
    //srand(static_cast <unsigned> (time(0)));
    
    //test1();
    //test2();
    //test3();
    test4();
    test5();

    return 0;
}
