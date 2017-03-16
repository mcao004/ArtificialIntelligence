
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "LOOValidator.h"
#include "NNClassifier.h"
#include "dataReader.h"

using namespace std;

double LO = -256;
double HI = 256;

double rando();

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


void printSubset(vector<int> subset);
void ForwardSelection(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv);
void BackwardsElimination(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv);

void test1() {
    cout << "Test 1" << endl;
    NNClassifier nnc;
    vector<Instance> v;
    vector<double> features;
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
    vector<double> features;
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
    vector<double> features;
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

double rando() {
    return LO + static_cast <double> ( rand()) / (static_cast <double> (RAND_MAX/(HI-LO)));
}

void test4() {
    cout << endl;
    cout << "Small Dataset test for classifier" << endl;
    
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

    double correct = 0.0; // stores how many we got correct 
    for (unsigned i = 0; i < allData.size(); i++) {
        //cout << i << endl;
        vector<Instance>::iterator it;
        
        Instance currentInst = allData.at(i);
        it = allData.begin() + i;
        //currentInst = *it;
        it = allData.erase(it);
        
        // use the copy as training data, and currentInst as the newData to compare it to
        double classType = nnc.classify(allData, currentInst, subset);

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
    cout << "Large Dataset test for classifier" << endl;
    
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

    double correct = 0.0; // stores how many we got correct 
    for (unsigned i = 0; i < allData.size(); i++) {
        //cout << i << endl;
        vector<Instance>::iterator it = allData.begin();
        
        Instance currentInst = allData.at(i);
        it = allData.begin() + i;
        //currentInst = *it;
        it = allData.erase(it);
        
        // use the copy as training data, and currentInst as the newData to compare it to
        double classType = nnc.classify(allData, currentInst, subset);

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

void test6() {
    cout << "Validator Test" << endl;

    string filename = "Datasets/cs_170_large80.txt";
    double accuracy = 0;
    dataReader dr(filename);
    
    // initialize the data
    vector<Instance> data = dr.read();
    NNClassifier nnc;
    // make arbitrary subset
    vector<int> subset;
    //subset.push_back(1);
    //subset.push_back(15);
    //subset.push_back(27);
    for (unsigned i = 1; i < 40; i++) {
        subset.push_back(i);
    }


    // make a leave-one-out validator and pass into it all of the data, a classifier and the subset
    LOOValidator lv;

    accuracy = lv.validate(data, nnc, subset);
    cout << "Accuracy: " << accuracy << endl;
}

void test7() {
    cout << "Forward Selection test" << endl;

    string filename = "Datasets/cs_170_small80.txt";
    dataReader dr(filename);
    vector<Instance> data = dr.read();
    NNClassifier nnc;
    LOOValidator lv;
    
    ForwardSelection(data, nnc, lv);
}

void test8() {
    cout << "Backwards Elimination test" << endl;

    string filename = "Datasets/cs_170_small80.txt";
    dataReader dr(filename);
    vector<Instance> data = dr.read();
    NNClassifier nnc;
    LOOValidator lv;
    
    BackwardsElimination(data, nnc, lv);

}

int main() {
    //srand(static_cast <unsigned> (time(0)));
    
    cout << "Pick a test(number): " << endl;
    cout << "\t1) Nearest Neighbor Classifier Test with features obj1{1,12,5} and obj2{2,0,0}" << endl;
    cout << "\t2) Nearest Neighbor Classifier Test with features obj{1,-90,42} and obj2{2,777,164}" << endl;
    cout << "\t3) Nearest Neighbor Classifier Test with features obj1{random, random} and obj2{random, random}" << endl;
    cout << "\t4) Validator Prototype with feature subset{3,5,7} on small dataset80" << endl;
    cout << "\t5) Validator Prototype with feature subset{1,15,27} on large dataset80" << endl;
    cout << "\t6) Validator Test with {1,15,27} on large80" << endl;
    cout << "\t7) Forward Selection Test on small80" << endl;
    cout << "\t8) Backwards Elimination Test with small80" << endl;
    int input;
    cin >> input;

    switch(input){
        case 1:
            test1();
            break;
        case 2:
            test2();
            break;
        case 3:
            test3();
            break;
        case 4:
            test4();
            break;
        case 5:
            test5();
            break;
        case 6:
            test6();
            break;
        case 7:
            test7();
            break;
        case 8:
            test8();
            break;
        default:
            cout << "Invalid. Proceeding with default: 8)" << endl;
            test8();
            break;
    }
    //test1();
    //test2();
    //test3();
    //test4();
    //test5();
    //test6();
    //test7();
    test8();

    return 0;
}
