
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "LOOValidator.h"
#include "NNClassifier.h"
#include "dataReader.h"

using namespace std;

void printSubset(vector<int> subset);
void ForwardSelection(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv);
void BackwardsElimination(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv);
void MyAlgo(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv);


int main() {
    cout << fixed << setprecision(2);
    cout << "Welcome to Marvin Cao's Feature Selection Algorithm." << endl;
    cout << "Type in the name of the file to test: ";

    string filename;
    int algorithm;

    cin >> filename;
    filename = "Datasets/" + filename;
    cout << endl;
    
    cout << "Type the number of the algorithm you want to run." << endl;
    cout << endl;
    cout << "\t1) Forward Selection" << endl;
    cout << "\t2) Backwards Elimination" << endl;
    cout << "\t3) Marvin's Special Algorithm" << endl;
    cout << endl;
    cout << "\t\t\t";
    cin >> algorithm;

    dataReader dr(filename);
    vector<Instance> data = dr.read();
    NNClassifier nnc;
    LOOValidator lv;

    switch(algorithm) {
        case 1: 
            ForwardSelection(data, nnc, lv);
            break;
        case 2:
            BackwardsElimination(data, nnc, lv);
            break;
        case 3:
            MyAlgo(data, nnc, lv);
            break;
        default:
            cout << "no" << endl;
    } 
}
