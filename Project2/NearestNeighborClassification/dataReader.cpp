
#include "dataReader.h"

dataReader::dataReader(string inputfilename) {
    inputstream.open(inputfilename.c_str());
    if (!inputstream.is_open()) {
        cout << "Error opening data file: " << inputfilename << endl;
    }
}

dataReader::~dataReader() {
    inputstream.close();
}

Instance dataReader::lineToItem(string s) {
    stringstream ss(s);
    
    cout.precision(10);
    string singlef; // single feature/current feature
    vector<float> features;
    features.clear();
    while(ss >> singlef) { // for each input
        // singlef has #.#######e+-###
        // for now do not normalize,
        features.push_back(atof(singlef.substr(0,singlef.find('e')).c_str())); // substring from start to e
        // cout << features.at(features.size()-1) << " ";
        // actually I'll try to normalize
        //float exponent = atoi(singlef.substr(singlef.find('e')+1).c_str());
        //features.at(features.size()-1) *= (float)pow((float)10.0,exponent);
    }
    //cout << endl;
    
    return Instance(features);
}

vector<Instance> dataReader::read() {
    vector<Instance> result(2048,Instance());

    unsigned i = 0;
    char input[4096];
    while(!inputstream.eof()) {
        inputstream.getline(input,4096);
        //cout << "Stored in " << i << ": " << input << endl;
        result.at(i) = lineToItem(input);
        i++;
        //result.push_back(lineToItem(input));
    }
    result.resize(i);
    //result.at(result.size()-1).print();
    result.pop_back();
    return result;
}
