
#include <queue>
#include <iomanip>
#include <algorithm> // to use find on a vector
#include "dataReader.h"
#include "NNClassifier.h"
#include "LOOValidator.h"

// Function pointers the change the algorithm we are using
// Three algorithms to search through different subsets of features
// We will make the subsets based on the assumption that all the Instances in data are uniform in the number of features they have

void printSubset(vector<int> subset) {
    cout << "{";
    for (unsigned i = 0; i < subset.size(); i++) {
        cout << subset.at(i);
        if (i != subset.size()-1) {
            cout << ",";
        }
    }
    cout << "}";
}

// adds one feature each time
void ForwardSelection(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv) {
    cout << fixed << setprecision(1);
    if (data.empty()) {
        cout << "Error: given empty dataset" << endl;
        return;
    }
    unsigned featurenum = data.at(0).numFeatures();
    // keep track of best subset, parent subset, and child subsets
    vector<int> best_subset;
    vector<int> best_child;
    double best_child_acc;
    vector<int> parent_subset;
    double best_acc = 0.0;

    cout << "Beginning Search." << endl;
    
    // run the loop until the current subset contains all features
    for (unsigned l = 1; l < featurenum; l++) {
        // reinit best_child for new parent
        best_child.clear();
        // indicator we have no best child at the moment
        best_child_acc = 0;
        // for each feature not in the parent subset

        cout << endl;
        for (unsigned m = 1; m < featurenum; m++) {
            // if not in parent subset, try it
            if (find(parent_subset.begin(), parent_subset.end(), m) == parent_subset.end()) {
                // try it by adding it to copy of parent subset
                vector<int> temp_subset(parent_subset);
                temp_subset.push_back(m);
                // get accuracy for this temp_subset(current child)
                double temp_acc = lv.validate(data, nnc, temp_subset);
                
                // print out the result
                cout << "\tUsing feature(s) ";
                printSubset(temp_subset);
                cout << " accuracy is " << temp_acc*100 << " %" << endl;

                // if better than current best child
                // will update if first-seen child
                if (temp_acc > best_child_acc) {
                    // update
                    best_child = temp_subset;
                    best_child_acc = temp_acc;
                }
                // if new subset is better
                if (temp_acc > best_acc) {
                    // update
                    best_subset = temp_subset;
                    best_acc = temp_acc;
                }
            }
        }// end looking at children of parent_subset
        // print out warning if we have to
        cout << endl;
        if (best_child != best_subset) {
            cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
        }
        cout << "Feature set ";
        printSubset(best_child);
        cout << " was best, accuracy is " << best_child_acc*100 << " %" << endl;
        // assign new parent
        parent_subset = best_child;
    }

    cout << endl;
    cout << "Finished search!!! The best feature subset is ";
    printSubset(best_subset);
    cout << ", which has an accuracy of " << best_acc*100 << " %" << endl;
    cout << endl;

}

// removes one feature each time
void BackwardsElimination(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv) {
    cout << fixed << setprecision(1);
    if (data.empty()) {
        cout << "Error: given empty dataset" << endl;
        return;
    }
    unsigned featurenum = data.at(0).numFeatures();
    // keep track of best subset, parent subset, and child subsets
    vector<int> best_subset;
    vector<int> best_child;
    double best_child_acc;
    vector<int> parent_subset;
    // start with full subset
    for (unsigned i = 1; i < featurenum; i++) {
        parent_subset.push_back(i);
    }
    double best_acc = 0.0;

    cout << "Beginning Search." << endl;
    
    for (unsigned l = 1; l < featurenum-1; l++) {
        // if l = 1, need to check the full subset
        if (l == 1) {
            best_acc = lv.validate(data, nnc, parent_subset);

            cout << "\tUsing feature(s) ";
            printSubset(parent_subset);
            cout << " accuracy is " << best_acc*100 << " %" << endl;
        }
        
        // reinit best_child for new parent
        best_child.clear();
        // indicator we have no best child at the moment
        best_child_acc = 0;
        // for each feature not in the parent subset

        cout << endl;
        // m is the index of the feature to remove
        for (unsigned m = 0; m < parent_subset.size(); m++) {
            vector<int> temp_subset(parent_subset);
            temp_subset.erase(temp_subset.begin() + m);
            
            //get accuracy for this subset
            double temp_acc = lv.validate(data, nnc, temp_subset);

            cout << "\tUsing feature(s) ";
            printSubset(temp_subset);
            cout << " accuracy is " << temp_acc*100 << " %" << endl;

            // if better than current best child
            // will update if first-seen child
            if (temp_acc >= best_child_acc) {
                // update
                best_child = temp_subset;
                best_child_acc = temp_acc;
            }
            // if new subset is better
            if (temp_acc >= best_acc) {
                // update
                best_subset = temp_subset;
                best_acc = temp_acc;
            }
        }// end looking at children of parent_subset
        // print out warning if we have to
        cout << endl;
        if (best_child != best_subset) {
            cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
        }
        cout << "Feature set ";
        printSubset(best_child);
        cout << " was best, accuracy is " << best_child_acc*100 << " %" << endl;
        // assign new parent
        parent_subset = best_child;
    }

    cout << endl;
    cout << "Finished search!!! The best feature subset is ";
    printSubset(best_subset);
    cout << ", which has an accuracy of " << best_acc*100 << " %" << endl;
    cout << endl;
}

// Forward Selection Algorithm with branching
// how would I know at this moment?
void MyAlgo(vector<Instance> &data, const NNClassifier &nnc, const LOOValidator &lv) {
    cout << fixed << setprecision(1);
    if (data.empty()) {
        cout << "Error: given empty dataset" << endl;
        return;
    }
    unsigned featurenum = data.at(0).numFeatures();
    // keep track of best subset, parent subset, and child subsets
    queue< vector<int> > Q; // will take in a set of children at end of loop
    vector<int> best_subset;
    vector<vector<int> > best_children;
    double best_child_acc;
    vector<int> parent_subset;
    double best_acc = 0.0;

    cout << "Beginning Search." << endl;

    Q.push(best_subset);
    
    // run the loop until the current subset contains all features
    while(!Q.empty()) {
        // take top of queue
        parent_subset = Q.front();
        Q.pop();
        // parent will serve as the base for its children

        // reinit best_children
        best_children.clear();
        // indicator we have no best child at the moment
        best_child_acc = 0;
        // for each feature not in the parent subset

        cout << endl;
        for (unsigned m = 1; m < featurenum; m++) {
            // if not in parent subset, try it
            if (find(parent_subset.begin(), parent_subset.end(), m) == parent_subset.end()) {
                // try it by adding it to copy of parent subset
                vector<int> temp_subset(parent_subset);
                temp_subset.push_back(m);
                // get accuracy for this temp_subset(current child)
                double temp_acc = lv.validate(data, nnc, temp_subset);
                
                // print out the result
                cout << "\tUsing feature(s) ";
                printSubset(temp_subset);
                cout << " accuracy is " << temp_acc*100 << " %" << endl;

                // if better than current best child
                // will also update if first-seen child
                if (temp_acc > best_child_acc) {
                    // dump set of best_children
                    best_children.clear();
                    best_children.push_back(temp_subset);
                    best_child_acc = temp_acc;
                } else if (temp_acc == best_child_acc) {
                    // if same also put into the best_children
                    best_children.push_back(temp_subset);
                }
                // if new subset is better
                if (temp_acc > best_acc) {
                    // update
                    best_subset = temp_subset;
                    best_acc = temp_acc;
                }
            }
        }// end looking at children of parent_subset
        // print out warning if we have to
        cout << endl;
        if (!best_children.empty() && best_children.at(0) != best_subset) {
            cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
        }
        cout << "Feature set(s) ";
        for (unsigned k = 0; k < best_children.size(); k++) {
            // push best_children into the queue
            Q.push(best_children.at(k));
            printSubset(best_children.at(k));
            if (k != best_children.size()-1) {
                cout << ", ";
            }
        }
        cout << " was best, accuracy is " << best_child_acc*100 << " %" << endl;

    }
    
    cout << endl;
    cout << "Finished search!!! The best feature subset is ";
    printSubset(best_subset);
    cout << ", which has an accuracy of " << best_acc*100 << " %" << endl;
    cout << endl;  

}


