#include<math.h>
#include<string>
#include<sstream>
#include<iostream>
#include<fstream>
#include<vector>
#include<exception>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "usage: fuzzycompare <source_file> <golden_file>" << endl;
        exit(-1);
    }

    ifstream sfp(argv[1]);
    ifstream gfp(argv[2]);

    if (!sfp or !gfp)
        cout << argv[1] << " or " << argv[2] << " cannot be opened." << endl;

    string str;
    vector<string>sstrvec, gstrvec;
    while (sfp >> str)
        sstrvec.push_back(str);
    while (gfp >> str)
        gstrvec.push_back(str);

    if (sstrvec.size() != gstrvec.size()) {
        cout << 2 << endl;
        exit(2);
    }

    float sfloat;
    float gfloat;
    for (int i=0; i<sstrvec.size(); i++) {
        // Skip the string that is float point number
        try {
            sfloat = stof(sstrvec[i]);
            gfloat = stof(gstrvec[i]);
        }
        catch (std::invalid_argument const &e) {
            continue;
        }

        // Check for Not a number
        if (isnan(sfloat)==true or isnan(gfloat)==true) {
            cout << "Not a Number detected!";
            exit(1);
        }

        // Main part for comparison by determining the esplon
        float esplon = gfloat * 0.01;
        if (fabs(sfloat - gfloat) > esplon) {
            cout << "s: " << sfloat << " <<>> g: " << gfloat << endl;
            exit(1);
        }
    }

    exit(0);
}
