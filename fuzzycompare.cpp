#include<math.h>
#include<string>
#include<sstream>
#include<iostream>
#include<fstream>
#include<vector>
#include<exception>
#include<cstdlib>

#include<pybind11/pybind11.h>

using namespace std;

int file(string src, string golden)
{
    ifstream sfp(src);
    ifstream gfp(golden);

    if (!sfp or !gfp) {
        cout << src << " or " << golden << " cannot be opened." << endl;
        return -1;
    }

    string str;
    vector<string>sstrvec, gstrvec;
    while (sfp >> str)
        sstrvec.push_back(str);
    while (gfp >> str)
        gstrvec.push_back(str);

    if (sstrvec.size() != gstrvec.size()) {
        cout << 2 << endl;
        return 2;
    }

    double sfloat;
    double gfloat;
    for (unsigned i=0; i<sstrvec.size(); i++) {
        // Skip the string that is float point number
        try {
            sfloat = stod(sstrvec[i]);
            gfloat = stod(gstrvec[i]);
        }
        catch (std::invalid_argument const &e) {
            continue;
        }

        // Check for Not a number
        if (isnan(sfloat)==true or isnan(gfloat)==true) {
//            cout << "Not a Number detected!";
            return 1;
        }

        // Main part for comparison by determining the esplon
        float esplon = fabs(gfloat * 0.01);
        if (fabs(sfloat - gfloat) > esplon) {
//            cout << "s: " << sfloat << " <<>> g: " << gfloat << endl;
            return 1;
        }
    }

    return 0;
}

#ifdef STANDALONE
int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "usage: fuzzycompare <source_file> <golden_file>" << endl;
        exit(-1);
    }

    string source_filename = argv[1];
    string golden_filename = argv[2];
    int rc = file(source_filename, golden_filename);
    exit(rc);
}
#else
PYBIND11_MODULE(fuzzycompare, m) {
    m.doc() = "Fuzzy compare"; // optional module docstring

    m.def("file", &file, "File comparison");
}
#endif
