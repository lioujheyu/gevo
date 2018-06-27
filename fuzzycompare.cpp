#include<math.h>
#include<string>
#include<sstream>
#include<iostream>
#include<fstream>
#include<vector>
#include<exception>
#include<cstdlib>
#include<tuple>

#include<pybind11/pybind11.h>

using namespace std;

tuple<int, string> file(string src, string golden)
{
    ifstream sfp(src);
    ifstream gfp(golden);

    if (!sfp or !gfp) {
        cout << src << " or " << golden << " cannot be opened." << endl;
        return make_tuple(-1, src + " or " + golden + " cannot be opened.");
    }

    string str;
    vector<string>sstrvec, gstrvec;
    while (sfp >> str)
        sstrvec.push_back(str);
    while (gfp >> str)
        gstrvec.push_back(str);

    if (sstrvec.size() != gstrvec.size())
        return make_tuple(-2, "number of string mismatched.");

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
        if (isnan(sfloat)==true or isnan(gfloat)==true)
            return make_tuple(1, "Not a Number detected!");

        // Main part for comparison by determining the esplon
        float esplon = fabs(gfloat * 0.01);
        if (fabs(sfloat - gfloat) > esplon) {
            string msg = "s: " + to_string(sfloat) + " <<>> g: " + to_string(gfloat);
            return make_tuple(1, msg);
        }
    }

    return make_tuple(1, "");
    // return 0;
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
    tuple<int, string> rc = file(source_filename, golden_filename);
    cout << get<1>(rc) << endl;
    exit(get<0>(rc));
}
#else
PYBIND11_MODULE(fuzzycompare, m) {
    m.doc() = "Fuzzy compare"; // optional module docstring

    m.def("file", &file, "File comparison");
}
#endif
