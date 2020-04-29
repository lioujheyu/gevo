#include<math.h>
#include<string>
#include<sstream>
#include<iostream>
#include<fstream>
#include<vector>
#include<exception>
#include<cstdlib>
#include<tuple>

#ifndef STANDALONE
#   include<pybind11/pybind11.h>
    namespace py = pybind11;
#endif

using namespace std;

/**
 * @in source filename for comparison
 * @in golden filename for comparison
 * @return <return code, message, max error, avg error>
 **/
tuple<int, string, double, double> file(string src, string golden, double epsilon)
{
    ifstream sfp(src);
    ifstream gfp(golden);

    if (!sfp or !gfp) {
        cout << src << " or " << golden << " cannot be opened." << endl;
        return make_tuple(-1, src + " or " + golden + " cannot be opened.", 0.0, 0.0);
    }

    string str;
    vector<string>sstrvec, gstrvec;
    while (sfp >> str)
        sstrvec.push_back(str);
    while (gfp >> str)
        gstrvec.push_back(str);

    if (sstrvec.size() != gstrvec.size())
        return make_tuple(2, "number of string mismatched.", 0.0, 0.0);

    double sval, gval;
    double maxErr=0.0, avgErr=0.0;
    // double epsilon = 0.01;
    string maxErrMsg;
    for (unsigned i=0; i<sstrvec.size(); i++) {
        // Skip the string that is float point number
        try {
            sval = stod(sstrvec[i]);
        }
        catch (invalid_argument const &e) {
            continue;
        }
        catch (out_of_range const &e) {
            sval = 0; // this will make error rate equals to 1
        }

        try {
            gval = stod(gstrvec[i]);
        }
        catch (invalid_argument const &e) {
            continue;
        }
        catch (out_of_range const &e) {
            throw; // golden cannot be out_of_range
        }

        // Check for Not a number
        if (isnan(sval)==true or isnan(gval)==true)
            return make_tuple(1, "Not a Number detected!", 0.0, 0.0);

        double err = fabs(sval - gval) / ((gval == 0)? 1 : gval);
        if (err > maxErr) {
            maxErr = (err > maxErr)? err : maxErr;
            maxErrMsg = "s: " + to_string(sval) + " <<>> g: " + to_string(gval);
        }

        avgErr += err;
    }

    avgErr = avgErr / sstrvec.size();
    if (maxErr > epsilon)
        return make_tuple(1, maxErrMsg, maxErr, avgErr);
    else
        return make_tuple(0, "pass", maxErr, avgErr);
}

#ifdef STANDALONE
int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "usage: fuzzycompare <source_file> <golden_file> [epsilon]" << endl;
        exit(-1);
    }

    string source_filename = argv[1];
    string golden_filename = argv[2];
    double epsilon = 0.01;
    if (argc == 4) {
        try {
            epsilon = stod(argv[3]);
        }
        catch (invalid_argument const &e) {
            cout << "epsilon is not a floating-point number" << endl;
            exit(-1);
        }
    }
    tuple<int, string, double, double> rc = file(source_filename, golden_filename, epsilon);
    cout << get<1>(rc) << endl;
    if (get<0>(rc) >= 0) {
        cout << "Max:" << get<2>(rc) << endl;
        cout << "Avg:" << get<3>(rc) << endl;
    }
    exit(get<0>(rc));
}
#else
PYBIND11_MODULE(fuzzycompare, m) {
    m.doc() = "Fuzzy compare"; // optional module docstring

    // By default, the allowed epsilon will be
    m.def("file", &file, "File comparison", py::arg("src"), py::arg("golden"), py::arg("epsilon") = 0.01);
}
#endif
