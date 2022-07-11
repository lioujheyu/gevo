#include<math.h>
#include<string>
#include<sstream>
#include<iostream>
#include<fstream>
#include<vector>
#include<exception>
#include<cstdlib>
#include<tuple>
#include<climits>

#ifndef STANDALONE
#   include<pybind11/pybind11.h>
    namespace py = pybind11;
#endif

using namespace std;

#define RELATIVE_ERROR 0
#define ABSOLUTE_ERROR 1
#define DEVIATION 2

/**
 * @in source filename for comparison
 * @in golden filename for comparison
 * @return <return code, max error, avg error>
 **/
tuple<int, double, double> file(string src, string golden, string epsilon_str, string deviation)
{
    ifstream sfp(src);
    ifstream gfp(golden);
    if (!sfp or !gfp)
        // The ifstream::exception will be triggered when reaching enf-of-file.
        // This bahvior makes it is difficult to distinguish between EOF and FileNotFound
        // The reason I still throw it is for python code to catch.
        throw ios_base::failure(src + " or " + golden + " cannot be opened.");

    vector<double> devvec;
    if (!deviation.empty()) {
        ifstream dfp(deviation);
        if (!dfp)
            throw ios_base::failure(deviation + " cannot be opened.");

        string str;
        while (dfp >> str)
            try {
                devvec.push_back(stod(str));
            }
            catch (const invalid_argument &e) {
                devvec.push_back(nan(""));
            }
    }

    double epsilon;
    unsigned error_mode;
    try {
        if (epsilon_str.back() == '|') {
            epsilon_str.pop_back();
            error_mode = ABSOLUTE_ERROR;
        }
        else {
            if (deviation.empty())
                error_mode = RELATIVE_ERROR;
            else
                error_mode = DEVIATION;
        }
        epsilon = stod(epsilon_str);
    }
    catch (const invalid_argument &e) {
        throw invalid_argument("epsilon:" + epsilon_str + " cannot be converted into a double");
    }

    string str;
    vector<string>sstrvec, gstrvec;
    while (sfp >> str)
        sstrvec.push_back(str);
    while (gfp >> str)
        gstrvec.push_back(str);

    if (sstrvec.size() != gstrvec.size())
        throw out_of_range("Number of strings mismatched");

    double sval, gval;
    double maxErr=0.0, avgErr=0.0;
    string maxErrMsg;
    for (unsigned i=0; i<sstrvec.size(); i++) {
        // Skip the string that is float point number
        try {
            sval = stod(sstrvec[i]);
        }
        catch (const invalid_argument &e) {
            continue;
        }
        catch (const out_of_range &e) {
            sval = 0; // this will make error rate equal to 1
        }

        try {
            gval = stod(gstrvec[i]);
        }
        catch (const invalid_argument &e) {
            continue;
        }
        catch (const out_of_range &e) {
            throw; // golden cannot be out_of_range
        }

        // Check for Not a number
        if (isnan(sval)==true or isnan(gval)==true)
            throw out_of_range("Not a Number");

        double err;
        double relative_epi = fabs(0.00001*fmax(sval, gval));
        if (error_mode == RELATIVE_ERROR)
            err = fabs(sval - gval) / ((gval == 0)? 1 : gval);
        else if (error_mode == ABSOLUTE_ERROR) 
            err = fabs(sval - gval);
        else { // DEVIATION
            if (devvec[i] == 0)
                err = (fabs(sval - gval) <= relative_epi)? 0 : INT_MAX; // TODO: lazy @@
            else
                err = fabs(sval - gval) / devvec[i];
        }
       
        maxErr = (err > maxErr)? err : maxErr;
        avgErr += err;
    }

    avgErr = avgErr / sstrvec.size();
    return make_tuple((maxErr > epsilon)?1:0, maxErr, avgErr);
}

#ifdef STANDALONE
int main(int argc, char *argv[])
{
    if (argc < 5) {
        cout << "usage: fuzzycompare <source_file> <golden_file> <epsilon[|s]> <deviation_file>" << endl;
        cout << "epsilon    The number denotes the tolerable error calculated" << endl;
        cout << "           by max((source - golden)/golden). This is relative" << endl;
        cout << "           error. By sepcifying '|' at the end of epsilon," << endl;
        cout << "           the error becomes max(|source - golden|). This is " << endl;
        cout << "           absolute error." << endl;
        exit(-1);
    }

    tuple<int, double, double> rc;
    try {
        rc = file(argv[1], argv[2], argv[3], argv[4]);
    }
    catch (exception &e) {
        cout << e.what() << endl;
        exit(-1);
    }

    if (get<0>(rc) >= 0) {
        cout << "Max:" << get<1>(rc) << endl;
        cout << "Avg:" << get<2>(rc) << endl;
    }
    if (get<0>(rc) == 0)
        cout << "Pass" << endl;
    else
        cout << "Not pass" << endl;
    exit(get<0>(rc));
}
#else
PYBIND11_MODULE(fuzzycompare, m) {
    py::register_exception<ios_base::failure>(m, "FcIOException", PyExc_IOError);
    
    m.doc() = "Fuzzy compare"; // optional module docstring

    // By default, the allowed epsilon will be 1% in the relative mode
    m.def("file", &file, "File comparison", 
          py::arg("src"), py::arg("golden"), py::arg("epsilo_str") = "0.01", py::arg("deviation") = "");
}
#endif
