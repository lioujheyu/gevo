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
#   include<pybind11/stl.h>
    namespace py = pybind11;
#endif

using namespace std;

int file(vector<string> &file_list) 
{
    if (file_list.size() < 2){
        throw invalid_argument("Need at least two files");
    }

    string prefix, fnmean, fnstddev;
    for (unsigned i=0; i<file_list[0].size(); ++i) {
        if (file_list[0][i] == file_list[1][i])
            prefix.push_back(file_list[0][i]);
        else
            break;
    }
    if (prefix.size() > 0){
        fnmean = prefix + "mean";
        fnstddev = prefix + "stddev";
    }
    else {
        fnmean = "mean.txt";
        fnstddev = "stddev.txt";
    }

    vector<vector<string>> files_content(file_list.size());
    for (unsigned i=0; i<file_list.size(); ++i){
        ifstream f(file_list[i]);
        if (!f)
            throw ios_base::failure(file_list[i] + " cannot be opened.");
        for (std::string line; std::getline(f, line);) {
            std::istringstream str(line);

            for (std::string word; str >> word;)
                files_content[i].push_back(word);
            files_content[i].push_back("\n");
        }
    }

    unsigned long n_words = files_content[0].size();
    for (unsigned i=1; i<file_list.size(); ++i){
        if (n_words != files_content[i].size())
            throw out_of_range("file:\"" + file_list[i] + "\" has size mismatched!");
    }

    ofstream fmean(fnmean, ofstream::out);
    ofstream fstddev(fnstddev, ofstream::out);
    for (unsigned long n=0; n<n_words; n++) {
        if (files_content[0][n] == "\n") {
            fmean << endl;
            fstddev << endl;
            continue;
        }

        double total=0, mean=0, variance=0, stddev=0;
        try {
            double val = stod(files_content[0][n]);
        }
        catch (const invalid_argument &e) {
            fmean << files_content[0][n] << " ";
            fstddev << "## ";
            continue;
        }
        catch (const out_of_range &e) {
            throw e;
        }

        vector<double> values(file_list.size());
        for (unsigned f=0; f<file_list.size(); f++) {
            try {
                values[f] = stod(files_content[f][n]);
            }
            catch (const invalid_argument &e) {
                throw range_error(files_content[0][n] + " and " + files_content[f][n] +  " mismatched ");
            }
        }

        for (auto v : values)
            total += v;
        mean = total/file_list.size();
        fmean << mean << " ";

        for (auto v : values) {
            double diff = fabs(v - mean);
            diff = (diff < 0.00001*fmax(fabs(v), fabs(mean))) ? 0 : diff;
            variance += diff * diff;
        }
        stddev = sqrt(variance/file_list.size());
        fstddev << stddev << " ";
    }
    fmean.close();
    fstddev.close();

    return 0;
}

#ifdef STANDALONE
int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "usage: variancecalc <in_file1> <in_file2> [in_file3 ... ]" << endl;
        exit(-1);
    }

    unsigned rc = 0;
    vector<string> file_list;
    for (int i=1; i<argc; ++i)
        file_list.push_back(string(argv[i]));

    rc = file(file_list);
    exit(rc);
}
#else
PYBIND11_MODULE(variancecalc, m) {
    py::register_exception<ios_base::failure>(m, "FcIOException", PyExc_IOError);
    
    m.doc() = "Calculate variance"; // optional module docstring

    // By default, the allowed epsilon will be 1% in the relative mode
    m.def("file", &file, "Calculate variance", py::arg("file_list"));
}
#endif