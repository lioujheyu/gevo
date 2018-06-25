fuzzycompare: fuzzycompare.cpp
	g++ -shared -Wall --std=c++11 -O3 $< -o $@`python3-config --extension-suffix` -fPIC `python3 -m pybind11 --includes`
	g++ -Wall --std=c++11 -O3 $< -o $@ -fPIC `python3 -m pybind11 --includes` -DSTANDALONE

clean:
	rm -f fuzzycompare

install:
	cp fuzzycompare /usr/local/bin

uninstall:
	rm -f /usr/local/bin/fuzzycompare
