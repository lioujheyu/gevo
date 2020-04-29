fuzzycompare: fuzzycompare.cpp
	g++ -shared -Wall --std=c++11 -O3 $< -o $@`python3-config --extension-suffix` -fPIC `python3 -m pybind11 --includes`
	g++ -DSTANDALONE -Wall --std=c++11 -g $< -o $@ -fPIC

clean:
	rm -f fuzzycompare

install:
	cp fuzzycompare /usr/local/bin

uninstall:
	rm -f /usr/local/bin/fuzzycompare
