fuzzycompare: fuzzycompare.cpp
	g++ --std=c++11 -O3 $< -o $@

clean:
	rm -f fuzzycompare

install:
	cp fuzzycompare /usr/local/bin

uninstall:
	rm -f /usr/local/bin/fuzzycompare
