# Simple Makefile for HIP test
CXX = /nix/store/l4rz8cbs9nns5xn7bbq6x0narqhwbbnc-hipcc-19.0.0-rocm/bin/hipcc
CXXFLAGS = -std=c++17
LDFLAGS = -lhiprtc

TARGET = simple_hip_test

$(TARGET): simple_hip_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: clean
