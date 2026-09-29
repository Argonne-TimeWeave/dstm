SST_CONFIG ?= sst-config
CXX := $(shell $(SST_CONFIG) --CXX)

CXXFLAGS := $(shell $(SST_CONFIG) --ELEMENT_CXXFLAGS) \
	-O2 -std=c++17 -fPIC

LDFLAGS := $(shell $(SST_CONFIG) --ELEMENT_LDFLAGS)

TARGET := libdcbench.so

.PHONY: all register clean

all: $(TARGET)

$(TARGET): datacenter.cc
	$(CXX) $(CXXFLAGS) -shared -o $@ $< $(LDFLAGS)

register: $(TARGET)
	sst-register dcbench dcbench_LIBDIR=$(CURDIR)

clean:
	rm -f $(TARGET)
