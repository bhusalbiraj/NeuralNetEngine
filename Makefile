CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 $(shell fltk-config --cxxflags)
LDFLAGS = $(shell fltk-config --ldflags)

TARGET = nn_gui
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean