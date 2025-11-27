# Makefile for CrNeuralNet

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I include
LDFLAGS = -framework Accelerate

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = main

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) main.cpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(OBJECTS) $(LDFLAGS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET)

# Rebuild everything
rebuild: clean all

# Run the test
run: $(TARGET)
	./$(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Build the project (default)"
	@echo "  clean   - Remove build files"
	@echo "  rebuild - Clean and build"
	@echo "  run     - Build and run the test"
	@echo "  help    - Show this help"

.PHONY: all clean rebuild run help
