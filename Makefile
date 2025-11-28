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

# Target executables
TRAIN_TARGET = train
MAIN_TARGET = main

# Default target
all: $(BUILD_DIR)/$(TRAIN_TARGET) $(BUILD_DIR)/$(MAIN_TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link train executable
$(BUILD_DIR)/$(TRAIN_TARGET): $(OBJECTS) train.cpp
	$(CXX) $(CXXFLAGS) -o $@ train.cpp $(OBJECTS) $(LDFLAGS)

# Link main executable
$(BUILD_DIR)/$(MAIN_TARGET): $(OBJECTS) main.cpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(OBJECTS) $(LDFLAGS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TRAIN_TARGET) $(MAIN_TARGET)

# Rebuild everything
rebuild: clean all

# Run train
train: $(BUILD_DIR)/$(TRAIN_TARGET)
	./$(BUILD_DIR)/$(TRAIN_TARGET)

# Run main
run: $(BUILD_DIR)/$(MAIN_TARGET)
	./$(BUILD_DIR)/$(MAIN_TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Build both train and main (default)"
	@echo "  train   - Build and run train"
	@echo "  run     - Build and run main"
	@echo "  clean   - Remove build files"
	@echo "  rebuild - Clean and build"
	@echo "  help    - Show this help"

.PHONY: all clean rebuild train run help
