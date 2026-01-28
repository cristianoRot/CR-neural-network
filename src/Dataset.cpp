// dataset.cpp

#include "Dataset.hpp"
#include "RNG.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <cctype>
#include <vector>
#include <numeric>
#include <iostream>

// ============================================================================
// Public Methods
// ============================================================================

Dataset::Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs)
    : inputs(std::move(inputs)), outputs(std::move(outputs))
{
    if (this->inputs.size() != this->outputs.size()) {
        throw std::invalid_argument("Error: Inputs and outputs must have the same size");
    }

    perm_idx.resize(this->inputs.size());
    std::iota(perm_idx.begin(), perm_idx.end(), 0);
}

const size_t Dataset::size() const { 
    return inputs.size(); 
}

Matrix Dataset::get_input(size_t start_idx, size_t batch_size) const
{
    size_t actual_batch_size = validate_batch_size(start_idx, batch_size, size());
    
    size_t input_dim = inputs[perm_idx[start_idx]].rows();
    Matrix batch_input(input_dim, actual_batch_size);
    
    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = perm_idx[start_idx + i];
        const Matrix& single_input = inputs[idx];
        
        for (size_t r = 0; r < input_dim; r++) {
            batch_input.set(r, i, single_input.get(r, 0));
        }
    }
    
    return batch_input;
}

std::vector<size_t> Dataset::get_output(size_t start_idx, size_t batch_size) const
{
    size_t actual_batch_size = validate_batch_size(start_idx, batch_size, size());
    
    std::vector<size_t> batch_labels;
    batch_labels.reserve(actual_batch_size);
    
    for (size_t i = 0; i < actual_batch_size; i++) {
        batch_labels.push_back(outputs[perm_idx[start_idx + i]]);
    }
    
    return batch_labels;
}

void Dataset::shuffle() { 
    std::shuffle(perm_idx.begin(), perm_idx.end(), RNG::get_engine());
}

std::vector<double> Dataset::get_class_weight() const
{
    if (outputs.empty()) {
        return {};
    }

    // Find maximum class index
    size_t max_class = *std::max_element(outputs.begin(), outputs.end());
    size_t num_classes = max_class + 1;

    // Count frequencies for each class
    std::vector<size_t> class_counts(num_classes, 0);
    for (size_t label : outputs) {
        class_counts[label]++;
    }

    // Calculate inverse frequencies
    std::vector<double> weights(num_classes);
    size_t total_samples = outputs.size();
    
    for (size_t i = 0; i < num_classes; i++) {
        if (class_counts[i] > 0) {
            // Inverse frequency: total_samples / (num_classes * class_count)
            weights[i] = static_cast<double>(total_samples) / (num_classes * class_counts[i]);
        } else {
            weights[i] = 0.0;
        }
    }

    return weights;
}

// ============================================================================
// Static Methods
// ============================================================================

Dataset Dataset::from_csv(
    const std::string& file_path,
    const std::vector<std::string>& input_columns,
    const std::string& output_column,
    size_t start_row,
    size_t end_row)
{
    std::cout << "Loading dataset from " << file_path << "..." << std::endl;

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open file: " + file_path);
    }

    // Read and parse header
    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Error: CSV file is empty");
    }

    std::vector<std::string> headers = split(header_line, ',');
    
    // Find input and output column indices
    std::vector<size_t> input_indices = find_input_column_indices(headers, input_columns, output_column);
    
    auto output_it = std::find(headers.begin(), headers.end(), output_column);
    if (output_it == headers.end()) {
        throw std::runtime_error("Error: Output column '" + output_column + "' not found in CSV");
    }
    size_t output_index = std::distance(headers.begin(), output_it);

    // Parse data rows
    std::vector<Matrix> inputs;
    std::vector<size_t> outputs;
    std::map<std::string, size_t> class_map;
    size_t next_class_index = 0;

    std::string line;
    size_t line_num = 1;
    size_t current_row = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip rows before start_row
        if (current_row < start_row) {
            current_row++;
            continue;
        }
        
        // Stop if we've reached end_row
        if (current_row >= end_row) {
            break;
        }
        
        current_row++;
        
        // Skip empty lines
        if (trim_whitespace(line).empty()) {
            continue;
        }

        std::vector<std::string> values = split(line, ',');
        
        if (values.size() != headers.size()) {
            throw std::runtime_error("Error: Line " + std::to_string(line_num) + 
                                  " has " + std::to_string(values.size()) + 
                                  " columns, expected " + std::to_string(headers.size()));
        }

        // Parse input values
        std::vector<double> input_values;
        input_values.reserve(input_indices.size());
        for (size_t idx : input_indices) {
            try {
                input_values.push_back(std::stod(values[idx]));
            } catch (const std::exception&) {
                throw std::runtime_error("Error: Cannot parse input value at line " + 
                                       std::to_string(line_num) + ", column '" + 
                                       headers[idx] + "': " + values[idx]);
            }
        }
        inputs.push_back(Matrix(input_values.size(), 1, input_values));

        // Parse output value
        std::string output_value = values[output_index];
        std::string trimmed_output = trim_whitespace(output_value);
        
        if (trimmed_output.empty()) {
            continue;
        }

        try {
            size_t output_index_value = parse_output_value(trimmed_output, class_map, next_class_index);
            outputs.push_back(output_index_value);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error: Cannot parse output value at line " + 
                                   std::to_string(line_num) + ": " + e.what());
        }
    }

    std::cout << "Dataset loaded successfully" << std::endl;

    if (inputs.empty()) {
        throw std::runtime_error("Error: No data rows found in CSV file");
    }

    return Dataset(inputs, outputs);
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string trim_whitespace(const std::string& s) {
    constexpr const char* whitespace = " \t\n\r";
    size_t first = s.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    size_t last = s.find_last_not_of(whitespace);
    return s.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(trim_whitespace(token));
    }
    return tokens;
}

size_t validate_batch_size(size_t start_idx, size_t batch_size, size_t dataset_size) {
    size_t actual_batch_size = std::min(batch_size, dataset_size - start_idx);
    if (actual_batch_size == 0) {
        throw std::invalid_argument("Error: Invalid batch start index");
    }
    return actual_batch_size;
}

size_t parse_output_value(const std::string& value, std::map<std::string, size_t>& class_map, size_t& next_class_index) {
    if (value.empty()) {
        throw std::runtime_error("Error: Empty output value");
    }

    // Try to parse as numeric (unsigned integer)
    bool is_numeric = true;
    for (char c : value) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            is_numeric = false;
            break;
        }
    }
    
    if (is_numeric) {
        try {
            return static_cast<size_t>(std::stoul(value));
        } catch (...) {
            // Fall through to string label handling
        }
    }
    
    // Not numeric, treat as string label
    auto it = class_map.find(value);
    if (it == class_map.end()) {
        class_map[value] = next_class_index++;
        return class_map[value];
    }
    return it->second;
}

std::vector<size_t> find_input_column_indices(
    const std::vector<std::string>& headers,
    const std::vector<std::string>& input_columns,
    const std::string& output_column)
{
    std::vector<size_t> input_indices;
    
    if (input_columns.size() == 1 && input_columns[0] == "ALL") {
        input_indices.reserve(headers.size() - 1);
        for (size_t i = 0; i < headers.size(); i++) {
            if (headers[i] != output_column) {
                input_indices.push_back(i);
            }
        }
    } else {
        input_indices.reserve(input_columns.size());
        for (const auto& col_name : input_columns) {
            auto it = std::find(headers.begin(), headers.end(), col_name);
            if (it == headers.end()) {
                throw std::runtime_error("Error: Input column '" + col_name + "' not found in CSV");
            }
            input_indices.push_back(std::distance(headers.begin(), it));
        }
    }
    
    return input_indices;
}
