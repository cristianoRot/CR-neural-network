// dataset.cpp

#include "Dataset.hpp"
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <cctype>

Dataset::Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs)
{
    if (inputs.size() != outputs.size())
    {
        throw std::invalid_argument("Error: Inputs and outputs must have the same size");
    }

    this->inputs = inputs;
    this->outputs = outputs;

    perm_idx.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        perm_idx.push_back(i);
    }
}

const size_t Dataset::size() const { return inputs.size(); }

const Matrix& Dataset::get_input(size_t index) const { return inputs[perm_idx[index]]; }

const size_t Dataset::get_output(size_t index) const { return outputs[perm_idx[index]]; }

void Dataset::shuffle() 
{ 
    std::shuffle(perm_idx.begin(), perm_idx.end(), std::mt19937{std::random_device{}()});
}

static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        tokens.push_back(token);
    }
    return tokens;
}

Dataset Dataset::from_csv(
    const std::string& file_path,
    const std::vector<std::string>& input_columns,
    const std::string& output_column)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open file: " + file_path);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Error: CSV file is empty");
    }

    std::vector<std::string> headers = split(header_line, ',');
    
    std::vector<size_t> input_indices;
    input_indices.reserve(input_columns.size());
    for (const auto& col_name : input_columns) {
        auto it = std::find(headers.begin(), headers.end(), col_name);
        if (it == headers.end()) {
            throw std::runtime_error("Error: Input column '" + col_name + "' not found in CSV");
        }
        input_indices.push_back(std::distance(headers.begin(), it));
    }

    auto output_it = std::find(headers.begin(), headers.end(), output_column);
    if (output_it == headers.end()) {
        throw std::runtime_error("Error: Output column '" + output_column + "' not found in CSV");
    }
    size_t output_index = std::distance(headers.begin(), output_it);

    std::vector<Matrix> inputs;
    std::vector<size_t> outputs;
    std::map<std::string, size_t> class_map;
    size_t next_class_index = 0;

    std::string line;
    size_t line_num = 1;
    while (std::getline(file, line)) {
        line_num++;
        
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }

        std::vector<std::string> values = split(line, ',');
        
        if (values.size() != headers.size()) {
            throw std::runtime_error("Error: Line " + std::to_string(line_num) + 
                                  " has " + std::to_string(values.size()) + 
                                  " columns, expected " + std::to_string(headers.size()));
        }

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

        std::string output_value = values[output_index];
        
        if (!output_value.empty()) {
            size_t first = output_value.find_first_not_of(" \t\n\r");
            if (first != std::string::npos) {
                size_t last = output_value.find_last_not_of(" \t\n\r");
                output_value = output_value.substr(first, last - first + 1);
            } else {
                output_value.clear();
            }
        }

        if (output_value.empty()) {
            continue;
        }

        size_t output_index_value;
        bool is_numeric = !output_value.empty();
        
        if (is_numeric) {
            size_t start = 0;
            if (output_value[0] == '+' || output_value[0] == '-') {
                start = 1;
            }
            for (size_t i = start; i < output_value.length(); i++) {
                if (!std::isdigit(static_cast<unsigned char>(output_value[i])) && output_value[i] != '.') {
                    is_numeric = false;
                    break;
                }
            }
        }
        
        if (is_numeric) {
            try {
                output_index_value = static_cast<size_t>(std::stoul(output_value));
            } catch (const std::exception&) {
                is_numeric = false;
            }
        }
        
        if (!is_numeric) {
            if (class_map.find(output_value) == class_map.end()) {
                class_map[output_value] = next_class_index++;
            }
            output_index_value = class_map[output_value];
        }

        outputs.push_back(output_index_value);
    }

    file.close();

    if (inputs.empty()) {
        throw std::runtime_error("Error: No data rows found in CSV file");
    }

    return Dataset(inputs, outputs);
}