#include "image.h"

Image::Image(std::string file_path) {
    std::ifstream f(file_path);

    std::string line;
    std::getline(f, line);
    this->label = line;

    while (std::getline(f, line)) {
        this->feature_vector.push_back(std::stof(line));
    }
}

std::vector<double> Image::get_feature_vector() {
    return this->feature_vector;
}

std::string Image::get_label() {
    return this->label;
}

int Image::get_label_index() {
    std::string label = this->get_label();
    int i = 0;
    while (i < label.size() && label[i] != '1') ++i;
    return i;
}
