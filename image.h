#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>
#include <fstream>

class Image {
    std::vector<double> feature_vector;
    std::string label;

public:
    Image(std::string file_path);
    std::vector<double> get_feature_vector();
    std::string get_label();
    int get_label_index();
};

#endif
