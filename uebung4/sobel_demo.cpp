#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <array>
#include <iostream>

namespace py = pybind11;


Eigen::MatrixXd sobel(Eigen::MatrixXd gray_img, bool is_x_dir) {
    
    // TODO: implement filter operation
    std::array<double, 3> filter1;
    std::array<double, 3> filter2;
    if (is_x_dir) {
        filter1 = {1, 2, 1};
        filter2 = {1, 0, -1};
    } else {
        filter1 = {1, 0, -1};
        filter2 = {1, 2, 1};
    }
    Eigen::MatrixXd filtered_img = Eigen::MatrixXd::Zero(gray_img.rows() - 2, gray_img.cols());

    for (int i = 1; i < gray_img.rows() - 1; i++) {
        for (int j = 0; j < gray_img.cols(); j++) {
            for (int k = i - 1; k < i + 2; k++) {
                filtered_img(i - 1, j) += gray_img(k, j) * filter1[k - i + 1];
            }
            if (i == 1 && j == 0) {
                std::cout << gray_img(0, 0) << "\n";
                std::cout << filter1[1] << "\n";
                std::cout << filtered_img(i - 1, j) << "\n";
            }
        }
    }
    Eigen::MatrixXd filtered_img2 = Eigen::MatrixXd::Zero(gray_img.rows() - 2, gray_img.cols()-2);

    for (int i = 0; i < filtered_img.rows(); i++) {
        for (int j = 1; j < gray_img.cols() - 1; j++) {
            for (int l = j - 1; l < j + 2; l++) {
                filtered_img2(i, j - 1) += filtered_img(i, l) * filter2[l - j + 1];
            }
            //filtered_img2(i, j - 1) /= 8;
        }
    }

    filtered_img2 /= 8.0;
    return filtered_img2;
}


PYBIND11_MODULE(sobel_demo, m) {
    m.doc() = "sobel operator using numpy!";
    m.def("sobel", &sobel);
}