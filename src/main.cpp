#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <algorithm>
#include <ctime>
#include <functional>
#include <stdexcept>
#include <iomanip>

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Widget.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/fl_draw.H>

using namespace std;

namespace NeuralNetEngine {

class DimensionMismatchException : public runtime_error {
public:
    explicit DimensionMismatchException(const string& msg)
        : runtime_error("Matrix Dimension Mismatch Error: " + msg) {}
};

class InvalidDatasetException : public runtime_error {
public:
    explicit InvalidDatasetException(const string& msg)
        : runtime_error("Dataset Processing Error: " + msg) {}
};

template <typename T = float>
class Matrix2D {
private:
    int rows_;
    int cols_;
    vector<vector<T>> data_;

public:
    Matrix2D() : rows_(0), cols_(0) {}

    Matrix2D(int rows, int cols, T fill_value = static_cast<T>(0))
        : rows_(rows), cols_(cols), data_(rows, vector<T>(cols, fill_value)) {}

    ~Matrix2D() = default;

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    T& operator()(int r, int c) { return data_[r][c]; }
    const T& operator()(int r, int c) const { return data_[r][c]; }

    Matrix2D operator+(const Matrix2D& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchException("Addition dimensions do not match.");
        }
        Matrix2D result(rows_, cols_);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result(i, j) = data_[i][j] + other(i, j);
            }
        }
        return result;
    }

    static Matrix2D<T> random(int rows, int cols, T min_val = static_cast<T>(-0.5), T max_val = static_cast<T>(0.5)) {
        Matrix2D<T> result(rows, cols);
        static mt19937 gen(1337);
        uniform_real_distribution<T> dist(min_val, max_val);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    static Matrix2D<T> dot(const Matrix2D<T>& A, const Matrix2D<T>& B) {
        if (A.cols() != B.rows()) {
            throw DimensionMismatchException("Matrix dot product inner dimensions must agree.");
        }
        Matrix2D<T> C(A.rows(), B.cols(), static_cast<T>(0));
        for (int i = 0; i < A.rows(); i++) {
            for (int k = 0; k < A.cols(); k++) {
                T valA = A(i, k);
                for (int j = 0; j < B.cols(); j++) {
                    C(i, j) += valA * B(k, j);
                }
            }
        }
        return C;
    }

    Matrix2D<T> transpose() const {
        Matrix2D<T> result(cols_, rows_);
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                result(j, i) = data_[i][j];
            }
        }
        return result;
    }

    void addBias(const Matrix2D<T>& bias) {
        if (bias.cols() != cols_) {
            throw DimensionMismatchException("Bias vector dimension must match matrix columns.");
        }
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                data_[i][j] += bias(0, j);
            }
        }
    }

    template <typename U>
    friend ostream& operator<<(ostream& os, const Matrix2D<U>& mat);
};

template <typename U>
ostream& operator<<(ostream& os, const Matrix2D<U>& mat) {
    os << "Matrix2D[" << mat.rows_ << "x" << mat.cols_ << "]";
    return os;
}

using Matrix = Matrix2D<float>;

class Layer {
protected:
    static int total_layers_count;

public:
    Layer() { total_layers_count++; }
    virtual ~Layer() { total_layers_count--; }

    static int getTotalLayersCount() { return total_layers_count; }

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& d_outputs) = 0;
    virtual void update(float learning_rate) = 0;
};

int Layer::total_layers_count = 0;

class DenseLayer : public Layer {
private:
    int n_inputs;
    int n_nodes;
    Matrix weights;
    Matrix bias;
    Matrix inputs;
    Matrix dweights;
    Matrix dbias;

public:
    DenseLayer(int inputs_count, int nodes_count)
        : n_inputs(inputs_count), n_nodes(nodes_count) {
        weights = Matrix::random(n_inputs, n_nodes, -0.5f, 0.5f);
        
        static mt19937 gen(random_device{}());
        float stddev = sqrt(2.0f / static_cast<float>(n_inputs));
        normal_distribution<float> dist(0.0f, stddev);

        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_nodes; j++) {
                weights(i, j) = dist(gen);
            }
        }
        bias = Matrix(1, n_nodes, 0.0f);
    }

    Matrix forward(const Matrix& input) override {
        this->inputs = input;
        Matrix output = Matrix::dot(input, weights);
        output.addBias(bias);
        return output;
    }

    Matrix backward(const Matrix& d_outputs) override {
        Matrix inputs_T = inputs.transpose();
        dweights = Matrix::dot(inputs_T, d_outputs);

        dbias = Matrix(1, n_nodes, 0.0f);
        for (int i = 0; i < d_outputs.rows(); i++) {
            for (int j = 0; j < n_nodes; j++) {
                dbias(0, j) += d_outputs(i, j);
            }
        }

        Matrix weights_T = weights.transpose();
        return Matrix::dot(d_outputs, weights_T);
    }

    void update(float learning_rate) override {
        for (int i = 0; i < n_inputs; i++) {
            for (int j = 0; j < n_nodes; j++) {
                weights(i, j) -= learning_rate * dweights(i, j);
            }
        }
        for (int j = 0; j < n_nodes; j++) {
            bias(0, j) -= learning_rate * dbias(0, j);
        }
    }
};

class ReLULayer : public Layer {
private:
    Matrix inputs;

public:
    Matrix forward(const Matrix& input) override {
        this->inputs = input;
        Matrix output(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
                output(i, j) = max(0.0f, input(i, j));
            }
        }
        return output;
    }

    Matrix backward(const Matrix& d_outputs) override {
        Matrix d_inputs(d_outputs.rows(), d_outputs.cols());
        for (int i = 0; i < d_outputs.rows(); i++) {
            for (int j = 0; j < d_outputs.cols(); j++) {
                d_inputs(i, j) = (inputs(i, j) > 0.0f) ? d_outputs(i, j) : 0.0f;
            }
        }
        return d_inputs;
    }

    void update(float learning_rate) override {}
};

class SoftmaxLayer : public Layer {
private:
    Matrix outputs;

public:
    Matrix forward(const Matrix& input) override {
        Matrix output(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            float max_val = input(i, 0);
            for (int j = 1; j < input.cols(); j++) {
                if (input(i, j) > max_val) max_val = input(i, j);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < input.cols(); j++) {
                output(i, j) = exp(input(i, j) - max_val);
                sum_exp += output(i, j);
            }

            for (int j = 0; j < input.cols(); j++) {
                output(i, j) /= sum_exp;
            }
        }
        this->outputs = output;
        return output;
    }

    Matrix backward(const Matrix& d_outputs) override {
        return d_outputs;
    }

    void update(float learning_rate) override {}
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual float calculate(const Matrix& predictions, const vector<int>& target_labels) = 0;
    virtual Matrix gradient(const Matrix& predictions, const vector<int>& target_labels) = 0;
};

class CategoricalCrossEntropyLoss : public Loss {
public:
    float calculate(const Matrix& predictions, const vector<int>& target_labels) override {
        float total_loss = 0.0f;
        int n_samples = predictions.rows();
        for (int i = 0; i < n_samples; i++) {
            int target_class = target_labels[i];
            float prob = predictions(i, target_class);
            prob = clamp(prob, 1e-7f, 1.0f - 1e-7f);
            total_loss += -log(prob);
        }
        return total_loss / n_samples;
    }

    Matrix gradient(const Matrix& predictions, const vector<int>& target_labels) override {
        int n_samples = predictions.rows();
        int n_classes = predictions.cols();
        Matrix d_inputs(n_samples, n_classes);

        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_classes; j++) {
                float target = (j == target_labels[i]) ? 1.0f : 0.0f;
                d_inputs(i, j) = (predictions(i, j) - target) / static_cast<float>(n_samples);
            }
        }
        return d_inputs;
    }
};

class NeuralNetwork {
private:
    vector<shared_ptr<Layer>> layers;
    shared_ptr<Loss> loss_function;

public:
    NeuralNetwork() {
        loss_function = make_shared<CategoricalCrossEntropyLoss>();
    }

    void addLayer(shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    void clearLayers() {
        layers.clear();
    }

    Matrix forward(const Matrix& X) {
        Matrix curr = X;
        for (auto& layer : layers) {
            curr = layer->forward(curr);
        }
        return curr;
    }

    void backward(const Matrix& d_loss) {
        Matrix curr_grad = d_loss;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            curr_grad = (*it)->backward(curr_grad);
        }
    }

    void update(float learning_rate) {
        for (auto& layer : layers) {
            layer->update(learning_rate);
        }
    }

    pair<float, float> evaluate(const Matrix& X, const vector<int>& y) {
        if (X.rows() == 0) return {0.0f, 0.0f};

        Matrix predictions = forward(X);
        float loss = loss_function->calculate(predictions, y);

        int correct = 0;
        for (int i = 0; i < X.rows(); i++) {
            int best_class = 0;
            float max_prob = predictions(i, 0);
            for (int j = 1; j < predictions.cols(); j++) {
                if (predictions(i, j) > max_prob) {
                    max_prob = predictions(i, j);
                    best_class = j;
                }
            }
            if (best_class == y[i]) correct++;
        }
        return {loss, (float)correct / X.rows()};//loss ra accuracy return garxa
    }

    pair<float, float> trainStep(const Matrix& X, const vector<int>& y, float learning_rate) {
        if (X.rows() == 0) return {0.0f, 0.0f};

        Matrix predictions = forward(X);
        float loss = loss_function->calculate(predictions, y);

        int correct = 0;
        for (int i = 0; i < X.rows(); i++) {
            int best_class = 0;
            float max_prob = predictions(i, 0);
            for (int j = 1; j < predictions.cols(); j++) {
                if (predictions(i, j) > max_prob) {
                    max_prob = predictions(i, j);
                    best_class = j;
                }
            }
            if (best_class == y[i]) correct++;
        }
        float accuracy = (float)correct / X.rows();

        Matrix d_loss = loss_function->gradient(predictions, y);
        backward(d_loss);
        update(learning_rate);

        return {loss, accuracy};
    }

    friend class NeuralNetworkDebugger;
};

class NeuralNetworkDebugger {
public:
    static void inspectLayers(const NeuralNetwork& net) {
        cout << "[Debugger] Neural Network contains " << net.layers.size() 
             << " layers. Total active Layer instances: " << Layer::getTotalLayersCount() << "\n";
    }
};

}

using namespace NeuralNetEngine;

struct Point2D {
    float x;
    float y;
    int label;
    bool is_validation = false;
};

void applyTrainValSplit(vector<Point2D>& points) {
    for (size_t i = 0; i < points.size(); i++) {
        points[i].is_validation = (i % 5 == 4);
    }
}

bool loadCSVData(const string& x_file, const string& y_file, vector<Point2D>& points) {
    ifstream fx(x_file), fy(y_file);
    if (!fx.is_open() || !fy.is_open()) {
        return false;
    }

    points.clear();
    string line_x, line_y;
    while (getline(fx, line_x) && getline(fy, line_y)) {
        if (line_x.empty() || line_y.empty()) continue;
        
        stringstream ss(line_x);
        string val1, val2;
        if (getline(ss, val1, ',') && getline(ss, val2, ',')) {
            Point2D pt;
            try {
                pt.x = stof(val1);
                pt.y = stof(val2);
                pt.label = stoi(line_y);
                points.push_back(pt);
            } catch (const exception& e) {
                continue;
            }
        }
    }
    applyTrainValSplit(points);
    return !points.empty();
}

void generateSpiralData(vector<Point2D>& points, int samples_per_class = 100) {
    points.clear();
    int classes = 3;
    for (int c = 0; c < classes; c++) {
        for (int i = 0; i < samples_per_class; i++) {
            float r = (float)i / samples_per_class * 1.5f;
            float t = c * 4.0f + (float)i / samples_per_class * 4.0f + ((float)rand()/RAND_MAX * 0.2f);
            Point2D pt;
            pt.x = r * sin(t);
            pt.y = r * cos(t);
            pt.label = c;
            points.push_back(pt);
        }
    }
    applyTrainValSplit(points);
}

void generateCirclesData(vector<Point2D>& points, int count = 300) {
    points.clear();
    for (int i = 0; i < count; i++) {
        float r = (float)rand() / RAND_MAX * 1.6f;
        float angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
        Point2D pt;
        pt.x = r * cos(angle);
        pt.y = r * sin(angle);
        
        if (r < 0.5f) pt.label = 0;
        else if (r < 1.0f) pt.label = 1;
        else pt.label = 2;
        
        points.push_back(pt);
    }
    applyTrainValSplit(points);
}

void generateXORData(vector<Point2D>& points, int count = 300) {
    points.clear();
    for (int i = 0; i < count; i++) {
        Point2D pt;
        pt.x = ((float)rand() / RAND_MAX * 3.0f) - 1.5f;
        pt.y = ((float)rand() / RAND_MAX * 3.0f) - 1.5f;
        
        if (pt.x > 0 && pt.y > 0) pt.label = 0;
        else if (pt.x < 0 && pt.y > 0) pt.label = 1;
        else if (pt.x < 0 && pt.y < 0) pt.label = 2;
        else pt.label = 0;
        
        points.push_back(pt);
    }
    applyTrainValSplit(points);
}

void pointsToMatrix(const vector<Point2D>& points, Matrix& X, vector<int>& y) {
    int n = points.size();
    X = Matrix(n, 2);
    y.resize(n);
    for (int i = 0; i < n; i++) {
        X(i, 0) = points[i].x;
        X(i, 1) = points[i].y;
        y[i] = points[i].label;
    }
}

void pointsToTrainValMatrices(const vector<Point2D>& points, 
                              Matrix& X_train, vector<int>& y_train,
                              Matrix& X_val, vector<int>& y_val) {
    vector<Point2D> train_pts, val_pts;
    for (const auto& pt : points) {
        if (pt.is_validation) val_pts.push_back(pt);
        else train_pts.push_back(pt);
    }
    if (train_pts.empty()) train_pts = points;
    if (val_pts.empty()) val_pts = points;

    pointsToMatrix(train_pts, X_train, y_train);
    pointsToMatrix(val_pts, X_val, y_val);
}

class DecisionBoundaryWidget : public Fl_Widget {
private:
    NeuralNetwork* net;
    vector<Point2D>* points;
    float range;
    
    bool has_prediction;
    float pred_x;
    float pred_y;

    function<void(float, float)> on_click_callback;

public:
    DecisionBoundaryWidget(int X, int Y, int W, int H, NeuralNetwork* network, vector<Point2D>* data_pts)
        : Fl_Widget(X, Y, W, H), net(network), points(data_pts), range(2.0f),
          has_prediction(false), pred_x(0.0f), pred_y(0.0f) {}

    void setOnClickCallback(function<void(float, float)> cb) {
        on_click_callback = cb;
    }

    void setPredictionMarker(float px, float py) {
        has_prediction = true;
        pred_x = px;
        pred_y = py;
        redraw();
    }

    void clearPredictionMarker() {
        has_prediction = false;
        redraw();
    }

    void draw() override {
        fl_color(30, 30, 35);
        fl_rectf(x(), y(), w(), h());

        if (net && points && !points->empty()) {
            int grid_res = 50;
            float cell_w = (float)w() / grid_res;
            float cell_h = (float)h() / grid_res;

            Matrix grid_inputs(grid_res * grid_res, 2);
            int idx = 0;
            for (int gx = 0; gx < grid_res; gx++) {
                for (int gy = 0; gy < grid_res; gy++) {
                    float px = -range + (float)gx / grid_res * (2.0f * range);
                    float py = range - (float)gy / grid_res * (2.0f * range);
                    grid_inputs(idx, 0) = px;
                    grid_inputs(idx, 1) = py;
                    idx++;
                }
            }

            Matrix preds = net->forward(grid_inputs);

            idx = 0;
            for (int gx = 0; gx < grid_res; gx++) {
                for (int gy = 0; gy < grid_res; gy++) {
                    float p0 = preds(idx, 0);
                    float p1 = preds(idx, 1);
                    float p2 = preds(idx, 2);
                    idx++;

                    uchar r = static_cast<uchar>(p0 * 220 + 20);
                    uchar g = static_cast<uchar>(p1 * 220 + 20);
                    uchar b = static_cast<uchar>(p2 * 220 + 20);

                    fl_color(r, g, b);
                    fl_rectf(x() + gx * cell_w, y() + gy * cell_h, cell_w + 1, cell_h + 1);
                }
            }
        }

        fl_color(100, 100, 110);
        int mid_x = x() + w() / 2;
        int mid_y = y() + h() / 2;
        fl_line(mid_x, y(), mid_x, y() + h());
        fl_line(x(), mid_y, x() + w(), mid_y);

        if (points) {
            for (const auto& pt : *points) {
                int px = x() + static_cast<int>((pt.x + range) / (2.0f * range) * w());
                int py = y() + static_cast<int>((range - pt.y) / (2.0f * range) * h());

                if (pt.label == 0) fl_color(255, 60, 60);
                else if (pt.label == 1) fl_color(60, 255, 60);
                else fl_color(60, 160, 255);

                if (pt.is_validation) {
                    fl_pie(px - 6, py - 6, 12, 12, 0, 360);
                    fl_color(255, 255, 255);
                    fl_pie(px - 3, py - 3, 6, 6, 0, 360);
                    fl_color(0, 0, 0);
                    fl_arc(px - 6, py - 6, 12, 12, 0, 360);
                } else {
                    fl_pie(px - 5, py - 5, 10, 10, 0, 360);
                    fl_color(0, 0, 0);
                    fl_arc(px - 5, py - 5, 10, 10, 0, 360);
                }
            }
        }

        if (has_prediction) {
            int px = x() + static_cast<int>((pred_x + range) / (2.0f * range) * w());
            int py = y() + static_cast<int>((range - pred_y) / (2.0f * range) * h());

            fl_color(255, 235, 0);
            fl_pie(px - 7, py - 7, 14, 14, 0, 360);
            fl_color(20, 20, 20);
            fl_pie(px - 3, py - 3, 6, 6, 0, 360);

            fl_color(255, 235, 0);
            fl_line(px - 14, py, px + 14, py);
            fl_line(px, py - 14, px, py + 14);
        }

        fl_color(180, 180, 190);
        fl_rect(x(), y(), w(), h());
    }

    int handle(int event) override {
        if (event == FL_PUSH) {
            int mx = Fl::event_x();
            int my = Fl::event_y();
            if (mx >= x() && mx <= x() + w() && my >= y() && my <= y() + h()) {
                float px = -range + (float)(mx - x()) / w() * (2.0f * range);
                float py = range - (float)(my - y()) / h() * (2.0f * range);

                if (on_click_callback) {
                    on_click_callback(px, py);
                }
                return 1;
            }
        }
        return Fl_Widget::handle(event);
    }
};

class LossGraphWidget : public Fl_Widget {
private:
    vector<float> train_loss_history;
    vector<float> val_loss_history;

public:
    LossGraphWidget(int X, int Y, int W, int H) : Fl_Widget(X, Y, W, H) {}

    void addLoss(float train_loss, float val_loss) {
        train_loss_history.push_back(train_loss);
        val_loss_history.push_back(val_loss);
        if (train_loss_history.size() > 200) {
            train_loss_history.erase(train_loss_history.begin());
            val_loss_history.erase(val_loss_history.begin());
        }
        redraw();
    }

    void clear() {
        train_loss_history.clear();
        val_loss_history.clear();
        redraw();
    }

    void draw() override {
        fl_color(20, 20, 25);
        fl_rectf(x(), y(), w(), h());

        fl_color(200, 200, 210);
        fl_font(FL_HELVETICA_BOLD, 11);
        fl_draw("Real-Time Loss Curve", x() + 10, y() + 16);

        fl_color(255, 180, 40);
        fl_draw("Train Loss", x() + 150, y() + 16);

        fl_color(40, 200, 240);
        fl_draw("Val Loss", x() + 250, y() + 16);

        if (train_loss_history.size() < 2) return;

        float max_loss = 1.0f;
        for (float l : train_loss_history) if (l > max_loss) max_loss = l;
        for (float l : val_loss_history) if (l > max_loss) max_loss = l;

        fl_color(255, 180, 40);
        for (size_t i = 1; i < train_loss_history.size(); i++) {
            int x1 = x() + static_cast<int>((float)(i - 1) / (train_loss_history.size() - 1) * (w() - 20)) + 10;
            int y1 = y() + h() - 8 - static_cast<int>(train_loss_history[i - 1] / max_loss * (h() - 32));
            int x2 = x() + static_cast<int>((float)i / (train_loss_history.size() - 1) * (w() - 20)) + 10;
            int y2 = y() + h() - 8 - static_cast<int>(train_loss_history[i] / max_loss * (h() - 32));
            fl_line(x1, y1, x2, y2);
        }

        fl_color(40, 200, 240);
        for (size_t i = 1; i < val_loss_history.size(); i++) {
            int x1 = x() + static_cast<int>((float)(i - 1) / (val_loss_history.size() - 1) * (w() - 20)) + 10;
            int y1 = y() + h() - 8 - static_cast<int>(val_loss_history[i - 1] / max_loss * (h() - 32));
            int x2 = x() + static_cast<int>((float)i / (val_loss_history.size() - 1) * (w() - 20)) + 10;
            int y2 = y() + h() - 8 - static_cast<int>(val_loss_history[i] / max_loss * (h() - 32));
            fl_line(x1, y1, x2, y2);
        }

        fl_color(140, 140, 150);
        fl_rect(x(), y(), w(), h());
    }
};

class AppWindow : public Fl_Double_Window {
private:
    NeuralNetwork net;
    vector<Point2D> dataset_points;

    DecisionBoundaryWidget* canvas;
    LossGraphWidget* loss_graph;

    Fl_Choice* dataset_choice;
    Fl_Value_Slider* lr_slider;
    Fl_Value_Slider* epochs_slider;
    Fl_Choice* hidden_neurons_choice;

    Fl_Button* btn_train;
    Fl_Button* btn_reset;

    Fl_Input* input_x;
    Fl_Input* input_y;
    Fl_Button* btn_predict;
    Fl_Box* pred_result_box;

    Fl_Box* status_box;

    int current_epoch;
    int hidden_neurons;

    int target_epochs_remaining;
    bool is_training_animating;

    bool has_active_prediction;
    float current_pred_x;
    float current_pred_y;

public:
    AppWindow(int W, int H, const char* title)
        : Fl_Double_Window(W, H, title), current_epoch(0), hidden_neurons(16),
          target_epochs_remaining(0), is_training_animating(false),
          has_active_prediction(false), current_pred_x(0.0f), current_pred_y(0.0f) {

        color(fl_rgb_color(36, 40, 48));

        canvas = new DecisionBoundaryWidget(20, 20, 660, 660, &net, &dataset_points);
        canvas->setOnClickCallback([this](float x, float y) {
            this->onCanvasClicked(x, y);
        });

        int panel_x = 700;
        int panel_y = 20;
        int panel_w = 440;

        Fl_Box* header = new Fl_Box(panel_x, panel_y, panel_w, 32, "2D Neural Network Classifier");
        header->labelsize(20);
        header->labelfont(FL_HELVETICA_BOLD);
        header->labelcolor(FL_WHITE);

        panel_y += 42;
        Fl_Box* grp1_box = new Fl_Box(panel_x, panel_y, panel_w, 175);
        grp1_box->box(FL_ROUNDED_BOX);
        grp1_box->color(fl_rgb_color(28, 31, 38));

        int inner_y = panel_y + 14;
        Fl_Box* title1 = new Fl_Box(panel_x + 15, inner_y, panel_w - 30, 20, "NETWORK & DATASET CONFIGURATION");
        title1->labelsize(12);
        title1->labelfont(FL_HELVETICA_BOLD);
        title1->labelcolor(fl_rgb_color(160, 175, 200));
        title1->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        inner_y += 28;
        Fl_Box* lbl_data = new Fl_Box(panel_x + 15, inner_y, 110, 26, "Dataset:");
        lbl_data->labelcolor(FL_WHITE);
        lbl_data->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        dataset_choice = new Fl_Choice(panel_x + 130, inner_y, 295, 26);
        dataset_choice->add("Spiral (X.csv & y.csv)");
        dataset_choice->add("Generated Spiral");
        dataset_choice->add("Concentric Circles");
        dataset_choice->add("XOR Quadrants");
        dataset_choice->value(0);
        dataset_choice->callback(onDatasetChanged, this);

        inner_y += 42;
        Fl_Box* lbl_nodes = new Fl_Box(panel_x + 15, inner_y, 110, 26, "Hidden Nodes:");
        lbl_nodes->labelcolor(FL_WHITE);
        lbl_nodes->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        hidden_neurons_choice = new Fl_Choice(panel_x + 130, inner_y, 295, 26);
        hidden_neurons_choice->add("4 Neurons");
        hidden_neurons_choice->add("8 Neurons");
        hidden_neurons_choice->add("12 Neurons");
        hidden_neurons_choice->add("16 Neurons");
        hidden_neurons_choice->value(3);
        hidden_neurons_choice->callback(onArchitectureChanged, this);

        inner_y += 42;
        lr_slider = new Fl_Value_Slider(panel_x + 15, inner_y, 410, 24, "Learning Rate");
        lr_slider->type(FL_HOR_SLIDER);
        lr_slider->bounds(0.001, 0.5);
        lr_slider->value(0.10);
        lr_slider->labelcolor(FL_WHITE);
        lr_slider->align(FL_ALIGN_TOP_LEFT);

        panel_y += 190;
        Fl_Box* grp2_box = new Fl_Box(panel_x, panel_y, panel_w, 145);
        grp2_box->box(FL_ROUNDED_BOX);
        grp2_box->color(fl_rgb_color(28, 31, 38));

        inner_y = panel_y + 14;
        Fl_Box* title2 = new Fl_Box(panel_x + 15, inner_y, panel_w - 30, 20, "TRAINING CONTROLS");
        title2->labelsize(12);
        title2->labelfont(FL_HELVETICA_BOLD);
        title2->labelcolor(fl_rgb_color(160, 175, 200));
        title2->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        inner_y += 32;
        epochs_slider = new Fl_Value_Slider(panel_x + 15, inner_y, 410, 24, "Epochs per TRAIN Press");
        epochs_slider->type(FL_HOR_SLIDER);
        epochs_slider->bounds(10, 2000);
        epochs_slider->step(10);
        epochs_slider->value(200);
        epochs_slider->labelcolor(FL_WHITE);
        epochs_slider->align(FL_ALIGN_TOP_LEFT);

        inner_y += 48;
        btn_train = new Fl_Button(panel_x + 15, inner_y, 200, 36, "TRAIN");
        btn_train->callback(onTrainClicked, this);
        btn_train->color(fl_rgb_color(40, 160, 90));
        btn_train->labelcolor(FL_WHITE);
        btn_train->labelfont(FL_HELVETICA_BOLD);

        btn_reset = new Fl_Button(panel_x + 225, inner_y, 200, 36, "Reset Weights");
        btn_reset->callback(onResetClicked, this);
        btn_reset->color(fl_rgb_color(200, 80, 80));
        btn_reset->labelcolor(FL_WHITE);

        panel_y += 160;
        Fl_Box* grp3_box = new Fl_Box(panel_x, panel_y, panel_w, 185);
        grp3_box->box(FL_ROUNDED_BOX);
        grp3_box->color(fl_rgb_color(28, 31, 38));

        inner_y = panel_y + 12;
        status_box = new Fl_Box(panel_x + 15, inner_y, 410, 36, "Ep: 0 | Tr: -- | Val: --");
        status_box->box(FL_ROUNDED_BOX);
        status_box->color(fl_rgb_color(20, 23, 28));
        status_box->labelcolor(fl_rgb_color(100, 220, 140));
        status_box->labelsize(13);
        status_box->labelfont(FL_HELVETICA_BOLD);

        inner_y += 44;
        loss_graph = new LossGraphWidget(panel_x + 15, inner_y, 410, 115);

        panel_y += 200;
        Fl_Box* grp4_box = new Fl_Box(panel_x, panel_y, panel_w, 150);
        grp4_box->box(FL_ROUNDED_BOX);
        grp4_box->color(fl_rgb_color(28, 31, 38));

        inner_y = panel_y + 14;
        Fl_Box* title4 = new Fl_Box(panel_x + 15, inner_y, panel_w - 30, 20, "INTERACTIVE POINT PREDICTION PROBE");
        title4->labelsize(12);
        title4->labelfont(FL_HELVETICA_BOLD);
        title4->labelcolor(fl_rgb_color(160, 175, 200));
        title4->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        inner_y += 28;
        Fl_Box* lbl_x = new Fl_Box(panel_x + 15, inner_y, 22, 28, "X:");
        lbl_x->labelcolor(FL_WHITE);
        input_x = new Fl_Input(panel_x + 40, inner_y, 75, 28);
        input_x->value("0.00");

        Fl_Box* lbl_y = new Fl_Box(panel_x + 130, inner_y, 22, 28, "Y:");
        lbl_y->labelcolor(FL_WHITE);
        input_y = new Fl_Input(panel_x + 155, inner_y, 75, 28);
        input_y->value("0.00");

        btn_predict = new Fl_Button(panel_x + 245, inner_y, 180, 28, "PREDICT");
        btn_predict->callback(onPredictClicked, this);
        btn_predict->color(fl_rgb_color(70, 130, 220));
        btn_predict->labelcolor(FL_WHITE);
        btn_predict->labelfont(FL_HELVETICA_BOLD);

        inner_y += 36;
        pred_result_box = new Fl_Box(panel_x + 15, inner_y, 410, 56, "Click canvas or enter X,Y to predict probabilities");
        pred_result_box->box(FL_ROUNDED_BOX);
        pred_result_box->color(fl_rgb_color(20, 23, 28));
        pred_result_box->labelcolor(fl_rgb_color(230, 230, 240));
        pred_result_box->labelsize(12);

        end();

        loadSelectedDataset();
        rebuildNetwork();
    }

    void rebuildNetwork() {
        if (is_training_animating) {
            Fl::remove_timeout(trainAnimTimerCallback, this);
            is_training_animating = false;
        }
        target_epochs_remaining = 0;
        net.clearLayers();
        
        net.addLayer(make_shared<DenseLayer>(2, hidden_neurons));
        net.addLayer(make_shared<ReLULayer>());

        net.addLayer(make_shared<DenseLayer>(hidden_neurons, hidden_neurons));
        net.addLayer(make_shared<ReLULayer>());

        net.addLayer(make_shared<DenseLayer>(hidden_neurons, 3));
        net.addLayer(make_shared<SoftmaxLayer>());

        current_epoch = 0;
        has_active_prediction = false;
        canvas->clearPredictionMarker();
        loss_graph->clear();
        updateStatusDisplay(0.0f, 0.0f, 0.0f, 0.0f);
        pred_result_box->copy_label("Click canvas or enter X,Y to predict probabilities");
        canvas->redraw();
    }

    void loadSelectedDataset() {
        int val = dataset_choice->value();
        if (val == 0) {
            if (!loadCSVData("data/X.csv", "data/y.csv", dataset_points)) {
                cout << "[Notice] X.csv/y.csv not found, generating spiral dataset...\n";
                generateSpiralData(dataset_points);
            }
        } else if (val == 1) {
            generateSpiralData(dataset_points);
        } else if (val == 2) {
            generateCirclesData(dataset_points);
        } else if (val == 3) {
            generateXORData(dataset_points);
        }
        rebuildNetwork();
    }

    void startAnimatedTraining(int additional_epochs) {
        if (dataset_points.empty()) return;

        target_epochs_remaining += additional_epochs;
        if (!is_training_animating) {
            is_training_animating = true;
            Fl::add_timeout(0.016, trainAnimTimerCallback, this);
        }
    }

    static void trainAnimTimerCallback(void* data) {
        AppWindow* app = static_cast<AppWindow*>(data);
        if (!app->is_training_animating || app->target_epochs_remaining <= 0) {
            app->is_training_animating = false;
            return;
        }

        Matrix X_train, X_val;
        vector<int> y_train, y_val;
        pointsToTrainValMatrices(app->dataset_points, X_train, y_train, X_val, y_val);

        float lr = (float)app->lr_slider->value();
        float train_loss = 0.0f, train_acc = 0.0f;
        float val_loss = 0.0f, val_acc = 0.0f;

        int batch = min(10, app->target_epochs_remaining);
        for (int i = 0; i < batch; i++) {
            auto train_res = app->net.trainStep(X_train, y_train, lr);
            train_loss = train_res.first;
            train_acc = train_res.second;

            auto val_res = app->net.evaluate(X_val, y_val);
            val_loss = val_res.first;
            val_acc = val_res.second;

            app->current_epoch++;
            app->target_epochs_remaining--;
        }

        app->loss_graph->addLoss(train_loss, val_loss);
        app->updateStatusDisplay(train_loss, train_acc, val_loss, val_acc);
        app->canvas->redraw();

        if (app->has_active_prediction) {
            app->predictPoint(app->current_pred_x, app->current_pred_y);
        }

        if (app->target_epochs_remaining > 0) {
            Fl::repeat_timeout(0.016, trainAnimTimerCallback, data);
        } else {
            app->is_training_animating = false;
        }
    }

    void predictPoint(float px, float py) {
        px = clamp(px, -2.0f, 2.0f);
        py = clamp(py, -2.0f, 2.0f);

        current_pred_x = px;
        current_pred_y = py;
        has_active_prediction = true;

        static char str_x[32], str_y[32];
        snprintf(str_x, sizeof(str_x), "%.2f", px);
        snprintf(str_y, sizeof(str_y), "%.2f", py);
        input_x->value(str_x);
        input_y->value(str_y);

        Matrix in_pt(1, 2);
        in_pt(0, 0) = px;
        in_pt(0, 1) = py;

        Matrix preds = net.forward(in_pt);
        float p0 = preds(0, 0);
        float p1 = preds(0, 1);
        float p2 = preds(0, 2);

        int best_class = 0;
        float max_prob = p0;
        if (p1 > max_prob) { max_prob = p1; best_class = 1; }
        if (p2 > max_prob) { max_prob = p2; best_class = 2; }

        const char* class_names[] = {"Class 0 (Red)", "Class 1 (Green)", "Class 2 (Blue)"};

        static char result_buf[256];
        snprintf(result_buf, sizeof(result_buf),
                 "C0 (Red): %.1f%% | C1 (Green): %.1f%% | C2 (Blue): %.1f%%\n=> Predicted: %s (%.1f%% Confidence)",
                 p0 * 100.0f, p1 * 100.0f, p2 * 100.0f,
                 class_names[best_class], max_prob * 100.0f);

        pred_result_box->copy_label(result_buf);
        canvas->setPredictionMarker(px, py);
    }

    void updateStatusDisplay(float t_loss, float t_acc, float v_loss, float v_acc) {
        static char buf[128];
        snprintf(buf, sizeof(buf), "Ep: %d | Tr: %.3f (%.0f%%) | Val: %.3f (%.0f%%)", 
                 current_epoch, t_loss, t_acc * 100.0f, v_loss, v_acc * 100.0f);
        status_box->copy_label(buf);
    }

    void onCanvasClicked(float px, float py) {
        predictPoint(px, py);
    }

    static void onDatasetChanged(Fl_Widget*, void* data) {
        static_cast<AppWindow*>(data)->loadSelectedDataset();
    }

    static void onArchitectureChanged(Fl_Widget*, void* data) {
        AppWindow* app = static_cast<AppWindow*>(data);
        int idx = app->hidden_neurons_choice->value();
        int neurons[] = {4, 8, 12, 16};
        app->hidden_neurons = neurons[idx];
        app->rebuildNetwork();
    }

    static void onTrainClicked(Fl_Widget*, void* data) {
        AppWindow* app = static_cast<AppWindow*>(data);
        int add_epochs = static_cast<int>(app->epochs_slider->value());
        app->startAnimatedTraining(add_epochs);
    }

    static void onResetClicked(Fl_Widget*, void* data) {
        static_cast<AppWindow*>(data)->rebuildNetwork();
    }

    static void onPredictClicked(Fl_Widget*, void* data) {
        AppWindow* app = static_cast<AppWindow*>(data);
        float px = 0.0f, py = 0.0f;
        try {
            if (app->input_x->value()) px = stof(app->input_x->value());
            if (app->input_y->value()) py = stof(app->input_y->value());
        } catch (...) {
            px = 0.0f; py = 0.0f;
        }
        app->predictPoint(px, py);
    }
};

int main(int argc, char** argv) {
    srand(static_cast<unsigned int>(time(NULL)));

    if (argc > 1 && string(argv[1]) == "--test") {
        cout << "[Test Mode] Running training verification on spiral dataset...\n";
        vector<Point2D> points;
        if (!loadCSVData("data/X.csv", "data/y.csv", points)) {
            generateSpiralData(points);
        }
        NeuralNetwork net;
        net.addLayer(make_shared<DenseLayer>(2, 16));
        net.addLayer(make_shared<ReLULayer>());
        net.addLayer(make_shared<DenseLayer>(16, 16));
        net.addLayer(make_shared<ReLULayer>());
        net.addLayer(make_shared<DenseLayer>(16, 3));
        net.addLayer(make_shared<SoftmaxLayer>());

        NeuralNetworkDebugger::inspectLayers(net);

        Matrix X;
        vector<int> y;
        pointsToMatrix(points, X, y);

        float loss = 0.0f, acc = 0.0f;
        for (int i = 0; i <= 2500; i++) {
            auto res = net.trainStep(X, y, 0.1f);
            loss = res.first;
            acc = res.second;
            if (i % 500 == 0) {
                cout << "Epoch " << i << " | Loss: " << loss << " | Accuracy: " << (acc * 100.0f) << "%\n";
            }
        }
        cout << "[Test Mode] Verification complete. Final Accuracy: " << (acc * 100.0f) << "%\n";
        return 0;
    }

    AppWindow* win = new AppWindow(1160, 740, "Interactive 2D Neural Network Classifier (FLTK)");
    win->show(argc, argv);

    return Fl::run();
}
