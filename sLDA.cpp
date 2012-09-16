#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define RUNIF() drand48()
#define RUNIF_SEED(s) srand48(s)

// Needs to implement size() and get()
class Document {

};

class Matrix {
public:
  explicit Matrix(unsigned int size, unsigned int num_topics) :
    num_topics_(num_topics),
    size_(size) {
    data_ = new int[num_topics * size];
  }

  ~Matrix() {
    delete data_;
  }

  // Purposefully organized so that topics are near each other for better cache
  // behavior.
  int get(unsigned int index, unsigned int topic) const {
    return data_[index * num_topics_ + topic];
  }

  void deserialize(FILE* f) {
    fread(data_, sizeof(int), size_ * num_topics_, f);
  }

  void serialize(FILE* f) {
    fwrite(data_, sizeof(int), size_ * num_topics_, f);
  }
private:
  unsigned int num_topics_;
  unsigned int size_;
  int* data_;
};

template <typename T>
class Vector {
public:
  explicit Vector(unsigned int size) :
    size_(size) {
    data_ = new T[size_];
  }

  ~Vector() {
    delete data_;
  }

  T get(unsigned int index) const {
    return data_[index];
  }

  T set(unsigned int index, const T& value) {
    return data_[index] = value;
  }

  void deserialize(FILE* f) {
    fread(data_, sizeof(T), size_, f);
  }

  void serialize(FILE* f) {
    fwrite(data_, sizeof(T), size_, f);
  }

private:
  unsigned int size_;
  T* data_;
};

class Model {
public:
  explicit Model(unsigned int num_topics,
                 unsigned int V,
                 unsigned int batch_size,
                 double document_smoothing,
                 double topic_smoothing) :
    num_topics_(num_topics),
    V_(V),
    topics_(V, num_topics),
    probs_(num_topics),
    batch_size_(batch_size),
    document_sums_(batch_size, num_topics),
    topic_sums_(num_topics),
    document_smoothing_(document_smoothing),
    topic_smoothing_(topic_smoothing) {
  }

  void inferDocumentOnce(const Document& doc, unsigned int batch_index) {
    // Need to initialize document_sums_;
    // Should we assume topics_ are fixed?

    for (int ii = 0; ii < doc.size(); ++ii) {
      int word = doc.get(word);

      // Compute un-normalized topic probabilities.
      double p_sum = 0.0;
      for (int kk = 0; kk < num_topics_; ++kk) {
        probs_.set(
          kk,
          (document_sums_.get(batch_index, kk) + document_smoothing_) *
          (topics_.get(word, kk) + topic_smoothing_) /
          topic_sums_.get(V_ * topic_smoothing_)
        );
        p_sum += probs_.get(kk);
      }

      // Sample a new assignment.
      double r = RUNIF();
      int new_k;
      for (new_k = 0; new_k < num_topics_; ++new_k) {
        if (r < probs_.get(new_k) / p_sum) {
          break;
        }
        r -= probs_.get(new_k) / p_sum;
      }
      assert(new_k != num_topics_);

      // Update counts.


    }
  }

private:
  unsigned int num_topics_;
  unsigned int V_;
  Matrix topics_;
  Vector<double> probs_;
  unsigned int batch_size_;
  Matrix document_sums_;
  Vector<int> topic_sums_;
  double document_smoothing_;
  double topic_smoothing_;
};
