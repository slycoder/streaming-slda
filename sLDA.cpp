#include <assert.h>
#include <boost/math/special_functions/digamma.hpp>
#include <google/sparse_hash_map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define RUNIF() drand48()
#define RUNIF_SEED(s) srand48(s)

using boost::math::digamma;
using google::sparse_hash_map;
using std::string;

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

  void increment(unsigned int index, unsigned int topic) {
    data_[index * num_topics_ + topic]++;
  }

  void decrement(unsigned int index, unsigned int topic) {
    data_[index * num_topics_ + topic]--;
  }

private:
  unsigned int num_topics_;
  unsigned int size_;
  int* data_;
};

template <typename T>
class SparseMatrix {
public:
  typedef sparse_hash_map<
      unsigned int, T
  > innerDataType;

  typedef sparse_hash_map<
      unsigned int, innerDataType
  > dataType;

  explicit SparseMatrix(T fallback) : fallback_(fallback) {
  }

  const T get(unsigned int index, unsigned int topic) const {
    typename dataType::const_iterator iter = data_.find(index);

    if (iter == data_.end()) {
      return fallback_;
    }

    typename innerDataType::const_iterator iter2 =
      iter->second.find(topic);

    if (iter2 == iter->second.end()) {
      return fallback_;
    }
    return iter2->second;
  }

  void clear() {
    data_.clear();
  }

  void set(unsigned int index, unsigned int topic, const T& val) {
    data_[index][topic] = val;
  }

  void increment(unsigned int index, unsigned int topic) {
    data_[index][topic]++;
  }

  void decrement(unsigned int index, unsigned int topic) {
    data_[index][topic]--;
  }

  const dataType& getData() const {
    return data_;
  }

  template <typename U>
  void add(const SparseMatrix<U>& other, double a, double b) {
    for (typename SparseMatrix<U>::dataType::const_iterator iter = other.getData().begin();
         iter != other.getData().end();
         ++iter) {
      for (typename SparseMatrix<U>::innerDataType::const_iterator iter2 = iter->second.begin();
           iter2 != iter->second.end();
           ++iter2) {
        data_[iter->first][iter2->first] =
          a * data_[iter->first][iter2->first] + b * iter2->second;
      }
    }
  }

protected:
  T fallback_;
  dataType data_;
};

template <typename T>
class Vector {
public:
  explicit Vector(unsigned int size) :
    size_(size),
    alloced_size_(size) {
    data_ = new T[size_];
  }

  ~Vector() {
    delete data_;
  }

  const T& get(unsigned int index) const {
    return data_[index];
  }

  T& get(unsigned int index) {
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

  unsigned int size() const {
    return size_;
  }

  void resize(unsigned int new_size) {
    if (new_size > alloced_size_) {
      delete data_;
      data_ = new T[new_size];
      alloced_size_ = new_size;
    }
    size_ = new_size;
  }

private:
  unsigned int size_;
  unsigned int alloced_size_;
  T* data_;
};

class RaggedArray : public Vector<Vector<int>*> {
public:
  explicit RaggedArray(unsigned int size) :
    Vector<Vector<int>*>(size) {
    for (unsigned int ii = 0; ii < size; ++ii) {
      Vector<Vector<int>*>::get(ii) = new Vector<int>(0);
    }
  }

  ~RaggedArray() {
    for (unsigned int ii = 0; ii < size(); ++ii) {
      delete Vector<Vector<int>*>::get(ii);
    }
    // Delete data as well?
  }

  Vector<int>& get(unsigned int index) const {
    return *Vector<Vector<int>*>::get(index);
  }

  int get(unsigned int index, unsigned int index2) const {
    return get(index).get(index2);
  }

  int set(unsigned int index, unsigned int index2, int val) {
    return get(index).set(index2, val);
  }

  void deserialize(FILE* f) {
    for (unsigned int ii = 0; ii < size(); ++ii) {
      get(ii).deserialize(f);
    }
  }

  void serialize(FILE *f) {
    for (unsigned int ii = 0; ii < size(); ++ii) {
      get(ii).serialize(f);
    }
  }
};

typedef Vector<int> Document;
typedef RaggedArray Corpus;

class Model {
public:
  explicit Model(unsigned int num_topics,
                 unsigned int V,
                 unsigned int batch_size,
                 double sigma2,
                 double document_smoothing,
                 double topic_smoothing) :
    num_topics_(num_topics),
    V_(V),
    sigma2_(sigma2),
    topics_(topic_smoothing),
    lambda_(FIXME),
    probs_(num_topics),
    batch_size_(batch_size),
    document_sums_(batch_size, num_topics),
    topic_counts_(0),
    document_smoothing_(document_smoothing),
    topic_smoothing_(topic_smoothing),
    coefs_(num_topics),
    assignments_(batch_size) {
  }

  // TODO: Initialization (do it in R?)

  void serialize(const string& prefix) {
    FILE* f;
    f = fopen((prefix + ".topics").c_str(), "wb");
    topics_.serialize(f);
    fclose(f);

    f = fopen((prefix + ".document_sums").c_str(), "wb");
    document_sums_.serialize(f);
    fclose(f);

    f = fopen((prefix + ".coefs").c_str(), "wb");
    coefs_.serialize(f);
    fclose(f);

    f = fopen((prefix + ".assignments").c_str(), "wb");
    assignments_.serialize(f);
    fclose(f);
  }

  void deserialize(const string& prefix) {
    FILE* f;
    f = fopen((prefix + ".topics").c_str(), "rb");
    topics_.deserialize(f);
    fclose(f);

    f = fopen((prefix + ".document_sums").c_str(), "rb");
    document_sums_.deserialize(f);
    fclose(f);

    f = fopen((prefix + ".coefs").c_str(), "rb");
    coefs_.deserialize(f);
    fclose(f);

    f = fopen((prefix + ".assignments").c_str(), "rb");
    assignments_.deserialize(f);
    fclose(f);
  }

  void inferDocumentOnce(const Document& doc,
                         unsigned int batch_index,
                         double y) {
    double y_hat = 0.0;
    for (unsigned int kk = 0; kk < num_topics_; ++kk) {
      y_hat += coefs_.get(kk) * document_sums_.get(batch_index, kk);
    }
    y_hat /= doc.size();

    for (unsigned int ii = 0; ii < doc.size(); ++ii) {
      int word = doc.get(word);
      int old_k = assignments_.get(batch_index, ii);
      document_sums_.decrement(batch_index, old_k);
      y_hat -= coefs_.get(old_k) / doc.size();
      topic_counts_.decrement(word, old_k);

      // Compute un-normalized topic probabilities.
      double p_sum = 0.0;
      for (unsigned int kk = 0; kk < num_topics_; ++kk) {
        double delta_k = coefs_.get(kk) / doc.size();
        double slda_part = exp(delta_k / sigma2_ *
                               ((y - y_hat) - delta_k / 2));
        probs_.set(
          kk,
          slda_part *
          (document_sums_.get(batch_index, kk) + document_smoothing_) *
          topics_.get(word, kk)
        );
        p_sum += probs_.get(kk);
      }

      // Sample a new assignment.
      double r = RUNIF();
      unsigned int new_k;
      for (new_k = 0; new_k < num_topics_; ++new_k) {
        if (r < probs_.get(new_k) / p_sum) {
          break;
        }
        r -= probs_.get(new_k) / p_sum;
      }
      assert(new_k != num_topics_);

      // Update assignment.
      assignments_.set(batch_index, ii, new_k);
      document_sums_.increment(batch_index, new_k);
      y_hat += coefs_.get(new_k) / doc.size();
      topic_counts_.increment(word, old_k);
    }
  }

  void updateTopicsFromLambda() {
    topics_.clear();
    // TODO: this should compute the digamma quantity.
  }

  void updateLambda(double corpus_size, double learning_rate) {
    double batch_ratio = corpus_size / batch_size_;

    lambda_.add(topic_counts_,
                (1 - learning_rate),
                learning_rate * batch_ratio);

    updateTopicsFromLambda();
  }

private:
  unsigned int num_topics_;
  unsigned int V_;
  double sigma2_;
  SparseMatrix<double> topics_;
  SparseMatrix<double> lambda_;
  Vector<double> probs_;
  unsigned int batch_size_;
  Matrix document_sums_;
  SparseMatrix<int> topic_counts_;
  double document_smoothing_;
  double topic_smoothing_;
  Vector<double> coefs_;
  RaggedArray assignments_;
};

class Learner {
public:
  explicit Learner(Model& model,
                   unsigned int batch_size,
                   unsigned int corpus_size,
                   double t0,
                   double kappa) :
    model_(model),
    corpus_(batch_size),
    corpus_size_(corpus_size),
    t0_(t0),
    kappa_(kappa) {
  }

  void processBatch(int iteration, FILE* f, int num_iterations) {
    corpus_.deserialize(f);
    // FIXME: Read annotations
    double y = 0.0;
    for (int ii = 0; ii < num_iterations; ++ii) {
      for (int dd = 0; dd < corpus_.size(); ++dd) {
        model_.inferDocumentOnce(corpus_.get(dd), dd, y);
      }
    }

    double learning_rate = pow(iteration + t0_, kappa_);
    model_.updateLambda(corpus_size_, learning_rate);
  }

  void processStream(FILE* f, int num_iterations) {
    int iteration = 0;
    while (!feof(f)) {
      processBatch(iteration, f, num_iterations);
      iteration++;
    }
  }

private:
  Model model_;
  Corpus corpus_;
  std::vector<double> annotations_;
  unsigned int corpus_size_;
  double t0_;
  double kappa_;
};

