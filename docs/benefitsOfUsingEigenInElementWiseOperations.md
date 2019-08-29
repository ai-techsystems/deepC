# Usage and performance benefits on using Eigen array method in element wise operations

## Below is a snippet code only for **2D**. One uses Eigen, and another just uses loop.

<details>
<summary>With Eigen</summary>

```cpp
tensor<T> eigen_compute(tensor<T> &a, tensor<T> &b){
    
    if (a.shape() != b.shape())
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Div operator.");
    if (a.rank() == 2 && b.rank() == 2) {
    
      tensor<T> result(a.shape()[0], b.shape()[1]);

      DNNC_EIGEN_MATRIX(eigenMatrixA, a);
      DNNC_EIGEN_MATRIX(eigenMatrixB, b);

      Matrix<T, Dynamic, Dynamic, RowMajor> eResult =
          eigenMatrixA.array() / eigenMatrixB.array();

      result.load(eResult.data());
      return result;
    }
    return tensor<T>();
  }

```
</details>

<details>
<summary>Without Eigen</summary>

```cpp
tensor<T> without_eigen_compute(tensor<T> &a, tensor<T> &b) {
    if (a.shape() != b.shape())
      throw std::invalid_argument(
          "tensor dimenions not appropriate for Div operator.");

    tensor<T> result(a.shape(), a.name());
    for (size_t i = 0; i < a.length(); i++)
      result[i] = a[i] / b[i];

    return result;
  }
```
</details>

## Now let's see the performance

<details>
<summary>Random array generation funtion</summary>

```cpp
void generate_random(float* a,int size){
  srand(time(0)); 
  int i;
  for (i=0;i<size;i++){
    a[i]=rand();
  }
}
```
</details>

### Going with relatively small matrix
<details>
<summary>Small matrix input</summary>

```cpp
int main() {
  float d1[100],d2[100];
  generate_random(d1,100);
  generate_random(d2,100);

  tensor<float> a(10, 10);
  a.load(d1);
  tensor<float> b(10, 10);
  b.load(d2);
  Div<float> m("localOpName");

  clock_t t;
  
  t = clock();
  auto result_1 = m.without_eigen_compute(a, b);
  t = clock() - t;
  double time_taken_1 = ((double)t)/CLOCKS_PER_SEC;
  
  t = clock();
  auto result_2 = m.eigen_compute(a, b);
  t = clock() - t;
  double time_taken_2 = ((double)t)/CLOCKS_PER_SEC;
  
  std::cout << time_taken_1 << " seconds took without eigen " << std::endl;
  std::cout << time_taken_2 << " seconds took with eigen" << std::endl;

  return 0;
}
```
</details>

#### Here Eigen is **~10x** faster than looping
### Going with relatively large matrix
<details>
<summary>Large matrix input</summary>

```cpp
int main() {
  float d1[1000000],d2[1000000];
  generate_random(d1,1000000);
  generate_random(d2,1000000);

  tensor<float> a(1000, 1000);
  a.load(d1);
  tensor<float> b(1000, 1000);
  b.load(d2);
  Div<float> m("localOpName");

  clock_t t;
  
  t = clock();
  auto result_1 = m.without_eigen_compute(a, b);
  t = clock() - t;
  double time_taken_1 = ((double)t)/CLOCKS_PER_SEC;
  
  t = clock();
  auto result_2 = m.eigen_compute(a, b);
  t = clock() - t;
  double time_taken_2 = ((double)t)/CLOCKS_PER_SEC;
  
  std::cout << time_taken_1 << " seconds took without eigen " << std::endl;
  std::cout << time_taken_2 << " seconds took with eigen" << std::endl;

  return 0;
```
</details>

#### Here Eigen is **~2x** faster than looping

## Eigen is excellent in memory handling and efficiency, rather than us looping through the tensor.
