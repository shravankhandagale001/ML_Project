# Machine Learning 

**Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.**


## 🧠Supervised Machine Learning

**What is Supervised Machine Learning?**
>*✔Supervised machine learning or more commonly, supervised learning, refers to algorithms that learn x to y or input to output mappings. The key characteristic of supervised learning is that you give your learning algorithm examples to learn from. That includes the right answers, whereby right answer, I mean, the correct label y for a given input x, and is by seeing correct pairs of input x and desired output label y that the learning algorithm eventually learns to take just the input alone without the output label and gives a reasonably accurate prediction or guess of the output.* 

**1. Regression**
  >*It is an algorithm which predicts a value from infinitely possible values. For example predicting house prices *

**2. Classification**
  >*Classification is a type of supervised learning where the goal is to predict a category or class label for a given input based on previously labeled data.For example (spam or not spam), Image recognition*

### 📈Linear Regression Model(Single Variable)
>*📑Linear regression with one variable (also called univariate linear regression) is a supervised learning algorithm that models the linear relationship between a single input feature x and a continuous output y.

It finds the best-fitting straight line that predicts y from x.*

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

$$ (x^{(i)},y^{(i)}) $$ = Training Examples
- w = parameter (weight)
- b = parameter (bias)
- i = index of training example
- x = input feature
- y = target/output variable
- f(x) = prediction/hypothesis function

#### Cost Function 
>* A cost function is a mathematical function that measures the difference between the predicted output and the actual output. It tells us how well the model is performing — the smaller the cost, the better the model's predictions are.*

>**🧠 Intuition:**
- Think of the cost function as a "badness" score:

- If the predictions are perfect → cost = 0

- If the predictions are far from the actual values → cost is high

>**Linear Regression Cost Function (MSE)**
#####   ✏**Formula**
$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
$$
- J(θ): Cost function
- m: number of training examples
- $\hat{y}^{(i)}$ = estimated Value
- $y^{(i)}$ = True Value/Actual Value
- Loss: error for a single training example
  
![Line Plot](images/cost.png)
![Line Plot](images/cost1.png)
>**Contour Plot For finding Best values of w and b for the model**
![Line Plot](images/c3.png)

📋 Summary Table of Cost Function Visualization


| 🔍 Plot               | 🧠 Purpose                          | 📌 Insight                                                                 |
|-----------------------|------------------------------------|----------------------------------------------------------------------------|
| Top-Left (Line Plot)  | Model prediction using \( f(x) \)  | Shows how the model with \( w = 0.13 \) and \( b = 71 \) fits the data     |
| Top-Right (Contour)   | Cost contours of \( J(w, b) \)     | Visualizes where the cost is low or high; optimal point at cost minimum   |
| Bottom (3D Surface)   | 3D plot of cost \( J(w, b) \)       | Height shows cost value; best-fit parameters are at the lowest valley      |

🔁 What’s Happening?
- As you change the weight w, the line shifts on the left plot.
- This changes how closely it fits the data — altering the error/cost.
- The right plot visually shows how well or poorly a particular w performs in terms of cost.

✅ Key Insight:
- Finding the minimum of J(w) is equivalent to finding the best possible line that fits the data.

#### Gradient Descent 
>**Gradient Descent is one of the most important optimization algorithms in machine learning. It's used to minimize a cost function (also called loss function) by updating the model parameters (like weights in linear regression) iteratively.**

**Outline!**
- Start with some value of (w,b).
- Keep changing (w,b) to reduce J(w,b).
- Until we settle at or near a minimum



>**Gradient Descent Operation**
![Line Plot](images/today.png)

>*In this graph your each baby step leads you towards the one of the local minima(Cost function) . It depends upon the first step of the Gradient descent that will lead to the local minima and once you reached the local minima you cant go back to the another local minima.*

✏**Formula**
>**\[
w := w - \alpha \frac{\partial J}{\partial w}
\]
\[
b := b - \alpha \frac{\partial J}{\partial b}
\]**

Where:
- \( \alpha \) is the **learning rate**
- \( \frac{\partial J}{\partial w} \), \( \frac{\partial J}{\partial b} \): gradients of the cost function

##### ⚙️ **Steps in Gradient Descent**

![Line Plot](images/today1.png)
1. Initialize \( w \) and \( b \)
2. Compute the gradients
3. Update parameters:
   - \( w := w - \alpha \frac{\partial J}{\partial w} \)
   - \( b := b - \alpha \frac{\partial J}{\partial b} \)
4. Repeat until convergence

---
*📍Important Graph*
![Line Plot](images/today2.png)

‼**Importance of Learning Rate(\(\alpha)\)**
![Line Plot](images/today3.png)
1. **If (\(\alpha)\) is to small** - The Learning algorithm will work but it will take a very long time bcoz its going to take these tiny baby steps.
2. **If (\(\alpha)\) is to large** - Overshoot, never touch minimum.
3. **If w is already at local minima?**
   ![Line Plot](images/today4.png)
   If parameter (w) is already at local minima then the further gradient descent will not change the parameter and the solution remains the same.

4. **If Learning Rate is fixed** -
  ![Line Plot](images/today5.png)


#### 🔗 Relationship Summary

- 🧮 **Linear Regression**:
  - Predicts output using a linear equation: \( \hat{y} = wx + b \)

- 📉 **Cost Function**:
  - Measures prediction error: \( J(w, b) \)

- 🔁 **Gradient Descent**:
  - Minimizes the cost function by updating \( w \), \( b \)

### 🔄 Flow:
Linear Regression → Cost Function → Gradient Descent (to optimize)



  

## 🧠Unsupervised Machine Learning -->

 **What is Unsupervised Machine Learning?**
 >*✔Unsupervised machine learning is a type of machine learning where the model is trained on data without labeled outputs. The algorithm tries to find patterns, groupings, or structures in the data without being told what to look for.*
- No labeled data (no target/output variable)

- Learns hidden patterns or intrinsic structures

- Often used for exploratory data analysis, clustering, and dimensionality reduction

**1. Clustering**
>*Groups similar data's into single category* --> -->


