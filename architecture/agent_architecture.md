### **Step 1: Generate N Diverse Responses from the Language Model**

* Prompt the language model **N times** with a non-zero temperature to produce **variation in responses**.
* Each generated response should include:

  * A **binary classification**: for example, `1 = has reservations`, `0 = no reservations`.
  * A **textual rationale or explanation** for the classification (optional but useful for feature extraction).

---

### **Step 2: Process Each Sample Individually**

#### → **Extract the Predicted Class (0 or 1)**

* Apply a simple rule or regular expression to classify the response based on key phrases (e.g., `"no reservation"` → 0, `"a concern was raised"` → 1).

#### → **Analyze the Justification**

* Examine the rationale behind the classification to understand what triggered it:

  * What specific phrases or terms suggested the classification?
  * Does the reasoning rely on facts, speculation, or vague language?

---

### **Step 3: Score Each Response Using Lightweight Features**

To estimate the **confidence** of each classification, compute a score using a logistic regression model trained on simple, interpretable features:

| **Feature**                   | **Description**                                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **F1. Sentence Count**        | Measures the number of sentences in the response. Longer outputs may reflect more careful reasoning.                 |
| **F2. Trigger Word Presence** | Detects keywords like “concern,” “issue,” or “but” that are often associated with reservations.                      |
| **F3. Lexical Consistency**   | Computes Jaccard similarity between the input and the output to assess how well the response stays on topic.         |
| **F4. Format Validity**       | Checks if the output adheres to a consistent structure or format (e.g., begins with "Answer:" or ends with a label). |
| **F5. Answer Certainty**      | Counts hedging expressions such as “might,” “probably,” or “unclear,” which suggest lower confidence.                |
| **F6. Past Agreement**        | Measures how similar the current response is to previously seen high-confidence responses.                           |

---

### **Step 4: Aggregate Votes Using Confidence Scores**

For each response, add its predicted class to a running total, weighted by its confidence score:

<!-- ```python
votes = {0: 0.0, 1: 0.0}
votes[predicted_class] += confidence_score
``` -->

This forms a **soft voting mechanism**, where more confident responses contribute more heavily to the final decision.

---

### **Step 5: Apply Early Stopping Based on Confidence**

After each new response is added, evaluate whether the decision can be made confidently:

<!-- ```python
total = sum(votes.values())
top_class = max(votes, key=votes.get)

if votes[top_class] / total >= τ:
    return top_class
``` -->

* If the leading class accumulates a sufficient share of the total confidence (as defined by threshold $\tau$), the process stops and the classification is returned.
* Otherwise, sampling continues until the threshold is met or a maximum sample limit is reached.
