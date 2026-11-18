# Experiment 6 : Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim
To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset
A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed
1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## Program & Output
\# Import Required Python Libraries  
import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori, association_rules  

\# Create the Sample Grocery Transaction Dataset  
transactions = [  
    ['Milk', 'Bread', 'Eggs'],  
    ['Milk', 'Bread'],  
    ['Milk', 'Eggs'],  
    ['Bread', 'Butter'],  
    ['Milk', 'Bread', 'Butter', 'Eggs'],  
    ['Bread', 'Eggs'],  
    ['Milk', 'Butter'],  
    ['Milk', 'Bread', 'Butter'],  
    ['Bread', 'Butter', 'Eggs'],  
    ['Milk', 'Bread', 'Butter', 'Eggs']  
]  
  
\# Convert the Dataset into a One-Hot Encoded Binary Matrix using TransactionEncoder  
te = TransactionEncoder()  
te_ary = te.fit(transactions).transform(transactions)  
df = pd.DataFrame(te_ary, columns=te.columns_)  
  
print("--- 1. One-Hot Encoded Binary Matrix ---")  
print(df)  

\# Apply the Apriori Algorithm to Identify Frequent Itemsets (min_support = 0.3)  
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)  
print("\n--- 2. Frequent Itemsets (Support >= 0.3) ---")  
print(frequent_itemsets)  

\# Generate Association Rules from Frequent Itemsets    
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)    
print("\n--- 3. Generated Association Rules ---")    
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  

\# Filter Strong Rules Based on Thresholds (Confidence >= 0.6 and Lift > 1.0)  
strong_rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] > 1.0)]  
print("\n--- 4. Filtered Strong Association Rules ---")  
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  

<img width="772" height="746" alt="image" src="https://github.com/user-attachments/assets/e67b8284-d9f2-4aff-942c-5515e1ac0ba8" />

## Result
Thus to perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics successfully.
