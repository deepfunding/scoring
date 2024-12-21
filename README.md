# Scoring Mechanism for Deep Funding

## Introduction

This code is designed to help fairly distribute rewards among contributors based on multiple input evaluations and user preferences. By combining different assessment methods through optimization, we ensure that rewards reflect the collective input (given by AI and humans) and defined criteria.

It could be used to determine fair rewards for contributors by:

1. **Defining Distributions:**
   - **AI Assessment:** Evaluations from AI on each contributor's impact.
   - **Community Feedback:** Humans also evaluate each contributor's impact.
   - **Egalitarian Approach:** Equal distribution to ensure fairness.

2. **Gathering Comparisons:**
   - Other humans compare pairs of contributors for an achievement, indicating who deserves more recognition and by how much.

3. **Optimization Process:**
   - Combines the distributions to minimize discrepancies, balancing expert opinions, community feedback, and fairness.

4. **Final Reward Distribution:**
   - Applies optimized weights to allocate specific rewards to each contributor.


## Project Structure

The project consists of two main Python scripts:

- **`scoring.py`**: Contains functions to calculate the cost of a proposed distribution and find the optimal weights to combine multiple distributions.
- **`example.py`**: Provides an example scenario using characters from *The Lord of the Rings*.

## How It Works

### `scoring.py`

- **Cost Function (`cost_function`)**:
  
  This function calculates the sum of squared differences between the log-probabilities of item pairs as provided by user comparisons. It measures how well a proposed distribution aligns with the provided preferences.

- **Find Optimal Weights (`find_optimal_weights`)**:
  
  This function uses the `scipy.optimize.minimize` method to determine the best weights for combining multiple input distributions. The goal is to minimize the cost function while ensuring that the weights sum to 1 and each weight is between 0 and 1.

### `example.py`

- **Character List**:
  
  A predefined list of 20 characters from *The Lord of the Rings* used to simulate the distribution of credit for defeating Sauron.

- **Distributions**:
  
  Three different probability distributions are provided:
  
  1. **GPT Distribution**: A hypothetical distribution based on a language model's assessment.
  2. **User Opinion**: A manually defined distribution reflecting personal judgment.
  3. **Egalitarian Split**: An equal distribution of credit among all characters.

- **User Comparisons**:
  
  The script interactively gathers user preferences by asking the user to compare pairs of characters and specify how much more credit one deserves over the other.

- **Optimization and Results**:
  
  After collecting user comparisons, the script calculates the optimal weights for combining the distributions to best match the user's preferences. It then displays the costs of each individual distribution, the optimal weights, the resulting combined distribution, and the cost of this ideal distribution.

### Prerequisites

- **Python**: Ensure you have Python 3.6 or higher installed on your system.
- **Python Packages**: The project relies on the following Python libraries:
  - `numpy`
  - `scipy`


### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/deepfunding/scoring.git
   cd optimal-distribution-weighting
   ```

2. **Install Dependencies**:

   ```bash
  pip install numpy scipy
  ```

### Running the Example

1. **Execute the Example Script**:

   ```bash
   python example.py
   ```

2. **Provide User Comparisons**:

   The script will prompt you to compare pairs of characters based on who deserves more credit for helping defeat Sauron. For each comparison:
   
   - **Choose Between Two Characters**: Type '1' or '2' to indicate your preference.
   - **Specify the Magnitude**: Input a number representing how many times more credit one character deserves over the other.

   **Example Interaction**:

   ```
   Who deserves more credit for helping defeat Sauron? Gandalf or Aragorn?
   Type '1' for Gandalf or '2' for Aragorn: 2
   How many times more credit does the second deserve? Give a number (e.g., 3): 3
   ```

3. **View Results**:

   After providing your comparisons, the script will display:
   
   - **Cost of Each Pure Distribution**: Indicates how well each individual distribution aligns with your preferences.
   - **Optimal Weights**: The best combination of weights to minimize the cost.
   - **Lowest-Cost Distribution**: The final adjusted distribution based on optimal weighting.
   - **Cost of Ideal Distribution**: The minimized cost achieved by the optimal combination.
