from scoring import find_optimal_weights, cost_function
import random
import math

# Task (the 'parent node' in deep funding)
task = 'helping defeat Sauron'

# Character list (the 'child nodes' in deep funding)
characters = [
    "Frodo Baggins",
    "Samwise Gamgee",
    "Gandalf",
    "Aragorn",
    "Legolas",

    "Gimli",
    "Meriadoc Brandybuck",
    "Peregrin Took",
    "Boromir",
    "Faramir",

    "Éowyn",
    "Éomer",
    "Théoden",
    "Treebeard",
    "Elrond",

    "Galadriel",
    "Celeborn",
    "Arwen Undómiel",
    "Bilbo Baggins",
    "Tom Bombadil",
]

# Different submitted distributions for who deserves what share of credit.
# In a live deep funding implementation, anyone would be able to submit a
# distribution (realistically using AI to generate it) and it would go into
# this list.
#
# Note that it actually does not matter whether or not a list sums to 1;
# if the contributions sum to t != 1, then the result will be equivalent
# to the case where you instead provide [v/t for v in contributions]
#
# In a live deep funding round, anyone would be able to submit their own
# distribution.

distributions = {
    # GPT's answers
    'gpt': [
        0.25, 0.20, 0.15, 0.10, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02,
        0.03, 0.02, 0.02, 0.03, 0.01,
        0.02, 0.01, 0.01, 0.03, 0.01
    ],
    # Claude's answers
    'claude': [
        20, 18, 12, 8, 3,
        3, 2, 2, 2, 2,
        4, 2, 2, 3, 5,
        5, 1, 2, 3, 1
    ],
    # My own opinion from thinking about it for 1 minute
    'human': [
        0.17, 0.10, 0.10, 0.10, 0.05,
        0.05, 0.05, 0.05, 0.03, 0.03,
        0.01, 0.02, 0.02, 0.05, 0.02,
        0.02, 0.01, 0.01, 0.03, 0.01
    ],
    # Egalitarian split
    'egalitarian': [0.05] * 20,
}

# Convert credit distributions into a list of logits
logits = [[math.log(p) for p in dist] for dist in distributions.values()]

# Function to prompt user for pairwise comparisons
def gather_user_comparisons():
    samples = []
    for _ in range(5):
        # Randomly select two different names
        a, b = random.sample(range(len(characters)), 2)
        name_a, name_b = characters[a], characters[b]

        # Ask the user which deserves more credit
        print(f"Who deserves more credit for {task}? {name_a} or {name_b}?")
        choice = input(f"Type '1' for {name_a} or '2' for {name_b}: ").strip()

        # Ensure valid input
        while choice not in ['1', '2']:
            print("Invalid input. Please type '1' or '2'.")
            choice = input(f"Type '1' for {name_a} or '2' for {name_b}: ").strip()

        # Ask how many times more credit
        multiplier = float(input(f"How many times more credit does {'the second' if choice == '2' else 'the first'} deserve? Give a number (e.g., 3): ").strip())

        # Calculate log multiplier (negative if first name is chosen)
        log_multiplier = math.log(multiplier) if choice == '2' else -math.log(multiplier)

        # Store result as (index of first, index of second, log multiplier)
        samples.append((a, b, log_multiplier))
    return samples

# Main program
if __name__ == "__main__":
    print("Gathering user comparisons...")
    user_samples = gather_user_comparisons()

    optimal_weights = find_optimal_weights(logits, user_samples)
    final_logits = [
        sum([w * L[i] for w, L in zip(optimal_weights, logits)])
        for i in range(len(logits[0]))
    ]
    exp_logits = [math.exp(v) for v in final_logits]
    sum_exp_logits = sum(exp_logits)
    final_credit = [v / sum_exp_logits for v in exp_logits]
    for i, k in enumerate(distributions.keys()):
        print(
            "Cost of pure {} distribution: {:.4f}"
            .format(k, cost_function(logits[i], user_samples))
        )
    print(
        "Cost of lowest-cost distribution: {:.4f}"
        .format(cost_function(final_logits, user_samples))
    )
    print("\nOptimal weights for lowest-cost distribution:\n")
    for key, weight in zip(distributions.keys(), optimal_weights):
        padding = ' ' * (max(len(x) for x in distributions.keys()) - len(key))
        print(f"{key}:{padding} {weight:.3f}")
    print("\nLowest-cost distribution:\n")
    for name, credit in zip(characters, final_credit):
        padding = ' ' * (max(len(x) for x in characters) - len(name))
        print(f"{name}:{padding} {credit:.3f}")
