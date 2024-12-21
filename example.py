from scoring import find_optimal_weights, cost_function
import random
import math

# Character list
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

# Note that it actually does not matter whether or not these sum to 1;
# if the probabilities sum to t != 1, then the result will be equivalent
# to the case where you instead provide [v/t for v in probabilities]
distributions = [
    [
        0.25, 0.20, 0.15, 0.10, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02,
        0.03, 0.02, 0.02, 0.03, 0.01,
        0.02, 0.01, 0.01, 0.03, 0.01
    ], # Asked GPT how much credit each deserves
    [
        0.17, 0.10, 0.10, 0.10, 0.05,
        0.05, 0.05, 0.05, 0.03, 0.03,
        0.01, 0.02, 0.02, 0.05, 0.02,
        0.02, 0.01, 0.01, 0.03, 0.01
    ], # My own opinion from thinking about it for 1 minute
    [0.05] * 20, # Egalitarian split
]

# Convert probability distributions into a list of logits
logits = [[math.log(p) for p in dist] for dist in distributions]

# Function to prompt user for pairwise comparisons
def gather_user_comparisons():
    samples = []
    for _ in range(5):
        # Randomly select two different names
        a, b = random.sample(range(len(characters)), 2)
        name_a, name_b = characters[a], characters[b]

        # Ask the user which deserves more credit
        print(f"Who deserves more credit for helping defeat Sauron? {name_a} or {name_b}?")
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

    optimal_weights = [
        int(x * 1000) / 1000
        for x in find_optimal_weights(logits, user_samples)
    ]
    final_logits = [
        sum([w * L[i] for w, L in zip(optimal_weights, logits)])
        for i in range(len(logits[0]))
    ]
    exp_logits = [math.exp(v) for v in final_logits]
    sum_exp_logits = sum(exp_logits)
    final_credit = [
        int(v / sum_exp_logits * 1000) / 1000 for v in exp_logits
    ]
    print(
        "Cost of pure gpt distribution: {:.4f}"
        .format(cost_function(logits[0], user_samples))
    )
    print(
        "Cost of pure human distribution: {:.4f}"
        .format(cost_function(logits[1], user_samples))
    )
    print(
        "Cost of pure egalitarian distribution: {:.4f}"
        .format(cost_function(logits[2], user_samples))
    )
    print(f"Optimal weights for lowest-cost distribution: {optimal_weights}")
    print(f"Lowest-cost distribution: {final_credit}")
    print(
        "Cost of ideal distribution: {:.4f}"
        .format(cost_function(final_logits, user_samples))
    )
