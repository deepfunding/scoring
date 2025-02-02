import random, math
from scoring import find_optimal_weights, cost_function
import networkx as nx
import matplotlib.pyplot as plt

INTRO_STRING = """
Welcome to the multi-layer deep funding demo! In this example, we are
going to assign weights to various philosophical contributions that are
upstream of Ethereum's philosophy. This includes _direct_ dependencies, such as
as Bitcoin's monetary philosophy and the cypherpunk movement, and also
_dependencies of each dependency_: for example, Austrian economics as
a dependency of Bitcoin's monetary philosophy.

Note that in principle, a dependency graph of this type can go very deep:
Austrian economics itself depends on classical liberalism, which itself has
some degree of dependence on Islamic economic theories that were developed
roughly a millennium ago. Here, we are simplifying to two levels.

We represent the two-level graph as a _three-level_ graph, to incorporate
another subtle but important dimension: to what extent was each of Ethereum's
philosophical progenitors an original contribution, versus "just" a repackaging
of things that came before it? In other natural deep funding use cases, such as
math academic and software, this dimension is even more important.

This simulator puts you into the role of a juror. You will be given several
"comparison" questions, which ask you to make _local_ judgements: for a given
philosophy, and two of its "ancestors", judge which of its ancestors provided
more value, and how much more. There will also be questions asking you to
judge the degree of originality of a philosophy.

Several AI models have been asked to give a score for every edge in the
dependency graph. After you, the juror, give a few inputs on randomly-sampled
questions, the linear combination of AI models that are a best-fit to your
answers will be used to generate "final" weights for the entire graph.
"""

# The thing whose (recursive) dependencies we are trying to assign credit for
ROOT_NODE = "Ethereum"

# A list of dependencies of ROOT_NODE, and for each dependency a list of
# dependencies _of that dependency_
ITEMS = {
    "Bitcoin's monetary philosophy": [
        "Austrian Economics / sound money advocacy (Mises, Hayek, Rothbard)",
        "Free currency competition / denationalization of money (eg. Hayek)",
        "Libertarian minarchism and anarchism (Rothbard, Nozick, Rand)",
        "Monetarism (eg. Milton Friedman)",
        "Keynesian monetary theory",
        "Labor theory of value (eg. Marx)"
    ],
    "Cypherpunk Movement": [
        "Libertarian minarchism and anarchism (Rothbard, Nozick, Rand)",
        "Cryptographic Pioneers (Diffie, Hellman, Zimmermann, Chaum)",
        "Cyberpunk Literature (eg. Gibson’s Neuromancer, Neal Stephenson)",
        "Hacker Ethics (eg. Levy's Hackers)",
        "Institutional privacy advocacy (eg. EFF)",
        "Cryptoanarchism (eg. Tim May's manifesto, JP Barlow's declaration)",
        "Swiss direct democracy",
    ],
    "Free/Open Source Software (FOSS)": [
        "Hacker Ethics (eg. Levy's Hackers)",
        "Free Software Foundation (Stallman’s GNU Manifesto)",
        "Open Source Initiative (Raymond’s The Cathedral and the Bazaar)",
        "Unix Philosophy (modularity, simplicity)",
        "Commons-Based Peer Production (Benkler)",
        "Wikipedia and other early-2000s online collaborations",
    ],
    "Decentralized Governance / DAOs": [
        "Left-Anarchist Political Theory (eg. Proudhon, Bakunin)",
        "Elinor Ostrom's Commons Governance",
        "Cybernetics (eg. Stafford Beer’s Viable System Model)",
        "Daemon (Daniel Suarez)",
        "Wikipedia and other early-2000s online collaborations",
        "Participatory Democracy (Bookchin’s Communalism)",
        "Swiss direct democracy",
        "Medieval feudal governance models"
    ],
    "Mechanism Design & Game Theory": [
        "Nash Equilibrium (John Nash)",
        "Early Mechanism Design Theory (Hurwicz, Maskin, Myerson)",
        "Cooperative Game Theory (eg. Shapley values)",
        "Schelling Points (Thomas Schelling, 1960)",
        "Transaction cost theory (eg. Coase)"
    ],
    "Techno-Progressivism": [
        "Transhumanism (Bostrom, Kurzweil)",
        "Cyberpunk Literature (eg. Gibson’s Neuromancer, Neal Stephenson)",
        "Decentralized Web Ideals (eg. Berners-Lee)",
        "Accelerationism (eg. Nick Land)",
        "Post-scarcity economy ideas"
    ]
}

# Repackaging the above into a "standardized" graph format. This graph format
# is three-layered, where the middle layer represents the question of to what
# extent each top-level dependency is itself a contribution in its own right,
# versus a repackaging of its own dependencies.
GRAPH = {
    item: {
        "self": None,
        "dependencies": {
            value: None
            for value in values
        }
    }
    for item, values in ITEMS.items()
}

# Sample code to generate some queries that you can ask AI models to get
# weights for the whole graph. Note that there is probably an entire art
# to prompting (and perhaps extending with RAG or interner search or
# even fine-tuning) an AI well to give good answers to these questions.
def generate_ai_queries(parent, graph):
    if graph is None:
        return []
    elif set(list(graph.keys())) == {'self', 'dependencies'}:
        children = list(graph['dependencies'].keys())
        o = [
            f"Estimate to what extent {parent} can be described as a mere combination of pre-existing ideas such as {children}, and to what extent it's a novel thing that is more than the sum of its parts. Provide your answer as a python list [pct_original, pct_derivative], where the two numbers add up to 100, with no surrounding explanation text"
        ]
        o.extend(generate_ai_queries(parent, graph['dependencies']))
        return o
    else:
        children = list(graph.keys())
        o = [
            f"Estimate the relative influence that each of the following philosophies {children} had on {parent}. Provide your answer ONLY as a python list of numbers (eg. [10, 20, 30, 40]) that add up to 100, with no surrounding explanation text"
        ]
        for parent, child_node in graph.items():
            o.extend(generate_ai_queries(parent, child_node))
        return o

# The scores given by each AI model, arranged in depth-first-search order
distributions = {
    'gpt_o3': [
        # Bitcoin's monetary philosophy (20% overall)
        20,                # Top‐level credit share
        40, 60,           # 40% original, 60% derivative
        30, 20, 20, 15, 10, 5,  # Children
        # Cypherpunk Movement (20% overall)
        20,                # Top‐level credit share
        50, 50,           # 50% original, 50% derivative
        15, 30, 10, 15, 10, 10, 10, # Children
        # Free/Open Source Software (FOSS) (15% overall)
        15,                # Top‐level credit share
        30, 70,           # 30% original, 70% derivative
        20, 25, 20, 15, 10, 10, #Children
        # Decentralized Governance / DAOs (20% overall)
        20,                # Top‐level credit share
        60, 40,           # 60% original, 40% derivative
        15, 20, 15, 5, 10, 15, 10, 10, #Children
        # Mechanism Design & Game Theory (15% overall)
        15,                # Top‐level credit share
        70, 30,           # 70% original, 30% derivative
        30, 30, 10, 20, 10, # Children
        # Techno-Progressivism (10% overall)
        10,                # Top‐level credit share
        40, 60,           # 40% original, 60% derivative
        25, 20, 25, 15, 15 # Children
    ],
    'gpt_o1': [
        15,
        40, 60,
        40, 25, 20, 10, 3, 2,
        20,
        40, 60,
        15, 20, 15, 10, 15, 20, 5,
        20,
        60, 40,
        15, 25, 20, 20, 10, 10,
        15,
        40, 60,
        15, 20, 10, 5, 15, 10, 20, 5,
        15,
        70, 30,
        20, 30, 20, 20, 10,
        15,
        40, 60,
        30, 20, 15, 15, 20
    ],
    'deepseek': [
        25,
        30, 70,
        35, 25, 20, 15, 3, 2,
        20,
        40, 60,
        20, 25, 10, 10, 15, 15, 5,
        20,
        30, 70,
        20, 25, 20, 15, 10, 10,
        10,
        35, 65,
        20, 25, 10, 5, 20, 10, 5, 5,
        20,
        30, 70,
        35, 30, 20, 10, 5,
        5,
        35, 65,
        25, 15, 25, 10, 25
    ],
    'claude': [
        15,
        30, 70,
        35, 20, 25, 10, 5, 5,
        25,
        35, 65,
        20, 25, 10, 15, 15, 10, 5,
        20,
        25, 75,
        20, 25, 20, 15, 10, 10,
        10,
        45, 55,
        10, 15, 10, 5, 15, 20, 15, 10,
        20,
        20, 80,
        25, 35, 15, 15, 10,
        10,
        30, 70,
        30, 15, 25, 15, 15
    ]
}

# The actual scoring function operates over logits, so we convert all
# scores into logits.
logits = [[math.log(p) for p in dist] for dist in distributions.values()]

# Function to ask the user to compare two children of a parent
def ask_comparison(parent, name_a, name_b):
    # Ask the user which deserves more credit
    print(" ")
    print(f"Which philosophy had more influence over {parent}? {name_a} or {name_b}?")
    choice = input(f"Type '1' for {name_a} or '2' for {name_b}: ").strip()

    # Ensure valid input
    while choice not in ['1', '2']:
        print("Invalid input. Please type '1' or '2'.")
        choice = input(f"Type '1' for {name_a} or '2' for {name_b}: ").strip()

    # Ask how many times more credit
    multiplier = float(input(f"How many times more influence did {'the second' if choice == '2' else 'the first'} have? Give a number (e.g., 3): ").strip())

    # Calculate log multiplier (negative if first name is chosen)
    return math.log(multiplier) if choice == '2' else -math.log(multiplier)

# Function to ask the user to give the level of originality of a node
def ask_about_originality(parent, children):
    children_txt = '* ' + '\n* '.join(children)
    print(" ")
    print(f"To what extent is {parent} a significant philosophical development in its own right, as opposed to being a mere recombination of pre-existing ingredients such as the following? \n\n{children_txt}\n")
    value = float(input("Eg. answer 0.75 for developments that are very significant in their own right, 0.25 for mostly-recombinations, and 0.5 if somewhere in between: "))
    return math.log(value / (1 - value))

# Get the total number of nodes in a given graph or subtree
def get_total_node_count(node):
    if node is None:
        return 1
    else:
        return 1 + sum([get_total_node_count(v) for v in node.values()])

# In a vector, given a particular position of the current node, get
# the positions of its children, based on depth-first order
def get_children_positions(my_position, node):
    o = [my_position + 1]
    for child in node.values():
        o.append(o[-1] + get_total_node_count(child))
    return o[:-1]

# Ask a user enough questions to get a list of sample values, which
# models can then be tested against.
def gather_user_comparisons(parent, graph):
    top_level_nodes = list(graph.keys())
    top_level_width = len(top_level_nodes)
    samples = []
    start_positions = get_children_positions(-1, graph)
    # Top-level comparisons
    for _ in range(2):
        # Randomly select two different top-level dependencies
        a, b = random.sample(range(top_level_width), 2)
        name_a, name_b = top_level_nodes[a], top_level_nodes[b]
        # Store result as (index of first, index of second, log multiplier)
        samples.append((
            start_positions[a],
            start_positions[b],
            ask_comparison(parent, name_a, name_b)
        ))
    # Originality rating
    for _ in range(2):
        # Randomly select a single top-level dependency
        a = random.randrange(top_level_width)
        a_self_index = start_positions[a] + 1
        a_dependencies_index = start_positions[a] + 2
        name_a = top_level_nodes[a]
        # Store result as (index of first, index of second, log multiplier)
        samples.append((
            a_self_index,
            a_dependencies_index,
            ask_about_originality(
                name_a,
                list(graph[name_a]['dependencies'].keys())
            )
        ))
    # Comparing two children of a given top-level dependency
    for _ in range(4):
        # Randomly select a child node
        a = random.randrange(top_level_width)
        name_a = top_level_nodes[a]
        # Randomly select two children of that child
        lower_level_nodes = list(graph[name_a]["dependencies"].keys())
        lower_level_width = len(lower_level_nodes)
        b, c = random.sample(range(lower_level_width), 2)
        b_index, c_index = start_positions[a] + 3 + b, start_positions[a] + 3 + c
        name_b, name_c = lower_level_nodes[b], lower_level_nodes[c]
        samples.append((b_index, c_index, ask_comparison(name_a, name_b, name_c)))
    return samples

def plot_tree(tree, weights, root):
    """
    Plots a tree (given as a nested dictionary) in a radial layout.
    The root is placed at the center (level 0) and nodes at level i are
    placed on a circle of radius i*k (pixels). Leaf nodes are assigned angles
    in DFS order (evenly spaced between 0° and 360°), and each internal node’s
    angle is computed as the median of the angles of its descendant leaves.
    
    Edge labels (weights) are assigned in DFS order (i.e. in the order the DFS
    traversal visits the edges) and are printed with up to 3 decimal places.
    
    Duplicate node names are handled by assigning each node a unique internal ID.
    
    Parameters:
      tree (dict): A nested dictionary representing the tree.
                   For example:
                     {"node_a": {"node_b": None, "node_c": None},
                      "node_d": {"node_e": None}}
      weights (list of float): Edge labels in DFS order.
                               For the above tree (with an extra root),
                               the order is:
                               [edge from root->first child, then recursively
                                the edges in that subtree, then remaining edges]
      root (str): The label for the top-level (root) node.
    """
    # Create a directed graph.
    G = nx.DiGraph()

    # Dictionaries to store each node's display label, its level (distance from root),
    # and (later) its assigned angle (in degrees).
    node_labels = {}
    level = {}
    node_angles = {}
    # For each node, store the list of its children (by unique id) in the order encountered.
    children_mapping = {}

    # We'll assign a unique ID to each node by appending an incrementing counter.
    node_counter = 0
    weight_index = 0

    # Create the root node.
    root_uid = f"{root}_{node_counter}"
    node_counter += 1
    G.add_node(root_uid)
    node_labels[root_uid] = root
    level[root_uid] = 0

    # --- Build the graph (and assign weights) using DFS ---
    def build_graph_dfs(parent_uid, subtree):
        nonlocal node_counter, weight_index
        # If there is no subtree, nothing to do.
        if not subtree:
            return
        for child_name, child_subtree in subtree.items():
            # Create a unique id for the child node.
            child_uid = f"{child_name}_{node_counter}"
            node_counter += 1
            G.add_node(child_uid)
            node_labels[child_uid] = child_name
            # Set the child's level.
            level[child_uid] = level[parent_uid] + 1
            # Record the child in the parent's children list.
            children_mapping.setdefault(parent_uid, []).append(child_uid)
            # Add the edge from parent to child.
            G.add_edge(parent_uid, child_uid)
            # Assign the next weight (if available) to the edge in DFS order.
            if weight_index < len(weights):
                G[parent_uid][child_uid]['weight'] = weights[weight_index]
                weight_index += 1
            else:
                G[parent_uid][child_uid]['weight'] = None
            # Recurse into the child's subtree, if any.
            if child_subtree:
                build_graph_dfs(child_uid, child_subtree)

    # Build the graph starting from the root.
    build_graph_dfs(root_uid, tree)

    # --- Compute the radial positions ---
    # (a) Collect all leaf nodes in DFS order.
    def dfs_collect_leaves(node, leaves):
        if node not in children_mapping:
            leaves.append(node)
        else:
            for child in children_mapping[node]:
                dfs_collect_leaves(child, leaves)

    dfs_leaves = []
    dfs_collect_leaves(root_uid, dfs_leaves)
    n_leaves = len(dfs_leaves)
    
    # (b) Assign each leaf an angle evenly spaced between 0° and 360°.
    for i, leaf in enumerate(dfs_leaves):
        angle = (i * 360 / n_leaves) % 360
        node_angles[leaf] = angle

    # (c) For internal nodes, assign their angle as the median of the angles
    # of all descendant leaves. (Since DFS order was used to collect leaves,
    # we do not re-sort the angles.)
    def assign_internal_angles(node):
        if node not in children_mapping:
            return [node_angles[node]]
        else:
            descendant_angles = []
            for child in children_mapping[node]:
                descendant_angles.extend(assign_internal_angles(child))
            n = len(descendant_angles)
            if n % 2 == 1:
                median_angle = descendant_angles[n // 2]
            else:
                median_angle = (descendant_angles[n // 2 - 1] + descendant_angles[n // 2]) / 2
            node_angles[node] = median_angle
            return descendant_angles

    assign_internal_angles(root_uid)

    # (d) Compute (x, y) coordinates for each node:
    #     Nodes at level i are placed at radius r = i * k.
    k = 100  # radial spacing in pixels per level.
    pos = {}
    for node in G.nodes():
        r = level[node] * k
        theta_rad = math.radians(node_angles[node])
        pos[node] = (r * math.cos(theta_rad), r * math.sin(theta_rad))

    # --- Prepare edge labels with weights formatted to 3 decimal places ---
    edge_labels = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight is not None:
            edge_labels[(u, v)] = f"{weight:.3f}"
        else:
            edge_labels[(u, v)] = ""

    # --- Draw the graph ---
    nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=1500,
            node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Take a vector of weights, and adjust it so that
# the weights of all children of a given parent sum to 1

def normalize_vector(node, vector, pos=-1):
    child_positions = get_children_positions(pos, node)
    children_sum = sum([vector[p] for p in child_positions])
    for p in child_positions:
        vector[p] /= children_sum
    for child_node, pos in zip(list(node.values()), child_positions):
        if child_node:
            normalize_vector(child_node, vector, pos)

if __name__ == '__main__':
    print(INTRO_STRING)

    user_samples = gather_user_comparisons(ROOT_NODE, GRAPH)

    print("\nObtained samples: ")
    for a, b, diff in user_samples:
        print(f"({a}, {b}, {diff:.3f})")
    print("")

    optimal_weights = find_optimal_weights(logits, user_samples)
    final_logits = [
        sum([w * L[i] for w, L in zip(optimal_weights, logits)])
        for i in range(len(logits[0]))
    ]
    final_edges = [math.exp(v) for v in final_logits]
    normalize_vector(GRAPH, final_edges)
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
    print("\nGraphing lowest-cost distribution...")
    plot_tree(GRAPH, final_edges, ROOT_NODE)
