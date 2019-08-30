""" This program finds groups of ingredients that commonly appear together.
    Enter search mode to find ingredient suggestions.
    Implementation is based on Apriori, but identifies common subsets instead of association rules.
    (Apriori learns rules, like {A} -> {B, C}. I don't think directions are appropriate here,
    so I modified Apriori to find subsets that are more common than expected, e.g. {A, B, C}.)
"""

import pandas as pd
import numpy as np
import pickle
import os

os.chdir("D:\summer2019\ingredient_group_analysis")

# irrelevant tags to exclude
INDICES_EXCLUDE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 20, 26, 27, 29, 30, 37,
                   47, 48, 57, 58, 59, 66, 73, 76, 78, 79, 80, 82, 88, 89, 91, 93, 94,
                   96, 103, 115, 118, 121, 124, 125, 127, 132, 133, 138, 141, 142, 143,
                   144, 145, 146, 147, 148, 149, 153, 161, 162, 163, 171, 172, 174, 175,
                   176, 177, 179, 181, 182, 183, 184, 186, 187, 189, 191, 195, 196, 198,
                   199, 200, 201, 202, 204, 205, 206, 207, 212, 214, 216, 218, 219, 221,
                   222, 223, 224, 227, 228, 233, 240, 241, 250, 251, 254, 256, 258, 261,
                   262, 263, 265, 266, 268, 269, 273, 275, 277, 278, 279, 282, 285, 286,
                   287, 288, 289, 290, 291, 294, 295, 297, 301, 302, 303, 304, 305, 306,
                   308, 310, 311, 313, 314, 319, 335, 336, 337, 338, 339, 340, 341, 342,
                   343, 344, 345, 346, 347, 348, 352, 357, 363, 364, 365, 371, 373, 374,
                   375, 378, 379, 381, 382, 383, 387, 389, 394, 395, 397, 398, 399, 400,
                   401, 402, 403, 404, 405, 406, 407, 409, 415, 416, 418, 421, 429, 431,
                   432, 433, 437, 438, 442, 443, 447, 448, 453, 456, 459, 461, 462, 463,
                   466, 471, 472, 475, 477, 486, 487, 489, 491, 493, 495, 498, 502, 503,
                   511, 514, 516, 523, 538, 540, 544, 548, 550, 560, 561, 562, 563, 564,
                   565, 572, 574, 578, 580, 581, 585, 586, 588, 589, 590, 594, 596, 597,
                   598, 599, 602, 604, 605, 612, 614, 615, 616, 624, 630, 631, 634, 636,
                   638, 639, 641, 646, 647, 650, 651, 652, 654, 657, 661, 662, 663, 666,
                   669, 670, 671, 673]

def colnames_to_file(df, filename):
    """ Outputs column names and their indices to a file.
        Used for manually removing bad tags.
    """
    indexed_cols = [str(i) + '\t' + df.columns[i] for i in range(len(df.columns))]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(indexed_cols))

def get_frequent_subsets(recipes, min_sup=15, min_score=3.5, max_size=3):
    """ Computes subsets up to size max_size that have a score of at least
        min_score, only looking at items that appear in at least min_sup recipes.
    """
    # C_k denotes candidate subsets size k
    # F_k denotes frequent subsets size k
    F_1 = [{t} for t in range(len(recipes.columns)) if np.sum(recipes.iloc[:,t]) > min_sup]
    freq_subsets = []
    subs_scores = []
    print("|F_1| = %d" % (len(F_1)))
    F_k = F_1
    k = 1
    while len(F_k) > 0:
        k += 1
        C_k = get_candidate_sets(F_k, F_1)
        scores = get_log_scores(recipes, C_k)
        freq_i = [i for i in range(len(C_k)) if scores[i] >= np.log(min_score)]
        F_k = [C_k[i] for i in freq_i]
        freq_subsets += F_k
        subs_scores += [scores[i] for i in freq_i]
        print("|F_%d| = %d" % (k, len(F_k)))
        if k == max_size: break ###
    return freq_subsets, subs_scores
        
def get_candidate_sets(F, F_1, option=1):
    """ Helper function. Returns the cartesian product F x F_1.
    """
    # option 1: F_{k-1} x F_1
    C = []
    for singleton in F_1:
        for subset in F:
            if max(singleton) > max(subset):
                C.append(subset.union(singleton))
    return C

def get_log_scores(recipes, C, min_sup=15, option=1):
    """ Computes scores of candidate sets in C that appear in at least min_sup recipes.
    """
    scores = [None] * len(C)
    for i, candidate in enumerate(C):
        # score the candidate set
        one_hot_cols = recipes.iloc[:,[*candidate,]]
        item_appearances = [np.nonzero(recipes[col])[0].tolist() for col in one_hot_cols]
        joint_count = len(set(item_appearances[0]).intersection(*item_appearances[1:]))
        #item_appearances = []
        #print(item_appearances[0], item_appearances[1])
        if joint_count >= min_sup:
            item_counts = [len(index_list) for index_list in item_appearances]
            N = len(recipes.index)
            # use log of scores for numerical stability
            log_numerator = np.log(joint_count) + (len(candidate)-1) * np.log(N)
            log_denominator = np.sum([np.log(ct) for ct in item_counts])
            scores[i] = log_numerator - log_denominator
        else:
            scores[i] = -np.inf
        pass
    return scores

def load_data():
    """ Loads raw recipe data and removes undesired columns.
    """
    recipes_raw = pd.read_csv('epi_r.csv')
    #recipes_tagged = recipes_raw.iloc[:,6:]
    def is_one_hot(col):
        return set(np.unique(col)) == {0, 1}
    tags = [t for t in recipes_raw.columns if is_one_hot(recipes_raw[t])]
    recipes_tagged = recipes_raw[tags]
    #colnames_to_file(recipes_tagged, "colnames_recipes_tagged.txt")
    ingredient_cols = [i for i in range(len(recipes_tagged.columns)) if i not in INDICES_EXCLUDE]
    recipes_tagged = recipes_tagged.iloc[:,ingredient_cols]
    return recipes_tagged

def save_groups(filename):
    """ Saves computed ingredient groups in pickle file.
    """
    recipes_tagged = load_data()
    item_sets, set_scores = get_frequent_subsets(recipes_tagged)
    item_sets_named = [[recipes_tagged.columns[i] for i in s] for s in item_sets]
    with open(filename, 'wb') as f:
        pickle.dump((item_sets, item_sets_named, set_scores), f)
    return item_sets_named, set_scores

def load_groups(filename):
    """ Loads precomputed ingredient groups from pickle file.
    """
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

def ingredient_search_mode(group_data, ingredient_list):
    """ Enters ingredient search mode.
    """
    item_sets, item_lists_named, set_scores = group_data
    print('Starting search mode. (Type "exit" to exit search mode.)')
    while True:
        query = input("Search for ingredient: ")
        query = query.lower()
        if query == 'exit':
            break
        elif query not in ingredient_list:
            print("ingredient not found")
        else:
            containing_lists = [l for l in item_lists_named if (query in l)]
            pairings = [l for l in containing_lists if len(l)==2]
            combos = [l for l in containing_lists if len(l) > 2]
            print("%d pairings found:" % (len(pairings)))
            pairings = [(l[1] if l[0]==query else l[0]) for l in pairings]
            print(set(pairings))
            print('-'*60)
            print("%d combinations found:" % (len(combos)))
            for c in combos:
                print(set(c) - {query})

if __name__ == '__main__':
    recipes_onehot = load_data()
    ingredient_list = recipes_onehot.columns
    ingredient_groups = load_groups("ingredient_groups.pickle")
    ingredient_search_mode(ingredient_groups, ingredient_list)

#item_sets, set_scores = save_groups("ingredient_groups.pickle")

