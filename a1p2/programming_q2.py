import numpy as np


def debug_matching_lists(actual_best, expected_best, actual_second, expected_second):
    def compare_lists(actual, expected, name):
        if len(actual) != len(expected):
            print(f"❌ {name}: Length mismatch! Expected {len(expected)}, got {len(actual)}.")
            return False
        
        for i, (act, exp) in enumerate(zip(actual, expected)):
            # Assuming format from your previous code: ((l_y, l_x), (r_y, r_x), score)
            if act[0] != exp[0] or act[1] != exp[1]:
                print(f"❌ {name}: Coordinate mismatch at index {i}.\n   Expected coords: {exp[:2]}\n   Got coords:      {act[:2]}")
                return False
            if not np.isclose(act[2], exp[2], atol=1e-5):
                print(f"❌ {name}: Score mismatch at index {i}.\n   Expected score: {exp[2]}\n   Got score:      {act[2]}")
                return False
                
        print(f"✅ {name}: All elements match perfectly.")
        return True

    print("--- Debugging Feature Matches ---")
    compare_lists(actual_best, expected_best, "Best Matches")
    compare_lists(actual_second, expected_second, "Second Best Matches")


def perform_feature_matching(NCCs_left, NCCs_right):
    best_matches = []
    second_best_matches = []

    for i, (l_y, l_x, l_ncc) in enumerate(NCCs_left):
        best_score = np.array([None, None, -np.inf])
        second_best_score = np.array([None, None, -np.inf])
        for r_y, r_x, r_ncc in NCCs_right:
            score = (l_ncc * r_ncc).sum()
            if score > best_score[2]:
                second_best_score = best_score
                best_score = np.array([(l_y, l_x), (r_y, r_x), score], dtype=object)
            elif score > second_best_score[2]:
                second_best_score = np.array([(l_y, l_x), (r_y, r_x), score], dtype=object)
        if best_score[0] is not None:
            best_matches.append(best_score)
        if second_best_score[0] is not None:
            second_best_matches.append(second_best_score)

    return best_matches, second_best_matches


def run_with_dir(dir: str):
    print(f"Running feature matching with data from: {dir.format('<inputs/outputs>')}")
    NCCs_left_component = np.load(f'{dir.format("inputs")}/ncc_left_component.npy', allow_pickle=True)
    NCCs_right_component = np.load(f'{dir.format("inputs")}/ncc_right_component.npy', allow_pickle=True)
    best_matching_corner_list, second_best_matching_corner_list = perform_feature_matching(NCCs_left_component, NCCs_right_component)

    expected_best_matching_corner_list = np.load(f'{dir.format("outputs")}/best_matching_corner_list.npy', allow_pickle=True)
    expected_second_best_matching_corner_list = np.load(f'{dir.format("outputs")}/second_best_matching_corner_list.npy', allow_pickle=True)

    print(f"len(best_matching_corner_list): {len(best_matching_corner_list)}")
    print(f"len(second_best_matching_corner_list): {len(second_best_matching_corner_list)}")
    print(f"len(NCCs_left_component): {len(NCCs_left_component)}")
    print(f"len(NCCs_right_component): {len(NCCs_right_component)}")
    print(f"len(expected_best_matching_corner_list): {len(expected_best_matching_corner_list)}")
    print(f"len(expected_second_best_matching_corner_list): {len(expected_second_best_matching_corner_list)}")


if __name__ == "__main__":
    run_with_dir('data/feature_matching_step2_{}')
    print()
    run_with_dir('data/feature_matching_step2_{}_APR13')

    # print(verify_student_answers(best_matching_corner_list, second_best_matching_corner_list))
    # debug_matching_lists(best_matching_corner_list, expected_best_matching_corner_list, second_best_matching_corner_list, expected_second_best_matching_corner_list)

    # print(f"second_best_matching_corner_list[:5]")
    # print(np.array(second_best_matching_corner_list[:5]))
    # print(f"expected_second_best_matching_corner_list[:5]")
    # print(expected_second_best_matching_corner_list[:5])