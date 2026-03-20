import numpy as np

    # w = 3
    # hw = w//2

    # cr = np.pad(corner_response, pad_width=hw, mode="edge")
    # nms_pts = []
    # for y in range(hw, image_height+hw):
    #     for x in range(hw, image_width+hw):
    #         window = cr[y-hw:y+hw+1, x-hw:x+hw+1]
    #         if cr[y,x] != 0 and cr[y,x] == window.max():
    #             nms_pts.append((y, x))
    # return nms_pts



def non_maximum_suppression(corner_response, image_width, image_height):
    w = 3
    hw = w//2

    ys, xs = np.where(corner_response > 0.1)
    cr = np.pad(corner_response, hw, mode="edge")

    nms_pts = []
    # scores = corner_response[ys, xs]
    for y, x in zip(ys, xs):
        ry = y + hw
        rx = x + hw
        window = cr[ry-hw:ry+hw+1, rx-hw:rx+hw+1]
        if cr[ry, rx] == window.max():
            nms_pts.append([int(y), int(x)])
    return np.array(nms_pts)

# def non_maximum_suppression(corner_response, image_width, image_height):
#     w = 3
#     hw = w//2

#     ys, xs = np.where(corner_response > 0)
#     cr = np.pad(corner_response, hw, mode="edge")
#     # NMS
#     selected_pts = []
#     for pt_y, pt_x in zip(ys, xs):
#         too_close = False
#         for cpt_x, cpt_y in selected_pts:
#             if np.sqrt((pt_x-cpt_x)**2 + (pt_y-cpt_y)**2) <= 3*3:
#                 too_close = True
#                 break
#         if not too_close:
#             selected_pts.append((pt_x, pt_y))
#     print(f"Points after NMS: {len(selected_pts)}")
#     return selected_pts

image_width = 210
image_height = 200

left_corner_response = np.load("a1/step4_left_corner_response.npy")
right_corner_response= np.load("a1/step4_right_corner_response.npy")
nms_left_response = non_maximum_suppression(left_corner_response, image_width, image_height)
nms_right_response = non_maximum_suppression(right_corner_response, image_width, image_height)


expected_nms_output_left = np.load("a1/step4_left_nms_corner_response.npy")
expected_nms_output_right = np.load("a1/step4_right_nms_corner_response.npy")

for nms_pt, expected_pt in zip(nms_left_response, expected_nms_output_left):
    print(f"nms_pt: {nms_pt}, expected_pt: {expected_pt}")
    if not np.array_equal(nms_pt, expected_pt):
        print(f"Mismatch: {nms_pt} != {expected_pt}")
        break
expected = (16, 91)
print(f"Window around expected point {expected}:")
print(left_corner_response[expected[0]-1:expected[0]+2, expected[1]-1:expected[1]+2])
# print(expected_nms_output_left)
# exit(0)

# print(nms_left_response)
# print(expected_nms_output_left)

print(len(nms_left_response), len(expected_nms_output_left))

print(np.array_equal(nms_left_response, expected_nms_output_left))
print(np.array_equal(nms_right_response, expected_nms_output_right))

