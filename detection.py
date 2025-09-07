### Change this to camera module to get image
# from coppeliasim_ctrl import CoppeliaCmd

### Change this to SAM
from SAM import SAM

from needleDetect.mask_sth import *
from needleDetect.visualize import *
from needleDetect.process_2d_sth import *
from needleDetect.process_pcd_sth import *

########################################### Physical ####################################################3

from needleDetect.Basler import Basler
class NeedleDetection:
    def __init__(self):
        self.camera_L = Basler(serial_number="40262045")
        self.camera_R = Basler(serial_number="40268300")
        self.camera_L.start()
        self.camera_R.start()

        self.sam_L = SAM(which='L')  ###
        self.sam_R = SAM(which='R')  ###

        self.image_L = None
        self.image_R = None

        self.map1_l = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/map1_l.npy")
        self.map2_l = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/map2_l.npy")
        self.map1_r = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/map1_r.npy")
        self.map2_r = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/map2_r.npy")

        self.image_L = self.get_image(which='L')
        self.sam_L.point_selector(self.image_L)
        self.image_R = self.get_image(which='R')
        self.sam_R.point_selector(self.image_R)

        self.P_L = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/P_L.npy")
        self.P_R = np.load("/home/surglab/icra26_nh_system/needleDetect/camera_calibration/P_R.npy")

        self.rotation_type_L = None
        self.rotation_type_R = None

    def get_image(self, which='L'):
        if which == 'L':
            self.image_L = self.camera_L.image
            self.image_L = self.rectify(self.image_L, which=which)
            return self.image_L
        elif which == 'R':
            self.image_R = self.camera_R.image
            self.image_R = self.rectify(self.image_R, which=which)
            return self.image_R

    def segment_image(self, which='L'):
        image = self.get_image(which=which)
        if which == 'L':
            return self.sam_L.segment(image)
        elif which == 'R':
            return self.sam_R.segment(image)

    def get_keypoints(self):
        # 1
        mask_L = self.segment_image(which='L')
        sub_masks_L = generate_segment_masks(self.image_L, mask_L)
        mask_head_L, mask_mid_L, mask_tip_L = sub_masks_L["head"], sub_masks_L["mid"], sub_masks_L["tip"]

        mask_R = self.segment_image(which='R')
        sub_masks_R = generate_segment_masks(self.image_R, mask_R)
        mask_head_R, mask_mid_R, mask_tip_R = sub_masks_R["head"], sub_masks_R["mid"], sub_masks_R["tip"]

        ## Visualize
        visualize_mask_segment(mask_head_L, mask_mid_L, mask_tip_L, which='L')
        visualize_mask_segment(mask_head_R, mask_mid_R, mask_tip_R, which='R')

        # 2
        mask_clustered_L = cluster_mask(mask_L)
        mask_clustered_R = cluster_mask(mask_R)
        if mask_clustered_L is None or mask_clustered_R is None:
            print("mask clustered is None")
            return None

        # 3
        ellipse_L = fit_ellipse(mask_clustered_L)  # (cx, cy, major, minor, angle)
        ellipse_R = fit_ellipse(mask_clustered_R)
        (cx_L, cy_L), (major_L, minor_L), angle_L = ellipse_L
        (cx_R, cy_R), (major_R, minor_R), angle_R = ellipse_R

        if ellipse_L is None or ellipse_R is None:
            print("ellipse is None")
            return None

        # 4
        samples_dual_res = sample_ellipse_ordered_dual(ellipse_L, ellipse_R, mask_L, mask_R, num_samples=100)
        if samples_dual_res is None:
            return None
        samples_L, samples_R, angles_L, angles_R = samples_dual_res

        # 5
        idx_head_L, idx_mid_L, idx_tip_L = get_intersection_idx(samples_L, mask_head_L, mask_mid_L, mask_tip_L)
        idx_head_R, idx_mid_R, idx_tip_R = get_intersection_idx(samples_R, mask_head_R, mask_mid_R, mask_tip_R)

        idx_lst = [idx_head_L, idx_mid_L, idx_tip_L, idx_head_R, idx_mid_R, idx_tip_R]
        for idx in idx_lst:
            if len(idx) == 0:
                print("idx is None")
                return None
        # 6
        samples_head_L, samples_mid_L, samples_tip_L = samples_L[idx_head_L], samples_L[idx_mid_L], samples_L[idx_tip_L]
        samples_head_R, samples_mid_R, samples_tip_R = samples_R[idx_head_R], samples_R[idx_mid_R], samples_R[idx_tip_R]

        ## Visualize
        visualize_subsamples(mask_L, samples_head_L, samples_mid_L, samples_tip_L, which="L")
        visualize_subsamples(mask_R, samples_head_R, samples_mid_R, samples_tip_R, which="R")

        # 7
        # Left rotation
        all_indices_L = np.sort(np.concatenate([idx_head_L, idx_mid_L, idx_tip_L]))
        segment_labels_L, unique_labels_L = label_segments(all_indices_L, idx_head_L, idx_mid_L, idx_tip_L)
        self.rotation_type_L = determine_rotation(unique_labels_L)

        # Right rotation
        all_indices_R = np.sort(np.concatenate([idx_head_R, idx_mid_R, idx_tip_R]))
        segment_labels_R, unique_labels_R = label_segments(all_indices_R, idx_head_R, idx_mid_R, idx_tip_R)
        self.rotation_type_R = determine_rotation(unique_labels_R)

        if self.rotation_type_L != self.rotation_type_R:
            print("Rotation type is different:", self.rotation_type_L, self.rotation_type_R)
            return None

        # 8
        ### Start | Find Intersection1(head -> mid), Intersection2(mid -> tip)
        angle_thresh = np.deg2rad(36) * 2.5  # 위에서 sampling 한 각도에 dependent
        intersection1_L, intersection2_L = find_segment_intersections(samples_L, angles_L, all_indices_L,
                                                                      segment_labels_L, self.rotation_type_L,
                                                                      angle_thresh)
        intersection1_R, intersection2_R = find_segment_intersections(samples_R, angles_R, all_indices_R,
                                                                      segment_labels_R, self.rotation_type_R,
                                                                      angle_thresh)

        ## Visualize
        visualize_intersections(mask_L, intersection1_L, intersection2_L, which="L")
        visualize_intersections(mask_R, intersection1_R, intersection2_R, which="R")

        # 9
        ### Start | Reconstruct Ellipse
        recon_res_L = reconstruct_ellipse(ellipse_L, intersection1_L, intersection2_L,
                                                                     self.rotation_type_L)
        if recon_res_L is None:
            return None
        reconstruct_pts_L, head_2d_L, tip_2d_L = recon_res_L

        recon_res_R = reconstruct_ellipse(ellipse_R, intersection1_R, intersection2_R,
                                                                     self.rotation_type_R)
        if recon_res_R is None:
            return None
        reconstruct_pts_R, head_2d_R, tip_2d_R = recon_res_R
        reconstruct_pts_L, reconstruct_pts_R = match_pts_again(ellipse_L, ellipse_R, reconstruct_pts_L,
                                                               reconstruct_pts_R)

        ## Visualize
        visualize_reconstructed_samples(mask_L, ellipse_L, reconstruct_pts_L, head_2d_L, tip_2d_L, intersection1_L,
                                        intersection2_L, which="L")
        visualize_reconstructed_samples(mask_R, ellipse_R, reconstruct_pts_R, head_2d_R, tip_2d_R, intersection1_R,
                                        intersection2_R, which="R")

        if samples_L is None or samples_R is None:
            print("No sampling points")
            return None

        if len(samples_L) < 5 or len(samples_L) < 5:
            print("Too small sampling points")
            return None
        return samples_L, samples_R, (cx_L, cy_L), (cx_R, cy_R), (head_2d_L, head_2d_R), (tip_2d_L, tip_2d_R)

    def get_needle_frame(self):
        get_kpts_res = self.get_keypoints()
        if get_kpts_res is None:
            print("key pts are None")
            return None
        key_pts_L, key_pts_R, (cx_L, cy_L), (cx_R, cy_R), (head_2d_L, head_2d_R), (tip_2d_L, tip_2d_R) = get_kpts_res


        # triangulate
        try:
            pts_3d = triangulate(key_pts_L, key_pts_R, self.P_L, self.P_R)
            start_3d = triangulate(np.array(head_2d_L), np.array(head_2d_R), self.P_L, self.P_R)[:3].squeeze()
            end_3d = triangulate(np.array(tip_2d_L), np.array(tip_2d_R), self.P_L, self.P_R)[:3].squeeze()
        except:
            print("triangulation failed")
            return None

        # ransac
        try:
            plane_normal, tangent, bitangent, plane_point, points_3d = plane_to_param(pts_3d)
            # target_radius = 0.024
            center_3d = (start_3d + end_3d) / 2

        except:
            print("ransac failed")
            return None


        print("len_points_3d:", len(points_3d))
        frames_se3 = compute_needle_frames(start_3d, end_3d, points_3d, center_3d, plane_normal, tangent,
                                           rotation_type=self.rotation_type_L)
        if frames_se3 is None or start_3d is None or end_3d is None:
            print("Computing needle frames got wrong")
            return None

        return frames_se3, (start_3d, end_3d, points_3d, center_3d)

    def rectify(self, img, which):
        if which == "L":
            img = cv2.remap(img, self.map1_l, self.map2_l, interpolation=cv2.INTER_LINEAR)  # undistorted
        elif which == "R":
            img = cv2.remap(img, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)  # undistorted
        return img

if __name__ == "__main__":
    import sys
    sys.path.append("/home/surglab/icra26_nh_system/segment-anything-2-real-time")

    # from coppeliasim_ctrl import CoppeliaCmd
    # cmd = CoppeliaCmd()
    # nd = NeedleDetection(cmd)
    nd = NeedleDetection()
    vis = True
    from needleDetect.visualize import visualize_needle_3d_live, visualize_frame_projection
    from needleDetect.ImgUtils import ImgUtils

    if vis:
        plt.ion()  # interactive mode 켜기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    while True:
        # rgb_L = nd.get_image(which='L')
        # cv2.imshow("rgb_L", rgb_L)
        # rgb_R = nd.get_image(which='R')
        # cv2.imshow("rgb_R", rgb_R)
        #
        # mask_L = nd.segment_image(which='L')
        # mask_R = nd.segment_image(which='R')
        # cv2.imshow("mask_L", mask_L)
        # cv2.imshow("mask_R", mask_R)


        needle_detection_res = nd.get_needle_frame()
        if needle_detection_res is None:
            print("needle_pose_w is None")
            continue

        # needle_pose_w, _ = needle_detection_res
        needle_pose_w, (start_3d, end_3d, points_3d, center_3d) = needle_detection_res
        print("needle_pose\n", needle_pose_w)

        if vis:
            visualize_needle_3d_live(ax, needle_pose_w, start_3d, end_3d, points_3d, center_3d)
            plt.draw()
            plt.pause(0.001)

            overlay_L, overlay_R = visualize_frame_projection(nd.image_L, nd.image_R, needle_pose_w, nd.P_L,
                                                              nd.P_R, points_3d, start_3d, center_3d, end_3d)
            cv2.imshow("Stereo Overlay", ImgUtils.stack_stereo_img(overlay_L, overlay_R, 0.5))
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()