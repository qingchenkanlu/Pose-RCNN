from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference, \
    keypointrcnn_loss, keypointrcnn_inference
import torch

import torch.nn.functional as F


class MyRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 # Pose
                 pose_roi_pool=None,
                 pose_head=None,
                 pose_predictor=None):
        super(MyRoIHeads, self).__init__(box_roi_pool,
                                         box_head,
                                         box_predictor,
                                         # Faster R-CNN training
                                         fg_iou_thresh, bg_iou_thresh,
                                         batch_size_per_image, positive_fraction,
                                         bbox_reg_weights,
                                         # Faster R-CNN inference
                                         score_thresh,
                                         nms_thresh,
                                         detections_per_img,
                                         # Mask
                                         mask_roi_pool=mask_roi_pool,
                                         mask_head=mask_head,
                                         mask_predictor=mask_predictor,
                                         # Point
                                         keypoint_roi_pool=keypoint_roi_pool,
                                         keypoint_head=keypoint_head,
                                         keypoint_predictor=keypoint_predictor)

        self.pose_roi_pool = pose_roi_pool
        self.pose_head = pose_head
        self.pose_predictor = pose_predictor

    @property
    def has_pose(self):
        if self.pose_roi_pool is None:
            return False
        if self.pose_head is None:
            return False
        if self.pose_predictor is None:
            return False

        return True

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        if self.has_mask:
            assert all("masks" in t for t in targets)
        if self.has_pose:
            assert all("poses" in t for t in targets)

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                assert t["poses"].dtype.is_floating_point
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)    # torch.Size([bs*1000, 256, 7, 7])
        box_features = self.box_head(box_features)    # torch.Size([bs*1000, 1024])
        class_logits, box_regression = self.box_predictor(box_features)  # torch.Size([bs*1000, 2]) torch.Size([bs*1000, 8])

        result, losses = [], {}    # result 是一个字典的列表, 每一个字典存着每张图片的预测值
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )
        # 如果是Mask R-CNN
        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_keypoint:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])   # shape=(num_pos, 4)
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            else:
                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        if self.has_pose:
            pose_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                pose_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)    # 所有被分配为正样本的proposal的下标
                    pose_proposals.append(proposals[img_id][pos])         # proposal的box(xmin, ymin, xmax, ymax)
                    pos_matched_idxs.append(matched_idxs[img_id][pos])    # 每个proposal对应哪个target pose
            pose_features = self.pose_roi_pool(features, pose_proposals, image_shapes)
            pose_features = self.pose_head(pose_features)
            pose_regression = self.pose_predictor(pose_features)

            loss_pose = {}
            if self.training:
                gt_poses = [t["poses"] for t in targets]    # a list of (rx, ry, rz, tz)
                loss_pose = posercnn_loss(pose_regression, gt_poses, labels, pos_matched_idxs)
                loss_pose = dict(loss_pose=loss_pose)
            else:
                pred_poses = postprocess_poses(pose_regression, pose_proposals)
                for poses, r in zip(pred_poses, result):
                    r['poses'] = poses
            losses.update(loss_pose)

        return result, losses


def posercnn_loss(pose_regression, gt_poses, labels, pose_matched_idxs):
    poses_per_image = [len(idxs) for idxs in pose_matched_idxs]
    pred_poses = pose_regression.split(poses_per_image, 0)

    loss = 0.0
    for pred_pose, gt_pose, label, idx in zip(pred_poses, gt_poses, labels, pose_matched_idxs):
        sampled_pos_inds_subset = torch.nonzero(label > 0).squeeze(1)
        labels_pos = label[sampled_pos_inds_subset]

        N, _ = pred_pose.shape
        pred_pose = pred_pose.reshape(N, -1, 4)
        pose_lose = F.smooth_l1_loss(pred_pose[:, 1], gt_pose[idx], reduction="sum")    # Only support 2 class train
        loss += pose_lose / label.numel()
    return loss


def postprocess_poses(pose_regression, pose_proposals):
    poses_per_image = [len(proposal) for proposal in pose_proposals]
    pred_poses = pose_regression.split(poses_per_image, 0)

    poses = []
    for pred_pose in pred_poses:
        N, _ = pred_pose.shape
        if N == 0:
            poses.append([])
            continue
        pred_pose = pred_pose.reshape(N, -1, 4)
        poses.append(pred_pose[:, 1])
    return poses
