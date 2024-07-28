import os.path as osp

import mmcv

#from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS
import torch
'''
COCO-Pose
1-nose 2-left_eye 3-right_eye 4-left_ear 5-right_ear 6-left_shoulder 7-right_shoulder 8-‘left_elbow’ -‘right_elbow’ 10-‘left_wrist’ 11-‘right_wrist’ 12-‘left_hip’ 13-‘right_hip’ 14-‘left_knee’ 15-‘right_knee’ 16-‘left_ankle’ 17-‘right_ankle
'''
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    #dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. For UCF101 and HMDB51, allowed choices are 'train1', 'test1',
            'train2', 'test2', 'train3', 'test3'. For NTURGB+D, allowed choices are 'xsub_train', 'xsub_val',
            'xview_train', 'xview_val'. For NTURGB+D 120, allowed choices are 'xsub_train', 'xsub_val', 'xset_train',
            'xset_val'. For FineGYM, allowed choices are 'train', 'val'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose. For a video with n frames, it is a
            valid training sample only if n * valid_ratio frames have human pose. None means not applicable (only
            applicable to Kinetics Pose). Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. None means not applicable (only applicable to Kinetics). Allowed choices are 0.5, 0.6, 0.7, 0.8, 0.9.
            Default: 0.5.
        class_prob (list | None): The class-specific multiplier, which should be a list of length 'num_classes', each
            element >= 1. The goal is to resample some rare classes to improve the overall performance. None means no
            resampling performed. Default: None.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 split=None,
                 valid_ratio=None,
                 box_thr=0.5,
                 class_prob=None,
                 memcached=False,
                 mc_cfg=('localhost', 22077),
                 debug=None,
                 **kwargs):
        modality = 'Pose'
        self.split = split
        if split == 'train':
            pipeline = train_pipeline
        elif split == 'test' or split=='val':
            pipeline = test_pipeline
        else:
            raise ValueError()
        super().__init__(
            ann_file, pipeline, start_index=0, modality=modality, memcached=memcached, mc_cfg=mc_cfg, **kwargs)
        #self.avg_motion = torch.zeros(3,100,17,2)# C T V M
        #self.count = 0
        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        self.num_per_cls_dict = [978 , 975 , 316 , 253 , 313 , 960 , 944 , 420 , 372 , 325 , 449 , 722 , 639 , 412 , 890 , 439 , 777 , 279 , 887 , 944 , 470 , 254 , 892 , 555 , 970 , 251 , 409 , 989 , 452 , 756 , 525 , 912 , 555 , 298 , 787 , 375 , 759 , 982 , 272 , 274 , 873 , 693 , 987 , 936 , 405 , 453 , 542 , 493 , 594 , 885 , 686 , 566 , 550 , 305 , 398 , 983 , 754 , 333 , 357 , 742 , 693 , 425 , 289 , 529 , 415 , 526 , 262 , 502 , 931 , 975 , 797 , 446 , 248 , 309 , 501 , 863 , 283 , 986 , 759 , 833 , 680 , 406 , 545 , 592 , 981 , 562 , 673 , 802 , 651 , 433 , 250 , 518 , 413 , 894 , 438 , 303 , 307 , 750 , 288 , 760 , 433 , 404 , 251 , 881 , 741 , 557 , 255 , 933 , 879 , 694 , 338 , 361 , 577 , 370 , 408 , 760 , 985 , 396 , 348 , 263 , 287 , 436 , 279 , 974 , 787 , 862 , 889 , 674 , 509 , 552 , 880 , 516 , 690 , 701 , 796 , 427 , 287 , 275 , 495 , 583 , 795 , 544 , 681 , 919 , 261 , 456 , 482 , 982 , 988 , 934 , 475 , 796 , 658 , 910 , 315 , 270 , 564 , 393 , 355 , 969 , 467 , 676 , 686 , 402 , 961 , 382 , 746 , 972 , 266 , 763 , 507 , 325 , 960 , 506 , 672 , 388 , 468 , 635 , 523 , 505 , 728 , 271 , 678 , 600 , 292 , 279 , 520 , 501 , 972 , 594 , 278 , 261 , 985 , 891 , 304 , 359 , 481 , 809 , 441 , 939 , 266 , 973 , 265 , 570 , 696 , 626 , 585 , 342 , 711 , 878 , 430 , 294 , 602 , 592 , 626 , 403 , 272 , 766 , 763 , 675 , 984 , 957 , 566 , 918 , 662 , 859 , 363 , 715 , 482 , 633 , 741 , 317 , 942 , 841 , 990 , 746 , 556 , 312 , 564 , 511 , 974 , 523 , 921 , 976 , 751 , 815 , 963 , 973 , 823 , 984 , 974 , 627 , 589 , 819 , 874 , 949 , 847 , 382 , 985 , 332 , 456 , 902 , 987 , 308 , 968 , 264 , 255 , 320 , 555 , 940 , 539 , 335 , 321 , 973 , 512 , 702 , 447 , 727 , 979 , 268 , 800 , 265 , 693 , 985 , 420 , 636 , 749 , 325 , 478 , 726 , 271 , 591 , 781 , 350 , 830 , 446 , 437 , 267 , 825 , 719 , 246 , 651 , 823 , 284 , 976 , 650 , 955 , 881 , 977 , 321 , 390 , 334 , 346 , 634 , 298 , 623 , 886 , 682 , 786 , 338 , 245 , 816 , 770 , 980 , 450 , 821 , 962 , 738 , 315 , 253 , 981 , 606 , 293 , 553 , 657 , 315 , 721 , 593 , 437 , 912 , 676 , 519 , 357 , 253 , 318 , 317 , 907 , 222 , 950 , 784 , 656 , 537 , 436 , 434 , 342 , 543 , 662 , 476 , 938 , 435 , 990 , 307 , 306 , 326 , 633 , 804 , 487 , 629 , 233 , 672 , 509 , 682 , 245 , 749 , 388 , 235 , 410 , 277 , 966 , 878 , 696 , 251 , 745 , 612 , 262 , 516 , 384 , 604 , 555 , 777 , 585 , 591 , 260 , 948 , 671 , 335 , 558 , 243 , 939 , 934]
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            assert isinstance(self.valid_ratio, float)
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)
            if self.memcached:
                item['key'] = item['frame_dir']
        #logger = get_root_logger()
        #logger.info(f'{len(self)} videos remain after valid thresholding')


    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.data_list_path != None:
            with open(self.data_list_path, 'r') as f:
                ske_list = [
                    line.strip() for line in f.readlines()
                ]

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            #print(split.keys())
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

            if self.data_list_path != None:
                data = [y for y in data if y[identifier] in ske_list]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        
        #print('sample:', data[0])
        
        return data


