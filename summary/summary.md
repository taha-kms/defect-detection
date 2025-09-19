# Experiment Summary

- Data root: `/workspace/defect-detection/data`
- Runs root: `/workspace/defect-detection/runs`

## 1. Best Overall

Best overall: **patchcore** on **toothbrush**
 run ID: **latest**

- image_auroc: 0.9889
- pixel_auroc: 0.9545
- auprc: 0.3219
- pro: 0.2793
- latency_sec: 0.9029
- threshold: 20.174141
- tp: 28.0
- fp: 0.0
- fn: 2.0
- tn: 12.0
- run_path: /workspace/defect-detection/runs/patchcore/toothbrush/latest

## 2. Per-Class Information 

### Class: bottle

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | patchcore | bottle | 3.2433 | 0.4135 | 7.0000 | 1.0000 | 0.9468 | 0.9208 | 0.0934 | /workspace/defect-detection/runs/patchcore/bottle/latest | 24.2099 | 19.0000 | 56.0000 |
| latest | padim | bottle | 0.0128 | 0.1668 | 19.0000 | 0.0000 | 0.8683 | 0.7040 | 0.0721 | /workspace/defect-detection/runs/padim/bottle/latest | 5230.2936 | 20.0000 | 44.0000 |
| latest | ae | bottle | 0.0057 | 0.0512 | 34.0000 | 2.0000 | 0.6845 | 0.3202 | 0.0211 | /workspace/defect-detection/runs/ae/bottle/latest | 2.4299 | 18.0000 | 29.0000 |
| latest | fastflow | bottle | 0.0157 | 0.1189 | 43.0000 | 0.0000 | 0.6341 | 0.4242 | 0.0052 | /workspace/defect-detection/runs/fastflow/bottle/latest | 49457.2461 | 20.0000 | 20.0000 |

### Class: cable

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | patchcore | cable | 2.9082 | 0.1794 | 48.0000 | 9.0000 | 0.6827 | 0.8893 | 0.1718 | /workspace/defect-detection/runs/patchcore/cable/latest | 69.4060 | 49.0000 | 44.0000 |
| latest | fastflow | cable | 0.0109 | 0.0546 | 64.0000 | 7.0000 | 0.5367 | 0.7012 | 0.0020 | /workspace/defect-detection/runs/fastflow/cable/latest | 71275.1523 | 51.0000 | 28.0000 |
| latest | padim | cable | 0.0090 | 0.0446 | 77.0000 | 0.0000 | 0.5287 | 0.6436 | 0.0466 | /workspace/defect-detection/runs/padim/cable/latest | 7079.3516 | 58.0000 | 15.0000 |
| latest | ae | cable | 0.0034 | 0.0718 | 0.0000 | 58.0000 | 0.4765 | 0.7197 | 0.1799 | /workspace/defect-detection/runs/ae/cable/latest | 2.0167 | 0.0000 | 92.0000 |

### Class: screw

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | padim | screw | 0.0081 | 0.0134 | 10.0000 | 3.0000 | 0.9502 | 0.8945 | 0.1516 | /workspace/defect-detection/runs/padim/screw/latest | 3274.1613 | 38.0000 | 109.0000 |
| latest | fastflow | screw | 0.0100 | 0.0284 | 12.0000 | 17.0000 | 0.8075 | 0.9413 | 0.0535 | /workspace/defect-detection/runs/fastflow/screw/latest | 10277.7143 | 24.0000 | 107.0000 |
| latest | ae | screw | 0.0034 | 0.0041 | 108.0000 | 0.0000 | 0.2998 | 0.4144 | 0.0351 | /workspace/defect-detection/runs/ae/screw/latest | 1.4832 | 41.0000 | 11.0000 |
| latest | patchcore | screw | 4.0649 | 0.0095 | 0.0000 | 41.0000 | 0.1595 | 0.8626 | 0.1233 | /workspace/defect-detection/runs/patchcore/screw/latest | 23.3490 | 0.0000 | 119.0000 |

### Class: toothbrush

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | patchcore | toothbrush | 0.9029 | 0.3219 | 2.0000 | 0.0000 | 0.9889 | 0.9545 | 0.2793 | /workspace/defect-detection/runs/patchcore/toothbrush/latest | 20.1741 | 12.0000 | 28.0000 |
| latest | padim | toothbrush | 0.0138 | 0.1813 | 4.0000 | 2.0000 | 0.8917 | 0.9435 | 0.1306 | /workspace/defect-detection/runs/padim/toothbrush/latest | 2878.1241 | 10.0000 | 26.0000 |
| latest | fastflow | toothbrush | 0.0219 | 0.0945 | 26.0000 | 0.0000 | 0.4861 | 0.8521 | 0.0863 | /workspace/defect-detection/runs/fastflow/toothbrush/latest | 10370.5639 | 12.0000 | 4.0000 |
| latest | ae | toothbrush | 0.0091 | 0.0083 | 21.0000 | 0.0000 | 0.4556 | 0.1177 | 0.0001 | /workspace/defect-detection/runs/ae/toothbrush/latest | 2.0476 | 12.0000 | 9.0000 |

### Class: transistor

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | patchcore | transistor | 2.8425 | 0.5930 | 8.0000 | 13.0000 | 0.8479 | 0.9636 | 0.1381 | /workspace/defect-detection/runs/patchcore/transistor/latest | 34.8976 | 47.0000 | 32.0000 |
| latest | padim | transistor | 0.0109 | 0.1596 | 32.0000 | 1.0000 | 0.5846 | 0.7620 | 0.0423 | /workspace/defect-detection/runs/padim/transistor/latest | 4627.9374 | 59.0000 | 8.0000 |
| latest | fastflow | transistor | 0.0136 | 0.0822 | 36.0000 | 1.0000 | 0.4117 | 0.5598 | 0.0015 | /workspace/defect-detection/runs/fastflow/transistor/latest | 947768.3742 | 59.0000 | 4.0000 |
| latest | ae | transistor | 0.0050 | 0.0693 | 0.0000 | 60.0000 | 0.4115 | 0.6057 | 0.0773 | /workspace/defect-detection/runs/ae/transistor/latest | 1.5410 | 0.0000 | 40.0000 |

### Class: zipper

| run_id | model | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | patchcore | zipper | 2.8062 | 0.1381 | 10.0000 | 9.0000 | 0.8480 | 0.8868 | 0.1640 | /workspace/defect-detection/runs/patchcore/zipper/latest | 18.2860 | 23.0000 | 109.0000 |
| latest | ae | zipper | 0.0033 | 0.0263 | 21.0000 | 15.0000 | 0.7114 | 0.5026 | 0.0357 | /workspace/defect-detection/runs/ae/zipper/latest | 1.9124 | 17.0000 | 98.0000 |
| latest | padim | zipper | 0.0085 | 0.0403 | 31.0000 | 11.0000 | 0.6972 | 0.6907 | 0.0514 | /workspace/defect-detection/runs/padim/zipper/latest | 3859.3966 | 21.0000 | 88.0000 |
| latest | fastflow | zipper | 0.0108 | 0.0405 | 110.0000 | 0.0000 | 0.5951 | 0.5177 | 0.0012 | /workspace/defect-detection/runs/fastflow/zipper/latest | 25898.5164 | 32.0000 | 9.0000 |

## 3. Per-Model Information 

### Model: ae

| run_id | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | zipper | 0.0033 | 0.0263 | 21.0000 | 15.0000 | 0.7114 | 0.5026 | 0.0357 | /workspace/defect-detection/runs/ae/zipper/latest | 1.9124 | 17.0000 | 98.0000 |
| latest | bottle | 0.0057 | 0.0512 | 34.0000 | 2.0000 | 0.6845 | 0.3202 | 0.0211 | /workspace/defect-detection/runs/ae/bottle/latest | 2.4299 | 18.0000 | 29.0000 |
| latest | cable | 0.0034 | 0.0718 | 0.0000 | 58.0000 | 0.4765 | 0.7197 | 0.1799 | /workspace/defect-detection/runs/ae/cable/latest | 2.0167 | 0.0000 | 92.0000 |
| latest | toothbrush | 0.0091 | 0.0083 | 21.0000 | 0.0000 | 0.4556 | 0.1177 | 0.0001 | /workspace/defect-detection/runs/ae/toothbrush/latest | 2.0476 | 12.0000 | 9.0000 |
| latest | transistor | 0.0050 | 0.0693 | 0.0000 | 60.0000 | 0.4115 | 0.6057 | 0.0773 | /workspace/defect-detection/runs/ae/transistor/latest | 1.5410 | 0.0000 | 40.0000 |
| latest | screw | 0.0034 | 0.0041 | 108.0000 | 0.0000 | 0.2998 | 0.4144 | 0.0351 | /workspace/defect-detection/runs/ae/screw/latest | 1.4832 | 41.0000 | 11.0000 |

### Model: fastflow

| run_id | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | screw | 0.0100 | 0.0284 | 12.0000 | 17.0000 | 0.8075 | 0.9413 | 0.0535 | /workspace/defect-detection/runs/fastflow/screw/latest | 10277.7143 | 24.0000 | 107.0000 |
| latest | bottle | 0.0157 | 0.1189 | 43.0000 | 0.0000 | 0.6341 | 0.4242 | 0.0052 | /workspace/defect-detection/runs/fastflow/bottle/latest | 49457.2461 | 20.0000 | 20.0000 |
| latest | zipper | 0.0108 | 0.0405 | 110.0000 | 0.0000 | 0.5951 | 0.5177 | 0.0012 | /workspace/defect-detection/runs/fastflow/zipper/latest | 25898.5164 | 32.0000 | 9.0000 |
| latest | cable | 0.0109 | 0.0546 | 64.0000 | 7.0000 | 0.5367 | 0.7012 | 0.0020 | /workspace/defect-detection/runs/fastflow/cable/latest | 71275.1523 | 51.0000 | 28.0000 |
| latest | toothbrush | 0.0219 | 0.0945 | 26.0000 | 0.0000 | 0.4861 | 0.8521 | 0.0863 | /workspace/defect-detection/runs/fastflow/toothbrush/latest | 10370.5639 | 12.0000 | 4.0000 |
| latest | transistor | 0.0136 | 0.0822 | 36.0000 | 1.0000 | 0.4117 | 0.5598 | 0.0015 | /workspace/defect-detection/runs/fastflow/transistor/latest | 947768.3742 | 59.0000 | 4.0000 |

### Model: padim

| run_id | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | screw | 0.0081 | 0.0134 | 10.0000 | 3.0000 | 0.9502 | 0.8945 | 0.1516 | /workspace/defect-detection/runs/padim/screw/latest | 3274.1613 | 38.0000 | 109.0000 |
| latest | toothbrush | 0.0138 | 0.1813 | 4.0000 | 2.0000 | 0.8917 | 0.9435 | 0.1306 | /workspace/defect-detection/runs/padim/toothbrush/latest | 2878.1241 | 10.0000 | 26.0000 |
| latest | bottle | 0.0128 | 0.1668 | 19.0000 | 0.0000 | 0.8683 | 0.7040 | 0.0721 | /workspace/defect-detection/runs/padim/bottle/latest | 5230.2936 | 20.0000 | 44.0000 |
| latest | zipper | 0.0085 | 0.0403 | 31.0000 | 11.0000 | 0.6972 | 0.6907 | 0.0514 | /workspace/defect-detection/runs/padim/zipper/latest | 3859.3966 | 21.0000 | 88.0000 |
| latest | transistor | 0.0109 | 0.1596 | 32.0000 | 1.0000 | 0.5846 | 0.7620 | 0.0423 | /workspace/defect-detection/runs/padim/transistor/latest | 4627.9374 | 59.0000 | 8.0000 |
| latest | cable | 0.0090 | 0.0446 | 77.0000 | 0.0000 | 0.5287 | 0.6436 | 0.0466 | /workspace/defect-detection/runs/padim/cable/latest | 7079.3516 | 58.0000 | 15.0000 |

### Model: patchcore

| run_id | class | latency_sec | auprc | fn | fp | image_auroc | pixel_auroc | pro | run_path | threshold | tn | tp |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| latest | toothbrush | 0.9029 | 0.3219 | 2.0000 | 0.0000 | 0.9889 | 0.9545 | 0.2793 | /workspace/defect-detection/runs/patchcore/toothbrush/latest | 20.1741 | 12.0000 | 28.0000 |
| latest | bottle | 3.2433 | 0.4135 | 7.0000 | 1.0000 | 0.9468 | 0.9208 | 0.0934 | /workspace/defect-detection/runs/patchcore/bottle/latest | 24.2099 | 19.0000 | 56.0000 |
| latest | zipper | 2.8062 | 0.1381 | 10.0000 | 9.0000 | 0.8480 | 0.8868 | 0.1640 | /workspace/defect-detection/runs/patchcore/zipper/latest | 18.2860 | 23.0000 | 109.0000 |
| latest | transistor | 2.8425 | 0.5930 | 8.0000 | 13.0000 | 0.8479 | 0.9636 | 0.1381 | /workspace/defect-detection/runs/patchcore/transistor/latest | 34.8976 | 47.0000 | 32.0000 |
| latest | cable | 2.9082 | 0.1794 | 48.0000 | 9.0000 | 0.6827 | 0.8893 | 0.1718 | /workspace/defect-detection/runs/patchcore/cable/latest | 69.4060 | 49.0000 | 44.0000 |
| latest | screw | 4.0649 | 0.0095 | 0.0000 | 41.0000 | 0.1595 | 0.8626 | 0.1233 | /workspace/defect-detection/runs/patchcore/screw/latest | 23.3490 | 0.0000 | 119.0000 |

