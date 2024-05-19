#======================================================================================================
#======================================================================================================
#======================================================================================================
#EVALUATION OF THE TRAINED MODEL
#======================================================================================================
#======================================================================================================
#======================================================================================================
from model import UNet
from data_processing import SDTDataset

import numpy as np
import torch
import pickle
from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import label, maximum_filter
from sklearn.metrics import mean_squared_error


def find_local_maxima(distance_transform, min_dist_between_points):
    # Use `maximum_filter` to perform a maximum filter convolution on the distance_transform
    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, n = label(maxima)

    return seeds, n


def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    inner_mask: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=inner_mask
    )

    return segmentation


def get_inner_mask(pred, threshold):
    inner_mask = pred > threshold
    return inner_mask


def evaluate(gt_labels: np.ndarray, pred_labels: np.ndarray, th: float = 0.5):
    """Function to evaluate a segmentation."""

    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(), gt_labels_rel.flatten()])

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1
    )
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}

    for l, c in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel, return_counts=True)

    pred_labels_count_dict = {}
    for l, c in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)

    # create iou table
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1), dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[int(v), int(u)] = iou

    # remove background
    iouMat = iouMat[1:, 1:]
    best_iou = np.max(iouMat, axis=0)

    # use IoU threshold th
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    accuracy = tp / (tp + fp + fn)

    return precision, recall, accuracy, best_iou


def get_metrics(model, dataloader, device='cuda', mode='mask'):
    #iterate over evaluation images
    metrics = dict()
    if mode == 'mask':
        metrics['precision_list'], metrics['recall_list'], metrics['accuracy_list'], metrics['iou_list'] = [], [], [], []
    elif mode == 'sdt':
        metrics['mse_list'] = []

    for idx, (image, mask, sdt) in enumerate(tqdm(dataloader)):

        #retrieve image
        image = image.to(device)

        #generate prediction from neural network
        pred = model(image)

        #removes redundant dimensions I think?
        image = np.squeeze(image.cpu())
        gt_labels = np.squeeze(mask.cpu().numpy())
        gt_sdt = np.squeeze(sdt.cpu().numpy())
        pred = np.squeeze(pred.cpu().detach().numpy())

        if mode == 'mask':
            # Do watershed and compare to gt mask
            threshold = threshold_otsu(pred)

            # Get inner mask
            inner_mask = get_inner_mask(pred, threshold=threshold)

            # Get the segmentation
            seg = watershed_from_boundary_distance(pred, inner_mask, min_seed_distance=20)

            precision, recall, accuracy, iou = evaluate(gt_labels, seg)
            metrics['precision_list'].append(precision)
            metrics['recall_list'].append(recall)
            metrics['accuracy_list'].append(accuracy)
            metrics['iou_list'].append(iou)
        
        elif mode == 'sdt':
            # Compare mse of predicted sdt to computed sdt
            metrics['mse_list'].append(mean_squared_error(pred.flatten(), gt_sdt.flatten()))

    return metrics

    
def main(modelpath: str="", filepath: str="", batch_size: int=1, shuffle: bool=False, workers:int=8):
    # load model from checkpoint
    model = UNet(
        depth=4,
        in_channels=1,
        out_channels=1,
        final_activation="Sigmoid",
        num_fmaps=16,
        fmap_inc_factor=2,
        downsample_factor=2,
        padding="same",
        upsample_mode="nearest",
    )
    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #upload validation dataset and generate STD from the binary mask
    validation_data = SDTDataset(filepath) #TODO adjust

    #convert to Torch data
    validation_data = DataLoader(validation_data, batch_size=batch_size, shuffle = shuffle, num_workers = workers)

    (precision_list,recall_list,accuracy_list) = ([],[],[],)
    metrics = dict()
    metrics["precision"], metrics["recall"], metrics["accuracy"] = get_metrics(model, validation_data)
    with open('inference_results.pkl', 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    filepath = ""
    modelpath = "/localscratch/dl4mia-project-segmentation/logs/model.pth"
    batch_size = 1
    shuffle = False
    workers = 8
    main(modelpath, filepath, batch_size, shuffle, workers)
