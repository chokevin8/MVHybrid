import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    n_global_crops = len(samples_list[0][0]["global_crops"])  # 2
    n_local_crops = len(samples_list[0][0]["local_crops"])  # 8
    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)  # This is going to be 2 x batch size per gpu
    N = n_tokens  # total number of tokens, which is (img/patch size)**2
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)  # max possible num of masked patches
    for i in range(n_samples_masked, B):  # from n_samples_masked to total batch #
        masks_list.append(torch.BoolTensor(mask_generator(0)))  # just zeroes, meaning no mask
    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)  # flattens H x W to one axis
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # Entire batch is flattened

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }
