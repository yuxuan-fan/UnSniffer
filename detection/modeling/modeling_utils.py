import torch


def covariance_output_to_cholesky(pred_bbox_cov):
    """
    这段代码实现了将输出转换为协方差 Cholesky 分解的函数。
    这个函数的目的是在某些预测任务中，从模型输出的协方差矩阵元素中构建出 Cholesky 分解，以便更容易地处理和利用模型的输出。
    Transforms output to covariance cholesky decomposition.
    Args:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:4]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)
    # import ipdb;
    # ipdb.set_trace()
    if pred_bbox_cov.shape[1] > 4:
        # print('hhh')
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        predicted_cov_cholesky[:, tril_indices[0],
                               tril_indices[1]] = pred_bbox_cov[:, 4:]

    return predicted_cov_cholesky


# 这两个函数都是用来确保在概率性预测任务中损失函数的一致性和稳定性，以帮助模型更好地学习和泛化。

def clamp_log_variance(pred_bbox_cov, clamp_min=-7.0, clamp_max=7.0):
    """
    该函数用于限制对数方差，以确保方差的范围在一个可接受的范围内。
    Tiny function that clamps variance for consistency across all methods.
    """
    pred_bbox_var_component = torch.clamp(
        pred_bbox_cov[:, 0:4], clamp_min, clamp_max)
    return torch.cat((pred_bbox_var_component, pred_bbox_cov[:, 4:]), dim=1)


def get_probabilistic_loss_weight(current_step, annealing_step):
    """
    该函数用于获取自适应的概率损失权重，以便在训练过程中动态调整损失函数中概率部分的重要性。
    Tiny function to get adaptive probabilistic loss weight for consistency across all methods.
    """
    probabilistic_loss_weight = min(1.0, current_step / annealing_step)
    probabilistic_loss_weight = (
        100 ** probabilistic_loss_weight - 1.0) / (100.0 - 1.0)

    return probabilistic_loss_weight
