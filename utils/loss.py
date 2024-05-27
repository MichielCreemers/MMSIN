import torch
from torch.nn import functional

class L2RankLoss(torch.nn.Module):
    """
    Custom built loss function which combines both the L2 loss (MSE)
    and the Rank loss (maintain relative order)
    """

    def __init__(self, **kwargs):
        super(L2RankLoss, self).__init__()
        #weights: how important is each loss
        self.l2_weight = 1  
        self.rank_weight = 1
        #threshold to define wether an entry is hard to rank or not
        self.rank_thres = 1
        #use absolute difference
        self.use_margin =False

    def forward(self, prediction, ground_truths):
        #reshape predictions and ground_truths into a 1 dimensional tensor
        prediction = prediction.view(-1)
        ground_truths = ground_truths.view(-1)

        #l2 loss
        l2_loss = functional.mse_loss(prediction, ground_truths) * self.l2_weight

        #ranking loss
        n = len(prediction)

        prediction = prediction.unsqueeze(0).repeat(n,1)
        prediction_t = prediction.t()

        ground_truths_unsq = ground_truths.unsqueeze(0).repeat(n,1)
        ground_truths_t = ground_truths_unsq.t()

        masks = torch.sign(ground_truths_unsq - ground_truths_t)
        masks_hard = (torch.abs(ground_truths_unsq - ground_truths_t) < self.rank_thres) & (torch.abs(ground_truths_unsq-ground_truths_t) > 0)

        if(self.use_margin):
            rank_loss = masks_hard * torch.relu(torch.abs(ground_truths_unsq - ground_truths_t) - masks * (prediction - prediction_t))

        else:
            rank_loss = masks_hard * torch.relu(- masks * (prediction - prediction_t))

        rank_loss = (rank_loss.sum() / (rank_loss.sum() + 1e-08)) * self.rank_weight

        total_loss = l2_loss + rank_loss

        return total_loss

