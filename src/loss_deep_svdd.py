import torch

class DSVDDLoss(torch.nn.Module):
    """

    Custom loss function for Deep Support Vector Data Description (Deep SVDD).
    
    This loss function computes the distance between each data point in the representation
    space and the center of the hypersphere and aims to minimize this distance for normal data points.

    Args:
    
        c (torch.Tensor):
            The center of the hypersphere in the representation space.
            
        reduction (str, optional): 
            Specifies the reduction to apply to the output. Choices are 'none', 'mean', 'sum'. Default is 'mean'.
                - If ``'none'``: no reduction will be applied;
                - If ``'mean'``: the sum of the output will be divided by the number of elements in the output;
                - If ``'sum'``: the output will be summed

    """
    
    def __init__(self, net, sde_model, data_loader, device, reduction='mean'):
        """
        Initializes the DSVDDLoss with the hypersphere center and reduction method.
        """
        
        super(DSVDDLoss, self).__init__()
        self.net = net
        self.data_loader = data_loader
        self.reduction = reduction
        self.device = device
        self.sde_model = sde_model
        self.svdd_c = self._set_c()

    def forward(self, rep, reduction=None):
        """
        Calculates the Deep SVDD loss for a batch of representations.

        Args:
        
            rep (torch.Tensor): 
                The representation of the batch of data.
            
            reduction (str, optional): 
                The reduction method to apply. If None, will use the specified 'reduction' attribute. Default is None.

        Returns:
        
            loss (torch.Tensor): 
                The calculated loss based on the representations and the center 'c'.
                
        """
        
        loss = torch.sum((rep - self.svdd_c) ** 2, dim=1)
        # print("SVDD loss: ", loss.shape, rep.shape)

        if reduction is None:
            reduction = self.reduction
        if reduction == 'mean':
            # print("SVDD loss: ", loss.shape, rep.shape, torch.mean(loss).shape)
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss

    def _set_c(self, eps=0.1):
        """
        Initializes the center 'c' for the hypersphere in the representation space.

        Args:
        
            net (nn.Module): 
                The neural network model.
            
            data_loader (DataLoader): 
                DataLoader for the data to compute the center from.
            
            eps (float, optional):
                Small value to ensure 'c' is away from zero. Default is 0.1.

        Returns:
        
            c (torch.Tensor):  
                The initialized center of the hypersphere.
            
        """
        
        self.net.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in self.data_loader:
                x = x.float().to(self.device)
                t = torch.rand(x.shape[0], device=self.device) * (self.sde_model.T - 1e-5) + 1e-5 # 1e-5 to avoid log(0)
                z = torch.randn_like(x)
                mean, std = self.sde_model.marginal_prob(x, t)
                perturbed_x = (mean + std[:, None, None] * z).to(self.device)
                z, _, _, _ = self.net(perturbed_x, t)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        
        return c