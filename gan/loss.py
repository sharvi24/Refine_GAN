import torch

from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


def discriminator_loss(logits_real, logits_fake):
    """ Computes the discriminator loss

    Args:
        logits_real: PyTorch Tensor giving scores for the real data
        logits_fake: PyTorch Tensor giving scores for the fake data

    Returns:
        loss: PyTorch Tensor containing the (scalar) loss for the discriminator.
    """
    real_loss = bce_loss(
        logits_real, torch.ones(logits_real.shape), reduction='mean')

    fake_loss = bce_loss(
        1 - logits_fake, torch.ones(logits_fake.shape), reduction='mean')
    loss = (real_loss + fake_loss) / 2
    return loss


def generator_loss(logits):
    """ Computes the generator loss.

    Args:
        logits: PyTorch Tensor giving scores for the fake data.

    Returns:
        loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    return bce_loss(logits_fake, torch.ones(logits_fake.shape).cuda().detach(),
                    reduction='mean')
