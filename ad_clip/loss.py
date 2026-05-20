import torch

def compute_negative_count_loss(image_feature, positive_text_feature, negative_text_features):
    #This function compute the count loss between one image feature and correspoing one positive text feature and multiple(5) negative text features
    #This loss is based on the formula of paper "teaching clip to count ten" from CVPR 2023.
    num_negative = negative_text_features.shape[0]
    
    if num_negative == 5:
      loss = -torch.log(
          torch.exp(image_feature@positive_text_feature.T) /
          (torch.exp(image_feature@positive_text_feature.T) + torch.exp(image_feature@negative_text_features[0].T) + torch.exp(image_feature@negative_text_features[1].T)
            + torch.exp(image_feature@negative_text_features[2].T) + torch.exp(image_feature@negative_text_features[3].T) + torch.exp(image_feature@negative_text_features[4].T))
      )
    elif num_negative == 1:
       loss = -torch.log(
          torch.exp(image_feature@positive_text_feature.T) /
          (torch.exp(image_feature@positive_text_feature.T) + torch.exp(image_feature@negative_text_features[0].T))
      )
    return loss