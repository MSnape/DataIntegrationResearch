import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MoEViewFusion(nn.Module):
    """
    Mixture of Experts (MoE) model for multi-view mammogram fusion.
    Experts are pre-trained torchvision.models.resnet50 single view instances.
    A gating network learns to combine the logits from these experts.
    """
    def __init__(self, num_classes: int, cc_expert_resnet: nn.Module, mlo_expert_resnet: nn.Module, freeze_experts: bool = True):
        """
        Initializes the MoEViewFusion model.

        Args:
            num_classes (int): The number of output classes for classification.
            cc_expert_resnet (nn.Module): The fully trained ResNet50 model for CC view.
                                          This model should have its final `fc` layer configured
                                          for `num_classes` and output raw logits.
            mlo_expert_resnet (nn.Module): The fully trained ResNet50 model for MLO view.
                                           This model should have its final `fc` layer configured
                                           for `num_classes` and output raw logits.
            freeze_experts (bool): Whether to freeze the parameters of the expert ResNet models.
                                   True by default, meaning only the gating network trains. This is 
                                   for future work and true was not tested.
        """
        super(MoEViewFusion, self).__init__()
        #self.num_classes = num_classes

        # Assign the pre-trained expert ResNet50 models
        self.cc_expert = cc_expert_resnet
        self.mlo_expert = mlo_expert_resnet

        # Freeze expert parameters if specified
        if freeze_experts:
            for param in self.cc_expert.parameters():
                param.requires_grad = False
            for param in self.mlo_expert.parameters():
                param.requires_grad = False
            print("Expert models (CC and MLO ResNet50s) frozen.")
        else:
            print("Expert models (CC and MLO ResNet50s) are trainable.")

        # Gating network: Takes concatenated features from the ResNet's global average pooling layer.
        # As it's ResNet50, the output *before* the final `fc` layer (after `avgpool`) is 2048 features.
        feature_dim = self.cc_expert.fc.in_features

        # The gating network will take concatenated features from both experts
        # and output 2 values (one for CC expert, one for MLO expert).
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim * 2, 512), 
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        # Initialize weights for new layers (gating network)
        nn.init.kaiming_normal_(self.gating_network[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.gating_network[0].bias)
        nn.init.kaiming_normal_(self.gating_network[2].weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.gating_network[2].bias)


    def forward(self, x_views):
        if not isinstance(x_views, list) or len(x_views) != 2:
            raise ValueError("Expected x_views to be a list of two tensors: [cc_image, mlo_image]")

        cc_image_batch, mlo_image_batch = x_views[0], x_views[1]

        # No `with torch.no_grad():` block here as we want the gating weights to be learnt here 
        # Experts produce their features (from avgpool) and logits (from fc layer)
        # Get features from CC expert (before its final FC layer)
        x_cc = self.cc_expert.conv1(cc_image_batch)
        x_cc = self.cc_expert.bn1(x_cc)
        x_cc = self.cc_expert.relu(x_cc)
        x_cc = self.cc_expert.maxpool(x_cc)
        x_cc = self.cc_expert.layer1(x_cc)
        x_cc = self.cc_expert.layer2(x_cc)
        x_cc = self.cc_expert.layer3(x_cc)
        x_cc = self.cc_expert.layer4(x_cc)
        cc_features = self.cc_expert.avgpool(x_cc)
        cc_features = torch.flatten(cc_features, 1) 

        # Get logits from CC expert (using its own final fc layer)
        cc_logits = self.cc_expert.fc(cc_features) 

        # Get features from MLO expert (before its final FC layer)
        x_mlo = self.mlo_expert.conv1(mlo_image_batch)
        x_mlo = self.mlo_expert.bn1(x_mlo)
        x_mlo = self.mlo_expert.relu(x_mlo)
        x_mlo = self.mlo_expert.maxpool(x_mlo)
        x_mlo = self.mlo_expert.layer1(x_mlo)
        x_mlo = self.mlo_expert.layer2(x_mlo)
        x_mlo = self.mlo_expert.layer3(x_mlo)
        x_mlo = self.mlo_expert.layer4(x_mlo)
        mlo_features = self.mlo_expert.avgpool(x_mlo)
        mlo_features = torch.flatten(mlo_features, 1) 

        # Get logits from MLO expert (using its own final fc layer)
        mlo_logits = self.mlo_expert.fc(mlo_features)

        # Gating network learns to combine information based on combined features
        combined_features_for_gating = torch.cat((cc_features, mlo_features), dim=1)  

        gate_raw_scores = self.gating_network(combined_features_for_gating) 
        gate_weights = F.softmax(gate_raw_scores, dim=1) 

        # Combine expert logits using the learned gate weights
        combined_logits = (cc_logits * gate_weights[:, 0].unsqueeze(1)) + \
                          (mlo_logits * gate_weights[:, 1].unsqueeze(1))

        # --- TEMPORARY CHANGE FOR DEBUGGING ---
        # returning the raw scores and weights so can see why they are vanishing 
        # Also able to see what the gate weights are by the end
        return combined_logits, gate_raw_scores, gate_weights 