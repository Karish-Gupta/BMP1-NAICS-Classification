import torch
import torch.nn as nn
import torch.nn.functional as F

class NAICSSequentialEnsemble(nn.Module):
    """
    A sequential ensemble where each level's model takes the original 
    text features + the prediction from the previous level
    """
    def __init__(self, input_dim, num_sectors, num_subsectors, num_groups, num_ind, num_nat_ind):
        super(NAICSSequentialEnsemble, self).__init__()
        
        # Level 1: Sector (2-digit)
        # Input: Text features
        self.model_l1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_sectors)
        )
        
        # Level 2: Subsector (3-digit)
        # Input: Text features + Level 1 output
        self.model_l2 = nn.Sequential(
            nn.Linear(input_dim + num_sectors, 512),
            nn.ReLU(),
            nn.Linear(512, num_subsectors)
        )
        
        # Level 3: Industry Group (4-digit)
        self.model_l3 = nn.Sequential(
            nn.Linear(input_dim + num_subsectors, 512),
            nn.ReLU(),
            nn.Linear(512, num_groups)
        )
        
        # Level 4: NAICS Industry (5-digit)
        self.model_l4 = nn.Sequential(
            nn.Linear(input_dim + num_groups, 512),
            nn.ReLU(),
            nn.Linear(512, num_ind)
        )
        
        # Level 5: National Industry (6-digit)
        self.model_l5 = nn.Sequential(
            nn.Linear(input_dim + num_ind, 512),
            nn.ReLU(),
            nn.Linear(512, num_nat_ind)
        )

    def forward(self, text_features):
        # Predict Sector
        logits1 = self.model_l1(text_features)
        
        # Predict Subsector (Concat original features with Level 1 logits)
        # Using logits allows the next model to see the "confidence" of the previous stage
        input2 = torch.cat((text_features, F.softmax(logits1, dim=1)), dim=1)
        logits2 = self.model_l2(input2)
        
        # Predict Industry Group
        input3 = torch.cat((text_features, F.softmax(logits2, dim=1)), dim=1)
        logits3 = self.model_l3(input3)
        
        # Predict NAICS Industry
        input4 = torch.cat((text_features, F.softmax(logits3, dim=1)), dim=1)
        logits4 = self.model_l4(input4)
        
        # Predict National Industry (Final 6-digit)
        input5 = torch.cat((text_features, F.softmax(logits4, dim=1)), dim=1)
        logits5 = self.model_l5(input5)
        
        return [logits1, logits2, logits3, logits4, logits5]

# Example Implementation of the Dictionary Check Logic
# def constrained_inference(model, text_input, naics_trie, top_k=3):
#     model.eval()
#     with torch.no_grad():
#         logits_list = model(text_input)
        
#         final_path = []
#         current_parent = "" # Root of the NAICS tree

#         for level_logits in logits_list:
#             # Get top K predictions for this level
#             probs = F.softmax(level_logits, dim=1)
#             top_probs, top_indices = torch.topk(probs, top_k)
            
#             found_valid = False
#             for i in range(top_k):
#                 candidate_code = str(top_indices[0][i].item()) # Map index to actual NAICS string
                
#                 # Check against your dictionary/trie
#                 if is_valid_child(current_parent, candidate_code, naics_trie):
#                     final_path.append(candidate_code)
#                     current_parent = candidate_code
#                     found_valid = True
#                     break
            
#             if not found_valid:
#                 print("Safety Warning: No valid child found in top-K. Reverting to highest probability anyway.")
#                 final_path.append(str(top_indices[0][0].item()))
        
#         return final_path

# def is_valid_child(parent_code, child_code, trie):
#     # logic to check if child_code starts with parent_code and exists in NAICS
#     return True # Placeholder