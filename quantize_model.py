# from quantize_static import quantize_model, QLinear
# from utils import save_checkpoint

# import torch
# import torch.nn as nn

# def check_weight_quantization(model):
#     quantized_layers = 0
#     for name, module in model.named_modules():
#         if isinstance(module, QLinear):
#             # print(f"Weights in layer {name} are quantized")
#             quantized_layers += 1
    
#     print(f"Quantized {quantized_layers} layers")

# def check_frozen_parameters(model):
#     frozen_layers = 0
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             # print(f"Frozen parameter: {name}")
#             frozen_layers += 1

#     print(f"Frozen {frozen_layers} layers")

# def main():
#     # hyperparameters
#     DEVICE = 'cuda'
#     TASK_NAMES = ['qnli', 'rte'] # note: mnli needs 3 classes for classification head
#     BATCH_SIZE = 32
#     SAVEPATH = 'trained-models/'
#     NUM_BITS=[16, 8] # weights/activation bit precision

#     # load roberta model
#     from fairseq.models.roberta import RobertaModel
#     model = RobertaModel.from_pretrained('roberta.base').model.to(DEVICE)

#     # add classification head for binary classification (qnli, qqp)
#     model.register_classification_head('sentence_classification_head', 2) # change 2 to 3 for mnli
#     classification_head = model.classification_heads['sentence_classification_head']

#     save_checkpoint(model, SAVEPATH, 'fp-roberta-base-2classhead.pth')

#     # freeze backbone weights, enable classification head weights
#     for param in model.parameters():
#         param.requires_grad = False

#     for param in classification_head.parameters():
#         param.requires_grad = True

#     save_checkpoint(model, SAVEPATH, 'fp-roberta-base-2classhead-freeze_backbone.pth')

#     print(model)
#     check_weight_quantization(model)
#     check_frozen_parameters(model)


#     # quantization
#     for num_bit in NUM_BITS:
#         print("\n\n\n")
#         print(f"{num_bit}-bit quantization")

#         # quantize nn.Linear layers, specify which layers to exclude
#         quantized_model = quantize_model(model, num_bits=num_bit, device=DEVICE)

#         for param in quantized_model.parameters():
#             param.requires_grad = False

#         for param in classification_head.parameters():
#             param.requires_grad = True

#         save_checkpoint(quantized_model, SAVEPATH, f'w{num_bit}a{num_bit}-roberta-base-2classhead-freeze_backbone.pth')

#         print(quantized_model)
#         check_weight_quantization(quantized_model)
#         check_frozen_parameters(quantized_model)
#         print("\n")

# if __name__ == "__main__":
#     main()