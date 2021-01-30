import torch
#from grow.neuron_grow import grow_filters, grow_one_neuron, replace_layers,grow_one_filter


def replace_multiple_layers(model, indexs, layers):
    layers = torch.nn.Sequential(
        *(replace_layers(model.layers, i, indexs,
                         layers) for i, _ in enumerate(model.layers)))
    model.layers = layers
    return model


def grow_layers(model, incremental_num, data=None, target=None, weights=None, mode_name='rankgroup'):
    for i, num in enumerate(incremental_num):
        if i == 0 and num != 0:
            model = grow_filters(model, layer_index=0, incremental_num=num, mode=mode_name)
        if i == 1 and num != 0:
            model = grow_filters(model, layer_index=3, incremental_num=num, mode=mode_name)
        if i == 2 and num != 0:
            model = grow_one_neuron(model, output=None, incremental_num=num, mode=mode_name)
    return model

#
# def create_layers(first_layer, second_layer, incremental_num):
#     if isinstance(first_layer, torch.nn.Conv2d):
#         new_first_layer = conv_layer_grow(first_layer, incremental_num)
#     else:
#         new_first_layer = fc_layer_grow(first_layer, incremental_num, weights=None)
#
#     if isinstance(second_layer, torch.nn.Conv2d):
#         new_second_layer = conv_layer_grow(second_layer, incremental_num, is_first=False)
#     else:
#         if isinstance(first_layer, torch.nn.Conv2d):
#             incremental_num = incremental_num * 16
#         new_second_layer = fc_layer_grow(second_layer, incremental_num, weights=None,
#                                          is_first=False)
#     return new_first_layer, new_second_layer
