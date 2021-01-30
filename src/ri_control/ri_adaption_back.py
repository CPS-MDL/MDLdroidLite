import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ours
from utils import target_std_scale_calculator, test
from grow.neuron_grow import device
from grow.growth_utils import find_modules, structure_model
from utils import device

#################################
# torch.manual_seed(1535)
np.random.seed(1535)
# copy baseline
random_num = []
# ours approach
cosine_rank = []
# rank baseline
importance_rank = []
flatten_factor = 16


#################################


def grow(model, growth_list, mode='rank_cosine', ff=16, inputs=None, target=None, features_output=None, test_loader=None, criterion=None):
    global flatten_factor
    flatten_factor = ff

    for growth in growth_list:
        if growth[1] == 0:
            continue
        growth_index, growth_num = growth
        layers = find_modules(model)
        is_con = isinstance(layers[growth_index], torch.nn.Conv2d)
        # print('Before: ', end='')
        # test(model, test_loader=test_loader, criterion=criterion, to_log=None,
        #      is_two_out=True)
        model = change_model(model, mode=mode, layer_index=growth_index, incremental_num=growth_num, inputs=inputs, target=target,
                             is_con=is_con)
        model = model.to(device)
        # print('After: ', end='')
        # test(model, test_loader=test_loader, criterion=criterion, to_log=None,
        #      is_two_out=True)
        # gen_gradient(model, criterion, inputs, target)

    return model


def change_model(model, mode='rankgroup', layer_index=0, incremental_num=2, inputs=None, target=None, is_con=True):
    if incremental_num == 0:
        return model

    mode_list = ['bridging']
    bn_layers = []

    if mode not in mode_list and is_con:
        n_model = grow_filters(model, layer_index=layer_index, incremental_num=incremental_num, mode=mode)
        if len(bn_layers) > 0:
            n_model = replace_bn_layers(n_model, index=layer_index, incremental_num=incremental_num)

    elif mode == 'bridging' and is_con:
        if len(bn_layers) > 0:
            model = replace_bn_layers(model, index=layer_index, incremental_num=incremental_num)
        n_model = grow_one_filter(model, layer_index=layer_index, incremental_num=incremental_num, data=inputs,
                                  target=target)

    else:
        if mode == 'bridging':
            features_output, _ = model(inputs)
        else:
            features_output = None
        n_model = grow_neuron(model, layer_index, features_output, incremental_num=incremental_num, mode=mode)

    return n_model


def grow_one_filter(model, layer_index, incremental_num, data, target, num=10):
    layers = find_modules(model)

    old_layer = layers[layer_index]
    old_layer2 = layers[layer_index + 1]

    struct = structure_model(model)
    _, first_layer_index = struct[layer_index]
    _, second_layer_index = struct[layer_index + 1]

    losses = []
    first_layers = []
    second_layers = []

    # start to compare each model with different layer
    # this function need to random check them ten times to get two best layers by comparing loss values.
    for i in range(num):
        if isinstance(old_layer2, torch.nn.Conv2d):
            new_first_layer, new_second_layer = create_layers(old_layer, old_layer2, incremental_num)
        else:
            new_first_layer, new_second_layer = create_layers(old_layer, old_layer2, incremental_num)

        model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)
        loss = test_batch(model, data, target)
        first_layers.append(new_first_layer)
        second_layers.append(new_second_layer)
        losses.append(loss)

    min_loss = losses[0]
    index = 0
    for i, l in enumerate(losses):
        if l < min_loss:
            min_loss = l
            index = i
    new_first_layer = first_layers[index]
    new_second_layer = second_layers[index]

    # then return the model which has the lowest loss value
    model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)

    return model


def test_batch(model, data, target):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        _, output = model(data)
        test_loss = criterion(output, target)  # sum up batch loss
    return test_loss.item()


# def replace_multiple_layers(model, first_layer, first_index, second_layer, second_index):
#     if isinstance(second_layer, torch.nn.Conv2d):
#         features = torch.nn.Sequential(
#             *(replace_layers(model.features, i, [first_index, second_index],
#                              [first_layer, second_layer]) for i, _ in enumerate(model.features)))
#         model.features = features
#         return model
#     else:
#         features = torch.nn.Sequential(
#             *(replace_layers(model.features, i, [first_index],
#                              [first_layer]) for i, _ in enumerate(model.features)))
#         classifier = torch.nn.Sequential(
#             *(replace_layers(model.classifier, i, [second_index],
#                              [second_layer]) for i, _ in enumerate(model.classifier)))
#         model.features = features
#         model.classifier = classifier
#         return model


def weight_adaption_copy_n(old_layer, incremental_num, is_first=True, is_multi=False, multi=16):
    global random_num
    global flatten_factor
    multi = flatten_factor
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None

    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num

    numerator = 2

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            random_list = np.random.choice(old_c_out, incremental_num)  # incremental_num to 1
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
            random_num = random_list
        else:
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            random_list = random_num
            for i, r in enumerate(random_list):
                weights[:, i] = old_weights[:, r] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            random_list = np.random.choice(old_c_out, incremental_num)
            for i, r in enumerate(random_list):
                weights[i] = old_weights[r]
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
            random_num = random_list
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
        else:
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            random_list = random_num
            for i, r in enumerate(random_list):
                start = r * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))

    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def get_rank(result, ascending=False):
    result = np.asarray(result)
    if not ascending:
        result = result * -1
    order = result.argsort()
    rank = order.tolist()
    return rank


def get_filter_rank(old_layer, is_first=True, ascending=False):
    weights_tensor = old_layer.weight.data.cpu().numpy()

    grad_tensor = old_layer.weight.grad.data.cpu().numpy()
    s_tensor = grad_tensor * weights_tensor
    if isinstance(old_layer, torch.nn.Conv2d):
        if is_first:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=(1, 2, 3)))
        else:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=(0, 2, 3)))
    elif isinstance(old_layer, torch.nn.Linear):
        if is_first:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=1))
        else:
            result = np.sqrt(np.sum(np.abs(s_tensor), axis=0))
    rank = get_rank(result, ascending=ascending)
    return rank


# weights adaption
def weight_adaption(old_layer, incremental_num=1, generate_mode='random', is_first=True, is_multi=False):
    if generate_mode == 'random':
        weight, bias = weight_adaption_random(old_layer, incremental_num, is_first=is_first)

    elif generate_mode == 'rank_baseline':
        weight, bias = weight_adaption_rank_baseline(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'copy_n':
        weight, bias = weight_adaption_copy_n(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)

    elif generate_mode == 'rank_cosine':

        weight, bias = weight_adaption_rank_cosine(old_layer, incremental_num, is_first=is_first, is_multi=is_multi)
    else:
        raise Exception('The generation mode is not found')

    return weight, bias


def weights_divide_first(old_weights, incremental_num, multi=1, mode='rankconnect'):
    global cosine_rank
    global random_num
    global importance_rank

    for i in range(incremental_num):
        if mode == 'rank_baseline':
            numerator = 1 / 2
            order = importance_rank[i]
        start = order * multi
        end = start + multi
        old_weights[start:end] = old_weights[start:end] * numerator
    if mode == 'rank_one':
        seed = 2
        order = importance_rank[0]
        start = order * multi
        end = start + multi
        old_weights[start:end] = old_weights[start:end] / seed

    return old_weights


def weights_divide(old_weights, incremental_num, is_multi=False, multi=16, twoD=False, mode='rankgroup'):
    global cosine_rank
    global random_num
    global importance_rank
    global flatten_factor
    multi = flatten_factor

    if not is_multi:
        multi = 1
    incremental_num = incremental_num // multi

    numerator = 1 - (incremental_num / (old_weights.shape[1] + incremental_num))
    numerator = numerator / 2

    for i in range(incremental_num):
        if mode == 'rank_one':
            break
        if mode == 'rankgroup':
            order = cosine_rank[i]
        # elif mode == 'randomMap':
        #     order = random_num
        #     seed = 2
        elif mode == 'copy_n':
            seed = 2
            order = random_num[i]
        elif mode == 'rank_baseline':
            order = importance_rank[i]
            numerator = 1 / 2
        else:
            raise Exception('the generation mode is not found')
        start = order * multi
        end = start + multi
        # old_weights[:, start:end] = old_weights[:, start:end] / seed
        old_weights[:, start:end] = old_weights[:, start:end] * numerator

    if mode == 'rank_one':
        seed = 2
        order = importance_rank[0]
        start = order * multi
        end = start + multi
        old_weights[:, start:end] = old_weights[:, start:end] / seed

    return old_weights


def conv_layer_grow(old_conv_layer, incremental_num, is_first=True, mode='rank'):
    if old_conv_layer is None:
        raise BaseException("No Conv layer found in classifier")
    old_weights = old_conv_layer.weight.data.cpu().numpy()
    old_c_out, old_c_in, k_h, k_w = old_weights.shape
    mode_list = ['random']

    if mode not in mode_list:
        weights, bias = weight_adaption(old_conv_layer, incremental_num, generate_mode=mode, is_first=is_first)
        old_grad = old_conv_layer.weight.grad.data.cpu().numpy()

    # create new_conv_layer
    if is_first:
        # if the old weight is also needed to change
        f_mode_list = ['rank_baseline', 'rank_one', 'rank_baseline_scale']  # , 'rank_cumulative'
        if mode in f_mode_list:
            old_weights = weights_divide_first(old_weights, incremental_num, mode=mode)
        new_conv_layer = torch.nn.Conv2d(in_channels=old_conv_layer.in_channels,
                                         out_channels=old_conv_layer.out_channels + incremental_num,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))

        # the first layer need to change the bias

        # weights
        new_weights = new_conv_layer.weight.data.cpu().numpy()
        new_weights[:old_c_out, :old_c_in, :, :] = old_weights

        if mode not in mode_list:
            new_weights[old_c_out:, :, :, :] = weights
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_c_out, :old_c_in, :, :] = old_grad

    else:
        m_list = ['rank_baseline', 'copy_n']
        if mode in m_list:
            old_weights = weights_divide(old_weights, incremental_num, twoD=False, mode=mode)

        new_conv_layer = torch.nn.Conv2d(in_channels=old_conv_layer.in_channels + incremental_num,
                                         out_channels=old_conv_layer.out_channels,
                                         kernel_size=old_conv_layer.kernel_size,
                                         stride=old_conv_layer.stride,
                                         padding=old_conv_layer.padding,
                                         dilation=old_conv_layer.dilation,
                                         groups=old_conv_layer.groups,
                                         bias=(old_conv_layer.bias is not None))
        # copy the weights from old_linear_layer , the new filter use init weight
        new_weights = new_conv_layer.weight.data.cpu().numpy()
        new_weights[:old_c_out, :old_c_in, :, :] = old_weights

        if mode not in mode_list:
            new_weights[:, old_c_in:, :, :] = weights
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_c_out, :old_c_in, :, :] = old_grad


    # copy new weights to new_conv_layer
    new_conv_layer.weight.data = torch.from_numpy(new_weights)
    new_conv_layer.weight.data = new_conv_layer.weight.data.to(device)

    # copy old_grad to new_conv_layer
    if mode not in mode_list:
        grad = old_conv_layer.weight.grad
        grad.data = torch.from_numpy(new_grad).to(device)
        new_conv_layer.weight.grad = grad

    return new_conv_layer


def fc_layer_grow(old_linear_layer, incremental_num, weights=None, is_first=True, mode='rank',
                  is_multi=False):
    if old_linear_layer is None:
        raise BaseException("No linear layer found in classifier")

    old_weights = old_linear_layer.weight.data.cpu().numpy()
    old_out, old_in = old_weights.shape
    mode_list = ['random', 'bridging']
    if weights is None:

        weights, bias = weight_adaption(old_linear_layer, incremental_num, generate_mode=mode, is_first=is_first,
                                        is_multi=is_multi)
    else:
        _, bias = weight_adaption(old_linear_layer, incremental_num, is_first=is_first)

    if mode not in mode_list:
        old_grad = old_linear_layer.weight.grad.data.cpu().numpy()

    # create new_linear_layer
    if is_first:
        f_mode_list = ['rank_baseline']
        if mode in f_mode_list:
            old_weights = weights_divide_first(old_weights, incremental_num, mode=mode)
        new_linear_layer = torch.nn.Linear(old_linear_layer.in_features,
                                           old_linear_layer.out_features + incremental_num)
        # copy the weights from old_linear_layer and add grow weights
        if weights is not None:
            # if mode not in mode_list:
            new_weights = np.vstack((old_weights, weights)).astype(np.float32)
        else:
            new_weights = new_linear_layer.weight.data.cpu().numpy()
            new_weights[:old_out, :] = old_weights

        # copy the bias and old grad to new layer
        if mode not in mode_list:
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:old_out, :] = old_grad
    else:
        new_linear_layer = torch.nn.Linear(old_linear_layer.in_features + incremental_num,
                                           old_linear_layer.out_features)
        m_list = ['rank_baseline', 'copy_n']  # , 'rank_one', 'randomMap', 'copy_one'
        if mode in m_list:
            old_weights = weights_divide(old_weights, incremental_num, is_multi, twoD=True, mode=mode)
        # copy the weights from old_linear_layer and add grow weights
        if weights is not None:
            # if mode not in mode_list:
            new_weights = np.hstack((old_weights, weights)).astype(np.float32)
        else:
            new_weights = new_linear_layer.weight.data.cpu().numpy()
            new_weights[:, :old_in] = old_weights
        # copy the old grad to new layers
        # if mode not in mode_list and mode != 'bridging':
        if mode not in mode_list:
            new_grad = np.zeros(new_weights.shape).astype(np.float32)
            new_grad[:, :old_in] = old_grad

    out_channel, in_channel = new_linear_layer.weight.data.cpu().numpy().shape

    # assert the new_weights's shape must be same
    assert new_weights.shape[0] == out_channel and new_weights.shape[
        1] == in_channel, 'the new weights shape in linear layer 1 is incorrect'

    # copy new weights to new_linear_layer
    new_linear_layer.weight.data = torch.from_numpy(new_weights)
    new_linear_layer.weight.data = new_linear_layer.weight.data.to(device)
    # copy old grad  to new_linear_layer
    # if mode not in mode_list and mode != 'bridging':
    if mode not in mode_list:
        grad = old_linear_layer.weight.grad
        grad.data = torch.from_numpy(new_grad).to(device)
        new_linear_layer.weight.grad = grad

    return new_linear_layer


def create_layers(first_layer, second_layer, incremental_num, mode='random'):
    new_first_layer = conv_layer_grow(first_layer, incremental_num, mode='random')
    if isinstance(second_layer, torch.nn.Conv2d):
        new_second_layer = conv_layer_grow(second_layer, incremental_num, is_first=False, mode='random')
    else:
        new_second_layer = fc_layer_grow(second_layer, incremental_num * 16, weights=None,
                                         is_first=False, mode='random')
    return new_first_layer, new_second_layer


def grow_neuron(model, layer_index, output, incremental_num, mode='rank'):
    # the first linear layer need to be changed
    # find old Linear layer
    layers = find_modules(model)
    old_linear_layer = layers[layer_index]
    old_linear_layer2 = layers[layer_index + 1]

    if mode == 'bridging':
        # get w_in, w_out

        w_in, w_out = weight_in_out(model, layer_index, output, incremental_num)
        new_linear_layer = fc_layer_grow(old_linear_layer, incremental_num, weights=w_in, mode='bridging')
    else:
        new_linear_layer = fc_layer_grow(old_linear_layer, incremental_num, mode=mode)

    if mode == 'bridging':
        new_linear_layer2 = fc_layer_grow(old_linear_layer2, incremental_num, weights=w_out, is_first=False,
                                          mode='bridging')
    else:
        new_linear_layer2 = fc_layer_grow(old_linear_layer2, incremental_num, is_first=False, mode=mode)

    struct = structure_model(model)
    _, first_layer_index = struct[layer_index]
    _, second_layer_index = struct[layer_index + 1]

    model = replace_multiple_layers(model, new_linear_layer, first_layer_index, new_linear_layer2, second_layer_index)

    return model


def weight_in_out(model, layer_index, output, incremental_num, a=0.3, ):
    layers = find_modules(model)
    current_layer = layers[layer_index]
    next_layer = layers[layer_index + 1]

    previous = output.data.cpu().numpy()
    previous_output = np.mean(previous, axis=0)

    gradient = next_layer.weight.grad.data.cpu().numpy()
    next_gradient = np.mean(gradient, axis=1)
    next_weights = next_layer.weight.data.cpu().numpy()
    current_weights = current_layer.weight.data.cpu().numpy()

    n = previous_output.shape[0]
    m = next_gradient.shape[0]
    gradient_matrix = np.zeros((n, m))
    for row in range(n):
        for col in range(m):
            gradient_matrix[row, col] = previous_output[row] * next_gradient[col]
    gradient_matrix_abs = np.sqrt(np.abs(gradient_matrix))
    sgn_matrix = np.sign(gradient_matrix)
    random_matrix = np.random.uniform(-1, 1, n * m)
    random_matrix = np.reshape(random_matrix, (n, m))
    g_matrix = gradient_matrix_abs * random_matrix

    # calculate w_in, w_out
    w_in = np.sum(g_matrix * sgn_matrix, axis=1)
    w_out = np.sum(g_matrix, axis=0)

    # multiply the a
    w_out = w_out * a * numpy_avg(np.abs(next_weights)) / numpy_avg(np.abs(w_out))
    w_in = w_in * a * numpy_avg(np.abs(current_weights)) / numpy_avg(np.abs(w_in))

    w_in = np.reshape(w_in, (1, -1))
    w_out = np.reshape(w_out, (-1, 1))

    # w_in, w_out need to further process
    # w_in and w_out times incremental numbers
    w_in = multi_weights(w_in, incremental_num, is_first=True)
    w_out = multi_weights(w_out, incremental_num, is_first=False)

    return w_in, w_out


def numpy_avg(np_array, axis=None):
    np_array[np.where(np_array == 0)] = np.nan
    if axis is not None:
        output = np.nanmean(np_array, axis=axis)
    else:
        output = np.nanmean(np_array)
    return output


def replace_layers(model_classifer, layer_index, layer_index_change, layers_change):
    # if the layer in the layer_index is needed to changed, then use layers_change to replace,
    # otherwise return the original ones.
    if layer_index in layer_index_change:
        return layers_change[layer_index_change.index(layer_index)]
    return model_classifer[layer_index]


def multi_weights(weights, incremental_num, is_first=True):
    new_weights = weights
    if is_first:
        dim = 0
    else:
        dim = 1
    for i in range(incremental_num - 1):
        new_weights = np.concatenate((new_weights, weights), axis=dim)
    return new_weights


def replace_bn_layers(model, index, incremental_num):
    dic_layers = find_bn_layers(model)
    current_layer = dic_layers[index][1]
    new_num_features = current_layer.num_features + incremental_num
    current_bias = current_layer.bias.data.cpu().numpy()
    current_weight = current_layer.weight.data.cpu().numpy()

    new_layer = nn.BatchNorm2d(
        num_features=new_num_features,
        eps=current_layer.eps,
        momentum=current_layer.momentum,
        affine=current_layer.affine,
        track_running_stats=current_layer.track_running_stats
    )
    new_weight = np.ones((new_num_features))
    new_bias = np.zeros((new_num_features))
    new_weight[0:current_layer.num_features] = current_weight
    new_bias[0:current_layer.num_features] = current_bias
    new_layer.bias.data = torch.from_numpy(new_bias.astype(np.float32))
    new_layer.weight.data = torch.from_numpy(new_weight.astype(np.float32))
    new_layer.bias.data = new_layer.bias.data.to(device)
    new_layer.weight.data = new_layer.weight.data.to(device)
    model.features[dic_layers[index][0]] = new_layer
    return model


def put_to_ndarray(small_nd, large_nd):
    if small_nd.ndim == 4:
        s_out, s_in, k_h, k_w = small_nd.shape
        large_nd[:s_out, :s_in, :k_h, :k_w] = small_nd
    elif small_nd.ndim == 2:
        s_out, s_in = small_nd.shape
        large_nd[:s_out, :s_in] = small_nd
    elif small_nd.ndim == 1:
        s_out = small_nd.shape[0]
        large_nd[:s_out] = small_nd
    return large_nd


# def find_bn_layers(model):
#     layers = {}
#     for index_str, module in model.features._modules.items():
#         if isinstance(module, torch.nn.BatchNorm2d):
#             layers[int(index_str)] = module
#     return layers


def find_bn_layers(model):
    layers = []
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.BatchNorm2d):
            layers.append((int(index_str), module))
    return layers


def grow_filters_old(model, layer_index, incremental_num, mode='rank'):
    global flatten_factor
    first_layer_index = 0
    old_conv_layer = None
    # find old conv layer
    for index_str, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(index_str) == layer_index:
            old_conv_layer = module
            break
        first_layer_index = first_layer_index + 1

    # the second conv layer need to be changed
    second_layer_index = 0
    old_conv_layer2 = None
    old_linear_layer = None
    for layer, module in model.features._modules.items():
        if isinstance(module, torch.nn.Conv2d) and int(layer) > first_layer_index:
            old_conv_layer2 = module
            second_layer_index = int(layer)
            break

    if old_conv_layer2 is None:
        for layer, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                second_layer_index = int(layer)
                break

    new_first_layer = conv_layer_grow(old_conv_layer, incremental_num, is_first=True, mode=mode)

    if old_conv_layer2 is not None:
        new_second_layer = conv_layer_grow(old_conv_layer2, incremental_num, is_first=False, mode=mode)
    else:
        new_second_layer = fc_layer_grow(old_linear_layer, incremental_num * flatten_factor, is_first=False, mode=mode,
                                         is_multi=True)

    model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)

    # if old_conv_layer2 is None
    # create a new fc1 layer

    # above code need to be in a function and return two layers and layer index.

    # this function need to random check them ten times to get two best layers by comparing loss values.

    return model


def replace_multiple_layers(model, first_layer, first_index, second_layer, second_index):
    if isinstance(first_layer, torch.nn.Conv2d) and isinstance(second_layer, torch.nn.Conv2d):
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index, second_index],
                             [first_layer, second_layer]) for i, _ in enumerate(model.features)))
        model.features = features
        return model
    elif isinstance(first_layer, torch.nn.Conv2d) and isinstance(second_layer, torch.nn.Linear):
        features = torch.nn.Sequential(
            *(replace_layers(model.features, i, [first_index],
                             [first_layer]) for i, _ in enumerate(model.features)))
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [second_index],
                             [second_layer]) for i, _ in enumerate(model.classifier)))
        model.features = features
        model.classifier = classifier
    elif isinstance(first_layer, torch.nn.Linear) and isinstance(second_layer, torch.nn.Linear):
        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [first_index, second_index],
                             [first_layer, second_layer]) for i, _ in enumerate(model.classifier)))
        model.classifier = classifier

    return model


def grow_filters(model, layer_index, incremental_num, mode='rank'):
    global flatten_factor
    layers = find_modules(model)
    old_layer = layers[layer_index]
    old_layer2 = layers[layer_index + 1]

    new_first_layer = conv_layer_grow(old_layer, incremental_num, is_first=True, mode=mode)

    if isinstance(old_layer2, torch.nn.Conv2d):
        new_second_layer = conv_layer_grow(old_layer2, incremental_num, is_first=False, mode=mode)
    else:
        new_second_layer = fc_layer_grow(old_layer2, incremental_num * flatten_factor, is_first=False, mode=mode,
                                         is_multi=True)

    struct = structure_model(model)
    _, first_layer_index = struct[layer_index]
    _, second_layer_index = struct[layer_index + 1]

    model = replace_multiple_layers(model, new_first_layer, first_layer_index, new_second_layer, second_layer_index)

    # if old_conv_layer2 is None
    # create a new fc1 layer

    # above code need to be in a function and return two layers and layer index.

    # this function need to random check them ten times to get two best layers by comparing loss values.

    return model


def weight_adaption_random(old_layer, incremental_num, is_first=True, val_min=-0.1, val_max=0.1):
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = torch.empty(incremental_num, old_c_in, k_h, k_w)
            weights.uniform_(val_min, val_max)
            new_bias_needed = torch.empty(incremental_num)
            new_bias_needed.uniform_(val_min, val_max)
        else:
            weights = torch.empty(old_c_out, incremental_num, k_h, k_w)
            weights.uniform_(val_min, val_max)

    else:
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = torch.empty(incremental_num, old_c_in)
            weights.uniform_(val_min, val_max)
            new_bias_needed = torch.empty(incremental_num)
            new_bias_needed.uniform_(val_min, val_max)
        else:
            weights = torch.empty(old_c_out, incremental_num)
            weights.uniform_(val_min, val_max)

    weights = weights.numpy()
    if new_bias_needed is not None:
        new_bias_needed = new_bias_needed.numpy()

    return weights, new_bias_needed


def insert_rank(rank):
    if not hasattr(insert_rank, "list"):
        insert_rank.list = []
    if len(insert_rank.list) == 3:
        insert_rank.list = []
    insert_rank.list.append(rank)


def weight_adaption_rank_cosine(old_layer, incremental_num, is_first=True, multi=16, is_multi=False, asc=True):
    global flatten_factor
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias = None
    cosine_array = []
    lambda_list = []
    global cosine_rank

    s_index = 0 if is_first else 1
    r = 1 - (incremental_num / (old_weights.shape[s_index] + incremental_num))

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            n_tensor = torch.zeros(incremental_num + old_c_out, old_c_in, k_h, k_w)
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :, :, :]
            x_weight = np.reshape(x_weight, (old_c_in, k_h * k_w))
            for index in range(old_c_out):
                y_weight = old_weights[index, :, :, :]
                y_weight = np.reshape(y_weight, (old_c_in, k_h * k_w))
                cosine_array.append(cosine_similarity(x_weight, y_weight)[0][0])
            c_rank = get_rank(cosine_array)
            # weights
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank[order]]
            # bias
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            # for order in range(incremental_num):
            #     new_bias[order] = old_bias[c_rank.index(order)]

            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            n_tensor = torch.zeros(old_c_out, incremental_num + old_c_in, k_h, k_w)
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for order in range(incremental_num):
                weights[:, order, :, :] = old_weights[:, cosine_rank[order], :, :]

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            n_tensor = torch.zeros(incremental_num + old_c_out, old_c_in)
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=asc)
            rank1_index = rank[0]
            x_weight = old_weights[rank1_index, :]
            for index in range(old_c_out):
                y_weight = old_weights[index, :]
                top = np.sum(x_weight * y_weight)
                bottom = np.sqrt(np.sum(x_weight ** 2)) * np.sqrt(np.sum(x_weight ** 2))
                cosine_array.append(top / bottom)
            c_rank = get_rank(cosine_array)
            # weight
            for order in range(incremental_num):
                weights[order] = old_weights[c_rank[order]]

            # bias
            new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            # for order in range(incremental_num):
            #     new_bias[order] = old_bias[c_rank.index(order)]

            # new_bias = np.random.uniform(-0.1, 0.1, incremental_num)
            cosine_rank = c_rank
            insert_rank(c_rank)
        else:
            multi = flatten_factor
            n_tensor = torch.zeros(old_c_out, incremental_num + old_c_in)
            if not is_multi:
                multi = 1
            weights = np.zeros((old_c_out, incremental_num))
            num_needed = incremental_num // multi
            for i in range(num_needed):
                start = cosine_rank[i] * multi
                end = start + multi
                start_i = i * multi
                end_i = start_i + multi
                weights[:, start_i:end_i] = old_weights[:, start:end]

    scale = target_std_scale_calculator(n_tensor, ratio=r)
    # scale = target_scale_calculator(n_tensor)

    old_scale = np.std(old_weights)
    # old_scale = max(np.abs(np.min(old_weights)), np.abs(np.max(old_weights)))

    numerator = scale / old_scale
    # old_scale_weight = max(np.abs(np.min(old_weights)), np.abs(np.max(old_weights)))
    # w = scale / old_scale_weight

    if is_first:
        lambda_list.append(numerator)
    weights = weights * numerator
    size_needed = np.random.uniform(-0.1, 0.1, weights.shape) * numerator
    base_weight = size_needed + weights
    # base_weight = weights
    print('STD: old={:2.10f}| new={:2.10f}| scale={:2.10f}'.format(old_scale, scale, numerator))
    # print('W: old={:2.10f}| new={:2.10f}| scale={:2.10f}'.format(old_scale_weight, scale, w))
    if new_bias is not None:
        new_bias = new_bias * numerator
    return base_weight, new_bias


def weight_adaption_rank_baseline(old_layer, incremental_num, is_first=True, is_multi=False, ascending=False):
    global importance_rank
    global flatten_factor
    old_weights = old_layer.weight.data.cpu().numpy()
    new_bias_needed = None
    # if divide_num == 'non-scale':
    #     numerator = 1
    # else:
    #     numerator = incremental_num if divide_num != 2 else divide_num
    numerator = 2

    if isinstance(old_layer, torch.nn.Conv2d):
        old_c_out, old_c_in, k_h, k_w = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in, k_h, k_w))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in, k_h, k_w))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num, k_h, k_w))
            for i in range(1, incremental_num):
                index = rank[i]
                weights[:, i] = old_weights[:, index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num, k_h, k_w))

    elif isinstance(old_layer, torch.nn.Linear):
        old_c_out, old_c_in = old_weights.shape
        if is_first:
            weights = np.zeros((incremental_num, old_c_in))
            rank = get_filter_rank(old_layer, ascending=ascending)
            importance_rank = rank
            for i in range(0, incremental_num):
                index = rank[i]
                weights[i] = old_weights[index] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (incremental_num, old_c_in))
            new_bias_needed = np.random.uniform(-0.1, 0.1, incremental_num)
        else:
            rank = importance_rank
            weights = np.zeros((old_c_out, incremental_num))
            if is_multi:
                multiplier = flatten_factor
            else:
                multiplier = 1
            for i in range(0, incremental_num // multiplier):
                index = rank[i]
                start = index * multiplier
                end = start + multiplier
                start_i = i * multiplier
                end_i = start_i + multiplier
                weights[:, start_i:end_i] = old_weights[:, start:end] / numerator
            size_needed = np.random.uniform(-0.1, 0.1, (old_c_out, incremental_num))
    base_weight = size_needed + weights
    return base_weight, new_bias_needed


def gen_gradient(model, criterion, inputs, target):
    feature_out, output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
