import torch 

def get_size_info(obj): # recursive
    if isinstance(obj, tuple):
        return tuple(get_size_info(o) for o in obj)
    elif isinstance(obj, list):
        return list(get_size_info(o) for o in obj)
    elif isinstance(obj, dict):
        return {k:get_size_info(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.size()
    elif isinstance(obj, (int, bool, type(None))): 
        return obj
    else:
        raise NotImplementedError(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')

def get_kth_batch_inputs(inputs, k, gpu_batch_size): # for both args, kwargs
    if isinstance(inputs, tuple): # e.g. args
        return tuple(get_kth_batch_inputs(inp, k, gpu_batch_size) for inp in inputs)
    elif isinstance(inputs, list): 
        return list(get_kth_batch_inputs(inp, k, gpu_batch_size) for inp in inputs)
    elif isinstance(inputs, dict): # e.g. kwargs
        return {key:get_kth_batch_inputs(value, k, gpu_batch_size) for key, value in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs[k * gpu_batch_size:(k + 1) * gpu_batch_size]
    elif isinstance(inputs, (int, bool, type(None))): # None, int, bool
        return inputs
    else:
        raise NotImplementedError(f'inputs: {inputs} of type \'{type(inputs)}\' is not implemented.')

def concat_outputs(outputs): # concat K outputs to one output
    assert len(outputs), 'empty outputs.'
    assert isinstance(outputs[0], (torch.Tensor, tuple)), f'Only supports layer output type of torch.Tensor or tuple. However, we get a {type(outputs[0])}.'
    
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    elif isinstance(outputs[0], tuple):
        def f(outputs):
            ans = []
            for elem in zip(*outputs):
                if isinstance(elem[0], torch.Tensor):
                    ans.append(torch.cat(elem, dim=0))
                elif isinstance(elem[0], tuple):
                    ans.append(f(elem))
                elif isinstance(elem[0], (int, bool, type(None))): 
                    ans.append(elem[0])
            return tuple(ans)

        return f(outputs)
            
