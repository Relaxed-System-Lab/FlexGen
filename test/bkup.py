# from copy import deepcopy

# class Home:
#     """ 
#     home of w/x/y (weights & caches) on g/c/d 
    
#     memory view:
#     W: 12 l h^2, X: 2 l b s h_kv, Y: l b s h_d

#     (l, ) *                 (l, b) *
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    + - s - 
#     + -  W(GPU) arrs  - +   + - X(GPU) vecs - - +   + - Y(GPU) vecs - - +    |
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    h
#     + -  W(CPU) arrs  - +   + - X(CPU) vecs - - +   + - Y(CPU) vecs - - +    |
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +      
#     + -  W(Disk) arrs - +   + - X(Disk) vecs  - +   + - Y(Disk) vecs  - +
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +
    
#     where an arr(array) is a vec(vector) with a fixed length in the dim of `s'.
#     More importantly, each arr | vec is with a fixed chunk_size in the dim of `h', 
#     a bunch of arrs | vecs comprises the whole memory view.

#     the gpu arr | vec is based on cuda tensor
#     the cpu arr | vec is based on pinned memory
#     the disk arr | vec is based on np.memmap
#     """
#     def __init__(self, **kwargs):
#         self.policy: Policy = kwargs.get('policy') 
#         self.hf_model = kwargs.get('hf_model')
#         self.layers, self.layers_name = find_module_list(self.hf_model)  
#         self.l = len(self.layers) 

#         self.weights_offload_dir: str = kwargs.get("weights_offload_dir")
#         self.disk_weights_loader: DiskWeightsLoader = DiskWeightsLoader(self.weights_offload_dir)

#         # chunks in h dim
#         self.num_chunks = kwargs.get("num_chunks", 16) # default to 16

#         # {chunk_id : device} dict, by policy
#         self.chunk_id_to_vector_device_dict = self.policy.get_id_to_gcd_dict(num_ids=self.num_chunks)

#         # name: vector(l, *, h // num_chunks) x num_chunks
#         self.w_home: dict[str, list[Vector]] = self.init_w_home() 
#         self.x_home: list[Vector] | None = None # vector(l, 2, b, s_x, h_kv // num_chunks) x num_chunks
#         self.y_home: list[Vector] | None = None # vector(l, b, s_y, h_d // num_chunks) x num_chunks
    
#     def init_w_home(self):
#         # load w files -> g/c/d
#         # return: {w_name: (l, *w_shape) splitted to g/c/d}
#         self.w_home = {}

#         w_tensor_list = [
#             (n, t.to('meta')) for n, t in named_module_tensors(self.layers[0], recurse=True) 
#             if isinstance(t, nn.Parameter)
#         ]

#         for w_name, t_meta in w_tensor_list:
#             # init self.w_home[w_name] as list of vectors
#             self.w_home[w_name] = [None for _ in range(self.num_chunks)]
            
#             # data_shape: list[int], dtype: torch.dtype | np.dtype, device: torch.device | str | int, dim: int, cap: int | None = None, **kwargs
#             assert t_meta.shape[-1] % self.num_chunks == 0
#             h_vector = t_meta.shape[-1] // self.num_chunks
#             data_shape = [self.l] + list(deepcopy(t_meta.shape))
#             data_shape[-1] = h_vector

#             for chunk_id in range(self.num_chunks):
#                 # device
#                 device = self.chunk_id_to_vector_device_dict[chunk_id]

#                 # dtype, file_name
#                 if device == 'disk':
#                     dtype = torch_to_numpy_dtype_dict[t_meta.dtype]
#                     file_name = f'{w_name}-chunk-{chunk_id}' # of l layers aggregated
#                 else:
#                     dtype = t_meta.dtype 
#                     file_name = None 

#                 # dim, cap; randomly select one other than -1 (h's dim)
#                 dim = -2 
#                 cap = data_shape[-2]

#                 # init vector storage
#                 self.w_home[w_name][chunk_id] = Vector(
#                     data_shape=data_shape,
#                     device=device,
#                     dtype=dtype,
#                     dim=dim,
#                     cap=cap,
#                     file_name=file_name, 
#                 )

#             # assign real weight data to vector storage
#             for layer_id in range(self.l):
#                 mmap = self.disk_weights_loader.open_memmap(
#                     key=self.layers_name + f'.{layer_id}.' + w_name
#                 ) 
                
#                 # to n chunks, and assign chunked data to the initialized home
#                 for chunk_id in range(self.num_chunks):
#                     data = mmap[..., h_vector * chunk_id: h_vector * (chunk_id + 1)] # (*, h_vector)

#                     # assign
#                     chunk_vector = self.w_home[w_name][chunk_id] # (l, *, h_vector)
#                     chunk_vector.data[layer_id, ...].copy_(data) 
            

#     def init_x_home(self):
#         ...

#     def init_y_home(self):
#         ...

from copy import deepcopy

class Home:
    """ 
    home of w/x/y (weights & caches) on g/c/d 
    
    memory view:
    W: 12 l h^2, X: 2 l b s h_kv, Y: l b s h_d

    (l, ) *                 (l, b) *
    + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    + - s - 
    + -  W(GPU) arrs  - +   + - X(GPU) vecs - - +   + - Y(GPU) vecs - - +    |
    + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    h
    + -  W(CPU) arrs  - +   + - X(CPU) vecs - - +   + - Y(CPU) vecs - - +    |
    + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +      
    + -  W(Disk) arrs - +   + - X(Disk) vecs  - +   + - Y(Disk) vecs  - +
    + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +
    
    where an arr(array) is a vec(vector) with a fixed length in the dim of `s'.
    More importantly, each arr | vec is with a fixed chunk_size in the dim of `h', 
    a bunch of arrs | vecs comprises the whole memory view.

    the gpu arr | vec is based on cuda tensor
    the cpu arr | vec is based on pinned memory
    the disk arr | vec is based on np.memmap
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs 

        # policy & hf_model
        self.policy: Policy = kwargs.get('policy') 
        self.mp: ModelPrepare = kwargs.get('mp')
        self.hf_model = mp.hf_model
        self.config = mp.config 

        # layers
        self.layers, self.layers_name = find_module_list(self.hf_model)  
        self.l = len(self.layers) 

        # weight files
        self.weights_offload_dir = kwargs.get("weights_offload_dir")
        self.disk_weights_loader = DiskWeightsLoader(self.weights_offload_dir)

        # chunks in h dim
        self.num_chunks = kwargs.get("num_chunks", 16) # default to 16

        self.h_d_num_chunks = kwargs.get("h_d_num_chunks", 16) # default to 16
        
        if isinstance(self.hf_model, OPTForCausalLM):
            self.h_d = self.config.hidden_size
            self.h_kv = self.h_d
        else:
            try:
                self.h_d = self.config.hidden_size
            except:
                raise ValueError(f'{self.hf_model.__class__.__name__}\'s config has no attribute `hidden_size`.')
            
            try:
                self.h_kv = self.config.hidden_size // (self.config.num_attention_heads // self.config.num_key_value_heads)
            except:
                raise ValueError(f'{self.hf_model.__class__.__name__}\'s config has no attribute `hidden_size` or `num_attention_heads` or `num_key_value_heads`.')

        self.h_d_chunk_size = self.h_d // self.h_d_num_chunks
        self.h_kv_chunk_size = self.h_kv // self.h_d_num_chunks

        # {chunk_id : device} dict, by policy
        self.chunk_id_to_vector_device_dict = self.policy.get_id_to_gcd_dict(num_ids=self.num_chunks)

        # name: vector(l, *, h // num_chunks) x num_chunks
        self.w_home: dict[str, list[Vector]] = self.init_w_home() 
        self.x_home: list[Vector] | None = None # vector(l, 2, b, s_x, h_kv // num_chunks) x num_chunks
        self.y_home: list[Vector] | None = None # vector(l, b, s_y, h_d // num_chunks) x num_chunks
    
    def set_chunks(self):
        # w 
        w_tensor_list = [
            (n, t.to('meta')) for n, t in named_module_tensors(self.layers[0], recurse=True) 
            if isinstance(t, nn.Parameter)
        ]  
        for w_name, t_meta in w_tensor_list:
            t_numel = t_meta.numel()
            size_ratio = t_numel / self.h_d 


    def init_w_home(self):
        # load w files -> g/c/d
        # return: {w_name: (l, *w_shape) splitted to g/c/d}
        self.w_home = {}

        w_tensor_list = [
            (n, t.to('meta')) for n, t in named_module_tensors(self.layers[0], recurse=True) 
            if isinstance(t, nn.Parameter)
        ]

        for w_name, t_meta in w_tensor_list:
            # init self.w_home[w_name] as list of vectors
            # {w_name: num_chunks} dict?
            self.w_home[w_name] = [None for _ in range(self.num_chunks)]
            
            # data_shape: list[int], dtype: torch.dtype | np.dtype, device: torch.device | str | int, dim: int, cap: int | None = None, **kwargs
            assert t_meta.shape[-1] % self.num_chunks == 0
            h_vector = t_meta.shape[-1] // self.num_chunks
            data_shape = [self.l] + list(deepcopy(t_meta.shape))
            data_shape[-1] = h_vector

            for chunk_id in range(self.num_chunks):
                # device
                device = self.chunk_id_to_vector_device_dict[chunk_id]

                # dtype, file_name
                if device == 'disk':
                    dtype = torch_to_numpy_dtype_dict[t_meta.dtype]
                    file_name = f'{w_name}-chunk-{chunk_id}' # of l layers aggregated
                else:
                    dtype = t_meta.dtype 
                    file_name = None 

                # dim, cap; randomly select one other than -1 (h's dim)
                dim = -2 
                cap = data_shape[-2]

                # init vector storage
                self.w_home[w_name][chunk_id] = Vector(
                    data_shape=data_shape,
                    device=device,
                    dtype=dtype,
                    dim=dim,
                    cap=cap,
                    file_name=file_name, 
                )

            # assign real weight data to vector storage
            for layer_id in range(self.l):
                mmap = self.disk_weights_loader.open_memmap(
                    key=self.layers_name + f'.{layer_id}.' + w_name
                ) 
                
                # to n chunks, and assign chunked data to the initialized home
                for chunk_id in range(self.num_chunks):
                    data = mmap[..., h_vector * chunk_id: h_vector * (chunk_id + 1)] # (*, h_vector)

                    # assign
                    chunk_vector = self.w_home[w_name][chunk_id] # (l, *, h_vector)
                    chunk_vector.data[layer_id, ...].copy_(data) 
            

    def init_x_home(self):
        ...

    def init_y_home(self):
        ...

