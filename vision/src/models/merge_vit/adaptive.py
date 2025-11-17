import torch
import torch.nn as nn
import functools
from copy import deepcopy
from torch.nn.utils import stateless
from src.models.heads import get_classification_head

from tqdm import tqdm

CPU_DEVICE = "cpu"


class AdaMergingModule(nn.Module):
    '''
    Module for Adaptive Merging
    '''

    def __init__(self, config, zero_shot_encoder, task_vectors):
        super(AdaMergingModule, self).__init__()
        self.config = config
        # Average(CART) or pretrained model(TA)
        self.origin = zero_shot_encoder
        self.exam_datasets = config.tasks
        self.clamp_weights = getattr(self.config, "clamp_weights", False)
        self.extend_clamp = getattr(self.config, "extend_clamp", False)
        self.soft_mask = getattr(self.config, "soft_mask", False)

        self.svd_keys = []
        self.normalized_merging_weights = getattr(
            self.config, "normalized_merging_weights", False)
        self.device = config.device

        if config.initial_rank_ratio == 0:
            config.initial_rank_ratio = 1 / len(task_vectors)

        if config.merge_method not in ["Iso_CTS"]:
            if config.merge_method == "TSV":
                self.svd_list = self._svd_tsv(task_vectors, config.initial_rank_ratio)
            else:
                self.svd_list = self._svd_vanilla(task_vectors)
            rlambdas = torch.ones(len(task_vectors), len(
                self.origin.state_dict())) * config.prior
            self.merge_weight = torch.nn.Parameter(rlambdas)

            self.merge_mask = nn.ParameterList(self._mask_init(task_vectors))
        else:
            self.svd_list = self._svd_iso_cts(task_vectors)
            rlambdas = torch.ones(len(task_vectors)+1, len(
                self.origin.state_dict())) * config.prior
            self.merge_weight = torch.nn.Parameter(rlambdas)
            # self.merge_mask = nn.ParameterList(self._mask_init_iso_cts(task_vectors))
            mm, cm = self._mask_init_iso_cts(task_vectors)
            self.merge_mask = nn.ParameterList(mm)
            self.common_mask = nn.ParameterList(cm)

        self.mask_temp = config.mask_temp
        # import ipdb; ipdb.set_trace()
        
        print("Module initialized with initial rank ratio:", config.initial_rank_ratio)
        print("Module initialized with merge method:", config.merge_method)

        self.classifier_names = []
        for dataset_name in self.exam_datasets:
            classification_head = get_classification_head(
                self.config, dataset_name)
            layer_name = f"classifier_{dataset_name}"
            self.add_module(layer_name, classification_head.to(self.device))
            self.classifier_names.append(layer_name)

        self._overall_requires_grad()
        self._merged_state_dict = None

    def straight_through_mask(self, mat_svd, mask):
        U, s, V_T = mat_svd
        s_masked = mask * s + (((mask > 0.5).float() - mask) * s).detach()
        return (U * s_masked) @ V_T

    def soft_mask_func(self, mat_svd, mask):
        U, s, V_T = mat_svd
        s_masked = mask * s
        return (U * s_masked) @ V_T

    def _overall_requires_grad(self):
        """
        Only merge_weight and merge_mask require gradients.
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
            if "merge_weight" in name or "merge_mask" in name or "common_mask" in name:
                param.requires_grad = True

    def forward(self, x, dataset_name):
        if self._merged_state_dict is None:
            if self.config.merge_method == "Iso_CTS":
                self._merge_weights_iso_cts()
            else:
                self._merge_weights()
        features = self.forward_model(x, dataset=None, args=None)
        layer_name = f"classifier_{dataset_name}"
        classification_head = getattr(self, layer_name)
        out = classification_head(features)
        return out

    def forward_model(self, inp, dataset=None, args=None):
        partial_functional_call = functools.partial(
            stateless.functional_call,
            self.origin,
            self._merged_state_dict,
        )
        return partial_functional_call((inp, dataset, args))

    def _merge_weights(self):
        origin_state = {key: value.detach().clone()
                        for key, value in self.origin.state_dict().items()}
        state_dict = origin_state

        if self.clamp_weights:
            if self.extend_clamp:
                layer_wise_weight = self.merge_weight.clamp(-0.5, 2)
            else:
                layer_wise_weight = self.merge_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.merge_weight
        if self.normalized_merging_weights:
            layer_wise_weight = layer_wise_weight.softmax(dim=0)

        for task_idx, (weight, each_task_vector) in enumerate(zip(layer_wise_weight, self.svd_list)):
            for w, m, (key, value) in zip(weight, self.merge_mask, each_task_vector.items()):
                # shape_ = value.shape
                # is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
                # # if ('attn' in key or 'mlp' in key) and not ('ln' in key or 'bias' in key):
                # if key not in self.svd_keys and is_2d_matrix:
                #     self.svd_keys.append(key)
                if key in self.svd_keys:
                    _val = value

                    if self.soft_mask:
                        task_vector = self.soft_mask_func(
                            _val, (m[task_idx] / self.mask_temp).sigmoid())
                    else:
                        task_vector = self.straight_through_mask(
                            _val, (m[task_idx] / self.mask_temp).sigmoid())
                else:
                    task_vector = value
                state_dict[key].add_(w * task_vector)
        self._merged_state_dict = state_dict

    def get_image_encoder(self):
        if self._merged_state_dict is None:
            if self.config.merge_method == "Iso_CTS":
                self._merge_weights_iso_cts()
            else:
                self._merge_weights()
        clone_model = deepcopy(self.origin)
        clone_model.load_state_dict(self._merged_state_dict)
        return clone_model

    def get_classification_head(self, dataset_name):
        return getattr(self, f"classifier_{dataset_name}")

    def reset_merged_state(self):
        self._merged_state_dict = None

    def _get_origin(self, task_vectors):

        if self.config.merge_method == "CART":
            state_dict = deepcopy(self.origin.state_dict())
            coeff = 1.0 / len(self.exam_datasets)
            processed_tvec = sum(task_vectors)
            for key in state_dict.keys():
                state_dict[key] = state_dict[key] + \
                    coeff * processed_tvec.vector[key]
            return state_dict
        else:
            return deepcopy(self.origin.state_dict())

    def _svd_vanilla(self, task_vectors):
        svd_list = []
        for idx, task_vector in tqdm(enumerate(task_vectors), total=len(task_vectors), desc="Decompose task vectors"):
            svd_vector = {}
            for key, value in tqdm(task_vector.vector.items(), desc=f"Decompose task {idx}"):
                shape_ = value.shape

                is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
                if key not in self.svd_keys and is_2d_matrix:
                    self.svd_keys.append(key)
                if key in self.svd_keys:
                    _val = value
                    ret = torch.linalg.svd(_val, full_matrices=False)
                    svd_vector[key] = ret
                else:
                    svd_vector[key] = value.to(self.device)
            svd_list.append(svd_vector)
        return svd_list
    
    def _svd_tsv(self, task_vectors, rank_ratio):
        num_tasks = len(task_vectors)
        svd_list = [{} for _ in range(num_tasks)]
        weight_keys = list(self.origin.state_dict().keys())
        
        for each_key in weight_keys:
            # is_svd_key = ("attn" in each_key or "mlp" in each_key) and not (
            #     "ln" in each_key or "bias" in each_key
            # )
            shape_ = task_vectors[0].vector[each_key].shape
            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in each_key)
            if is_2d_matrix:
                self.svd_keys.append(each_key)
                U_list, s_list, V_T_list = [], [], []
                for task_vector in task_vectors:
                    U, s, V_T = torch.linalg.svd(
                        task_vector.vector[each_key].to(self.device), full_matrices=False
                    )
                    dim = s.shape[0]
                    parsed_dim = max(1, int(rank_ratio * dim))
                    U_list.append(U[:, :parsed_dim])
                    s_list.append(s[:parsed_dim])
                    V_T_list.append(V_T[:parsed_dim, :])
                
                U_cat = torch.cat(U_list, dim=1)
                U_ortho = self.elem_whitening(U_cat)
                U_ortho_list = torch.chunk(U_ortho, num_tasks, dim=1)
                
                V_T_cat = torch.cat(V_T_list, dim=0)
                V_T_ortho = self.elem_whitening(V_T_cat)
                V_ortho_list = torch.chunk(V_T_ortho, num_tasks, dim=0)
                for idx in range(num_tasks):
                    svd_list[idx][each_key] = (U_ortho_list[idx], s_list[idx], V_ortho_list[idx])
            else:
                for idx in range(num_tasks):
                    svd_list[idx][each_key] = (1 / num_tasks) * task_vectors[idx].vector[each_key]
        
        return svd_list

    def elem_whitening(self, m):
        u, s, v_t = torch.linalg.svd(m.to(self.device), full_matrices=False)
        return (u @ v_t) 
    
    def _svd_iso_cts(self, task_vectors):
        num_tasks = len(task_vectors)
        svd_list = [{} for _ in range(num_tasks+1)]
        weight_keys = list(self.origin.state_dict().keys())
        for each_key in weight_keys:
            # is_svd_key = ("attn" in each_key or "mlp" in each_key) and not (
            #     "ln" in each_key or "bias" in each_key
            # )
            shape_ = task_vectors[0].vector[each_key].shape
            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in each_key)
            if is_2d_matrix:
                self.svd_keys.append(each_key)
                tvs = [task_vector.vector[each_key].to(self.device) for task_vector in task_vectors]
                new_vector = sum(tvs)

                common_space_index_s = int(min(shape_) * self.config.common_space_fraction)
                _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(self.exam_datasets)) * len(self.exam_datasets)
                common_space_index_s = min(shape_) - _task_specific_total_space_index_s

                U, s, V_T = torch.linalg.svd(
                    new_vector, full_matrices=False
                )
                common_space_u = U[:, :common_space_index_s]
                common_space_s = s[:common_space_index_s]
                common_space_v = V_T[:common_space_index_s, :]

                n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
                for idx, task_vector in enumerate(task_vectors):
                    w = task_vector.vector[each_key].to(self.device)
                    w_ts = w - common_space_u @ common_space_u.T @ w
                    u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
                    
                    if idx == 0:
                        combined_space_u = torch.zeros_like(u_ts, device=self.device)
                        combined_space_s = torch.zeros_like(s_ts, device=self.device)
                        combined_space_v = torch.zeros_like(v_ts, device=self.device)
                
                    combined_space_u[:, idx * n_dims_per_task : (idx + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                    combined_space_s[idx * n_dims_per_task : (idx + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                    combined_space_v[idx * n_dims_per_task : (idx + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
                
                combined_space_u[:, num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_u
                combined_space_s[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_s
                combined_space_v[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s, :] = common_space_v
                
                ### Orthogonalize combined_space_u and combined_space_v ###
                # u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
                # u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
                combined_space_u = self.elem_whitening(combined_space_u)
                combined_space_v = self.elem_whitening(combined_space_v)
                combined_space_s = torch.ones_like(combined_space_s, device=self.device) * combined_space_s.mean()

                for idx in range(num_tasks):
                    svd_list[idx][each_key] = (
                        combined_space_u[:, idx * n_dims_per_task : (idx + 1) * n_dims_per_task],
                        combined_space_s[idx * n_dims_per_task : (idx + 1) * n_dims_per_task],
                        combined_space_v[idx * n_dims_per_task : (idx + 1) * n_dims_per_task, :])
                svd_list[num_tasks][each_key] = (
                    combined_space_u[:, num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s],
                    combined_space_s[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s],
                    combined_space_v[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s, :]
                )
            else:
                for idx in range(num_tasks):
                    svd_list[idx][each_key] = (1 / num_tasks) * task_vectors[idx].vector[each_key]
                svd_list[num_tasks][each_key] = 0.0
        return svd_list

    def _svd_with_truncation_ratio(self, task_vectors, rank_ratio):
        svd_list = []
        for task_vector in task_vectors:
            svd_vector = {}
            for key, value in task_vector.vector.items():
                shape_ = value.shape
                is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
                if key not in self.svd_keys and is_2d_matrix:
                    self.svd_keys.append(key)
                if key in self.svd_keys:
                    _val = value.to(self.device)
                    U, s, V_T = torch.linalg.svd(_val, full_matrices=False)
                    full_dim = min(_val.shape[0], _val.shape[1])
                    truncated_dim = max(1, int(rank_ratio * full_dim))
                    U_truncated = U[:, :truncated_dim]
                    s_truncated = s[:truncated_dim]
                    V_T_truncated = V_T[:truncated_dim, :]
                    svd_vector[key] = (U_truncated, s_truncated, V_T_truncated)
                else:
                    svd_vector[key] = value.to(self.device)
            svd_list.append(svd_vector)
        return svd_list

    def _mask_init(self, task_vectors):
        # self.svd_keys = []
        merge_mask = []
        for key, value in self.origin.state_dict().items():
            # shape_ = value.shape
            # is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
            if key in self.svd_keys:
                # self.svd_keys.append(key)
                
                full_dim = min(value.shape[0], value.shape[1])
                if self.config.merge_method == "TSV":
                    dim = max(1, int(full_dim * self.config.initial_rank_ratio))
                else:
                    dim = full_dim
                if self.soft_mask:
                    mask = 2.0 * \
                        torch.ones(len(task_vectors), dim,
                                    dtype=torch.float32)
                else:
                    mask = 0.1 * \
                        torch.ones(len(task_vectors), dim,
                                    dtype=torch.float32)
                if (self.config.initial_rank_ratio < 1.0) and (self.config.merge_method != "TSV"):
                    preserved_dim = int(
                        dim * self.config.initial_rank_ratio)
                    if self.soft_mask:
                        mask[:, preserved_dim:] = 0.0
                    else:
                        mask[:, preserved_dim:] = -0.1
                merge_mask.append(torch.nn.Parameter(mask, requires_grad=True))
            else:
                merge_mask.append(torch.nn.Parameter(torch.ones(1)))
        return merge_mask

    def _mask_init_iso_cts(self, task_vectors):
        merge_mask = []
        common_mask = []
        num_tasks = len(task_vectors)

        for key, value in self.origin.state_dict().items():
            if key in self.svd_keys:
                full_dim = min(value.shape[0], value.shape[1])
                common_space_index_s = int(
                    full_dim * self.config.common_space_fraction)
                _task_specific_total_space_index_s = round(
                    (full_dim - common_space_index_s) / num_tasks) * num_tasks
                
                common_space_index_s = full_dim - _task_specific_total_space_index_s
                n_dims_per_task = int(
                    (full_dim - common_space_index_s) / num_tasks)
                
                mask = 0.1 * \
                        torch.ones(len(task_vectors), n_dims_per_task,
                                    dtype=torch.float32)
                merge_mask.append(torch.nn.Parameter(mask, requires_grad=True))
                
                ## common_svd
                common_masks = 0.1 * torch.ones(1, common_space_index_s, dtype=torch.float32)
                common_mask.append(torch.nn.Parameter(common_masks, requires_grad=True))
            else:
                merge_mask.append(torch.nn.Parameter(torch.ones(1)))
                ## common_svd
                common_mask.append(torch.nn.Parameter(torch.ones(1)))
        # return merge_mask
        return merge_mask, common_mask
    
    def _merge_weights_iso_cts(self):
        origin_state = {key: value.detach().clone()
                        for key, value in self.origin.state_dict().items()}
        state_dict = origin_state
        if self.clamp_weights:
            if self.extend_clamp:
                layer_wise_weight = self.merge_weight.clamp(-0.5, 2)
            else:
                layer_wise_weight = self.merge_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.merge_weight
        if self.normalized_merging_weights:
            layer_wise_weight = layer_wise_weight.softmax(dim=0)
        
        for task_idx, (weight, each_task_vector) in enumerate(zip(layer_wise_weight, self.svd_list)):
            if task_idx != len(self.exam_datasets):
                for w, m, (key, value) in zip(weight, self.merge_mask, each_task_vector.items()):
                    if key in self.svd_keys:
                        _val = value

                        if self.soft_mask:
                            task_vector = self.soft_mask_func(
                                _val, (m[task_idx].to(self.device) / self.mask_temp).sigmoid())
                        else:
                            # import ipdb; ipdb.set_trace()
                            task_vector = self.straight_through_mask(
                                _val, (m[task_idx].to(self.device) / self.mask_temp).sigmoid())
                    else:
                        task_vector = value / len(self.exam_datasets)
                    state_dict[key].add_(w * task_vector)
            else:
                for w,c, (key, value) in zip(weight, self.common_mask, each_task_vector.items()):
                    if key in self.svd_keys:
                        # U, s, V_T = value
                        # _val = (U * s) @ V_T
                        # task_vector = _val
                        # common_svd
                        _val= value
                        # import ipdb; ipdb.set_trace()
                        # print(key, c.shape, _val[0].shape, _val[1].shape, _val[2].shape)
                        task_vector = self.straight_through_mask(
                            _val, (c.to(self.device) / self.mask_temp).sigmoid())
                        state_dict[key].add_(w * task_vector)
                    else:
                        # task_vector = value
                        continue
                    

        self._merged_state_dict = state_dict