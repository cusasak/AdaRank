import torch
import torch.nn as nn
import itertools
from copy import deepcopy
from tqdm import tqdm
from src.models.modeling import ImageEncoder
from src.models.task_vectors import TaskVector
from typing import List, Optional, Union, Dict
import pandas as pd
CPU_DEVICE = "cpu"

class StaticMergeModule(nn.Module):

    def __init__(
        self, config, zero_shot_encoder: ImageEncoder, task_vectors: List[TaskVector]
    ):
        super(StaticMergeModule, self).__init__()
        self.config = config
        self.pretrained_model = zero_shot_encoder

        self.task_vectors = task_vectors

        self.exam_datasets = config.tasks
        self.device = config.device 

        rlambdas = (
            torch.ones(len(task_vectors), len(self.pretrained_model.state_dict()))
            * config.prior
        )
        self.merge_weight = torch.nn.Parameter(rlambdas)


    def _get_truncated_task_vectors(self, rank_ratio: float,
                                    prev_origin: Optional[Dict[str, torch.Tensor]]=None,
                                    new_origin: Optional[Dict[str, torch.Tensor]]=None,
                                    get_decomposed_tv: bool=False
                                    ) -> Union[List[TaskVector], List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:

        move_origin = False
        if prev_origin and new_origin:
            move_origin = True
        
        print(f"Truncating task vectors with rank ratio: {rank_ratio}")
        decomposed_task_vectors = []
        for task_vector in tqdm(self.task_vectors, desc="Truncating task vectors"):
            svd_vector = {}
            for key, value in task_vector.vector.items():
                if ("attn" in key or "mlp" in key) and not (
                    "ln" in key or "bias" in key
                ):  
                    if move_origin:
                        _val = (value + prev_origin[key]  - new_origin[key])
                    else:
                        _val = value
                    
                    U, s, V_T = torch.linalg.svd(
                        _val.to(self.device), full_matrices=False
                    )
                    dim = s.shape[0]
                    
                    if rank_ratio == 0.0:
                        U = torch.zeros_like(U)
                        s = torch.zeros_like(s)
                        V_T = torch.zeros_like(V_T)
                        parsed_dim = 1

                    elif rank_ratio == 1.0:
                        parsed_dim = dim    
                    else:
                        parsed_dim = max(1, int(rank_ratio * dim)) 
                    if get_decomposed_tv:
                        svd_vector[key] = {
                            "U": U[:, :parsed_dim].to(CPU_DEVICE),
                            "s": s[:parsed_dim].to(CPU_DEVICE),
                            "V_T": V_T[:parsed_dim, :].to(CPU_DEVICE),
                        }
                    else:
                        recon = (
                            U[:, :parsed_dim]
                            @ torch.diag(s[:parsed_dim])
                            @ V_T[:parsed_dim, :]
                        )
                        svd_vector[key] = recon.to(CPU_DEVICE)
                    value = value.to(CPU_DEVICE)
                else:
                    if move_origin:
                        svd_vector[key] = (value + prev_origin[key]  - new_origin[key])
  
                    else:
                        svd_vector[key] = value
                    
                    if rank_ratio == 0.0: 
                        svd_vector[key] = torch.zeros_like(svd_vector[key])
                    
            if not get_decomposed_tv:
                _recon_tv = TaskVector(vector=svd_vector)
            else:
                _recon_tv = svd_vector
                
            decomposed_task_vectors.append(_recon_tv)
        
        if get_decomposed_tv:
            return decomposed_task_vectors
        else:
            return decomposed_task_vectors
    
    def elem_whitening(self, m: torch.Tensor):
        u, s, v_t = torch.linalg.svd(m.to(self.device), full_matrices=False)
        return (u @ v_t) 
    
    def whitening_task_vector(self, decomposed_task_vectors: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]])->List[TaskVector]:

        num_tasks = len(decomposed_task_vectors)
            
        weight_keys = list(self.pretrained_model.state_dict().keys())
        
        for each_key in tqdm(weight_keys, desc="Whitening task vectors"):
            is_svd_key = ("attn" in each_key or "mlp" in each_key) and not (
                "ln" in each_key or "bias" in each_key
            )
            if is_svd_key:
                U_list, s_list, V_T_list = [], [], []
                for task_vector in decomposed_task_vectors:
                    U_list.append(task_vector[each_key]["U"])
                    s_list.append(task_vector[each_key]["s"].to(self.device))
                    V_T_list.append(task_vector[each_key]["V_T"])
 
                U_cat = torch.cat(U_list, dim=1)
                U_ortho = self.elem_whitening(U_cat)
                U_ortho_list = torch.chunk(U_ortho, num_tasks, dim=1)
                
                V_T_cat = torch.cat(V_T_list, dim=0)
                V_T_ortho = self.elem_whitening(V_T_cat)
                V_ortho_list = torch.chunk(V_T_ortho, num_tasks, dim=0)

                
                for idx in range(num_tasks):
                    decomposed_task_vectors[idx][each_key] = (U_ortho_list[idx] @ torch.diag(s_list[idx]) @ V_ortho_list[idx]).to(CPU_DEVICE)
            else:
                for idx in range(num_tasks):
                    decomposed_task_vectors[idx][each_key] = (1/num_tasks) * decomposed_task_vectors[idx][each_key].to(CPU_DEVICE)
            
        return [TaskVector(vector=tv) for tv in decomposed_task_vectors]
    
    
    def compare_singular_values(
        self,
        decomposed_task_vectors_avg: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        decomposed_task_vectors_pre: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        k: int = 10
    ):

        results = []

        concat_tv_dict_avg = self._build_concat_matrix(decomposed_task_vectors_avg, mode="avg")
        concat_tv_dict_pre = self._build_concat_matrix(decomposed_task_vectors_pre, mode="pre")

        weight_keys = list(self.pretrained_model.state_dict().keys())
        for layer_name in tqdm(weight_keys, desc="Comparing singular values"):
            is_svd_key = (("attn" in layer_name) or ("mlp" in layer_name)) and not (
                ("ln" in layer_name) or ("bias" in layer_name)
            )
            if not is_svd_key:
                continue
            
            big_mat_avg = concat_tv_dict_avg.get(layer_name, None)
            big_mat_pre = concat_tv_dict_pre.get(layer_name, None)
            
            if big_mat_avg is None or big_mat_pre is None:
                continue

            U_avg, S_avg, Vh_avg = torch.linalg.svd(big_mat_avg.to(self.device), full_matrices=False)
            U_pre, S_pre, Vh_pre = torch.linalg.svd(big_mat_pre.to(self.device), full_matrices=False)
            

            S_avg = S_avg ** 2
            
            sum_all_avg = S_avg.sum()
            sum_topk_avg = S_avg[:k].sum()
            ratio_avg = (sum_topk_avg / sum_all_avg).item()
            
            
            S_pre = S_pre ** 2
            
            sum_all_pre = S_pre.sum()
            sum_topk_pre = S_pre[:k].sum()
            ratio_pre = (sum_topk_pre / sum_all_pre).item()
    
            diff_ratio = ratio_avg - ratio_pre
            
            results.append({
                "layer_name": layer_name,
                "singular_value_ratio_avg": ratio_avg,
                "singular_value_ratio_pre": ratio_pre,
                "diff": diff_ratio
            })
        
        return results
    
 
        
    
    def _build_concat_matrix(
        self,
        decomposed_task_vectors: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]],
        mode: str = "avg"
    ) -> Dict[str, torch.Tensor]:

        num_tasks = len(decomposed_task_vectors)
        weight_keys = list(self.pretrained_model.state_dict().keys())
        
        concat_tv_dict = {}
        
        for each_key in tqdm(weight_keys, desc=f"Building concat matrix ({mode})"):

            is_svd_key = (("attn" in each_key) or ("mlp" in each_key)) and not (
                ("ln" in each_key) or ("bias" in each_key)
            )
            
            if is_svd_key:
                U_list, s_list, V_T_list = [], [], []
                for task_idx in range(num_tasks):
                    task_vec = decomposed_task_vectors[task_idx]
                    
                    U_list.append(task_vec[each_key]["U"])
                    s_list.append(task_vec[each_key]["s"])
                    V_T_list.append(task_vec[each_key]["V_T"])

                U_cat = torch.cat(U_list, dim=1)
                s_cat = torch.cat(s_list, dim=0)
                V_T_cat = torch.cat(V_T_list, dim=0)
                

                big_mat = (U_cat @ torch.diag(s_cat) @ V_T_cat).to(CPU_DEVICE)
                concat_tv_dict[each_key] = big_mat

            else:
                _temp = 0.0
                for idx in range(num_tasks):
                    _temp += (1.0 / num_tasks) * decomposed_task_vectors[idx][each_key].to(CPU_DEVICE)
                concat_tv_dict[each_key] = _temp
        
        return concat_tv_dict


    def _get_origin(self, coeff: float):
        state_dict = deepcopy(self.pretrained_model.state_dict()) 
        processed_tvec = sum(self.task_vectors)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key] + coeff * processed_tvec.vector[key]
            state_dict[key].to(self.device)
        return state_dict

    def _get_tsv_origin(self, rank_ratio: float, merge_coeff: float):
        state_dict = deepcopy(self.pretrained_model.state_dict()) 
        _decomposed_task_vectors = self._get_truncated_task_vectors(
            rank_ratio=rank_ratio,
            prev_origin=None,
            new_origin=None,
            get_decomposed_tv=True,
        )
        _whitened_task_vectors = self.whitening_task_vector(_decomposed_task_vectors)
        processed_tvec = sum(_whitened_task_vectors)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key] + merge_coeff * processed_tvec.vector[key]
    
        return state_dict
    
    def _merge_weights(self, merge_method: str):
        if merge_method == "CART":
            print(
                f"CART merge method with prior: {self.config.prior} and rank ratio: {self.config.initial_rank_ratio}"
            )
            avg_coeff = 1.0 / len(self.config.tasks)
            _theta_avg = self._get_origin(avg_coeff)
            state_dict = deepcopy(self.pretrained_model.state_dict())

            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_avg,
                )
            )  
                
            merge_coeff = self.config.prior
            for key in _theta_avg.keys():
                _theta_avg[key] = (
                    _theta_avg[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )

            self._merged_state_dict = _theta_avg

        elif merge_method in ["TSV", "TSVr"]:
            if merge_method == "TSV":
                rank_ratio = 1.0 / len(self.config.tasks)
            else:
                rank_ratio = self.config.initial_rank_ratio
                
            print(
                f"TSV merge method with prior: {self.config.prior} with {len(self.config.tasks)} tasks, rank ratio: {rank_ratio}"
            )

            self._merged_state_dict = self._get_tsv_origin(rank_ratio, merge_coeff=self.config.prior)

        elif merge_method == "TA":
            print(f"current coefficient: {self.config.prior}")
            self._merged_state_dict = self._get_origin(coeff=self.config.prior)

        elif merge_method == "AVG":
            coeff = 1.0 / len(self.config.tasks) 
            print(f"current coefficient: {coeff} with {len(self.config.tasks)} tasks")
            self._merged_state_dict = self._get_origin(coeff)

        elif merge_method == "Iso_c":
            state_dict = deepcopy(self.pretrained_model.state_dict())
            weight_keys = list(self.pretrained_model.state_dict().keys())
        
            for each_key in tqdm(weight_keys, desc="Merging weights with Iso_c"):
                # is_svd_key = ("attn" in each_key or "mlp" in each_key) and not (
                #     "ln" in each_key or "bias" in each_key
                # )
                shape_ = self.task_vectors[0].vector[each_key].shape
                is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in each_key)
                new_vector = {}
                tvs = [task_vector.vector[each_key].to(self.device) for task_vector in self.task_vectors]
                new_vector[each_key] = sum(tvs) / len(tvs)
                if is_2d_matrix:
                    new_vector[each_key] *= len(tvs)
                    U, s, V_T = torch.linalg.svd(
                        new_vector[each_key], full_matrices=False
                    )
                    s_mean = torch.ones_like(s) * s.mean()
                    new_vector[each_key] = torch.linalg.multi_dot([U, torch.diag(s_mean), V_T])

                state_dict[each_key] = state_dict[each_key].to(self.device) + self.config.prior * new_vector[each_key].to(self.device)
            self._merged_state_dict = state_dict
                    
        elif merge_method == "Iso_CTS":
            state_dict = deepcopy(self.pretrained_model.state_dict())
            weight_keys = list(self.pretrained_model.state_dict().keys())
            
            
            new_vectors = {}
            print("Computing SVD...")
            for each_key in tqdm(weight_keys, desc="Computing SVD for Iso_CTS"):
                # is_svd_key = ("attn" in each_key or "mlp" in each_key) and not (
                #     "ln" in each_key or "bias" in each_key
                # )
                shape_ = self.task_vectors[0].vector[each_key].shape
                is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in each_key)
                if not is_2d_matrix:
                    for i, task_vector in enumerate(self.task_vectors):
                        vec = task_vector.vector[each_key].to(self.device)
                        if i==0:
                            new_vectors[each_key] = vec.clone()
                        else:
                            new_vectors[each_key] += (vec - new_vectors[each_key]) / (i+1)
                    continue

                # print(f"Computing common space using sum from {each_key}...")
                combined_w = sum([tv.vector[each_key].to(self.device) for tv in self.task_vectors])

                ## check task specific size can be equally divided
                common_space_index_s = int(min(shape_) * self.config.common_space_fraction)
                _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(self.exam_datasets)) * len(self.exam_datasets)
                common_space_index_s = min(shape_) - _task_specific_total_space_index_s

                u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
                common_space_u = u[:, :common_space_index_s]
                common_space_s = s[:common_space_index_s]
                common_space_v = v[:common_space_index_s, :]

                ## calculate task specific space
                n_dims_per_task = int((min(shape_) - common_space_index_s) / len(self.exam_datasets))
                for i, task_vector in enumerate(self.task_vectors):
                    w = task_vector.vector[each_key].to(self.device)

                    # calculate the projection onto task specific space to remove the common space
                    w_ts = w - common_space_u @ common_space_u.T @ w
                    u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
                    
                    if i == 0:
                        combined_space_u = torch.zeros_like(u_ts, device=self.device)
                        combined_space_s = torch.zeros_like(s_ts, device=self.device)
                        combined_space_v = torch.zeros_like(v_ts, device=self.device)
                        
                    combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                    combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                    combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
                
                combined_space_u[:, len(self.exam_datasets) * n_dims_per_task : len(self.exam_datasets) * n_dims_per_task + common_space_index_s] = common_space_u
                combined_space_s[len(self.exam_datasets) * n_dims_per_task : len(self.exam_datasets) * n_dims_per_task + common_space_index_s] = common_space_s
                combined_space_v[len(self.exam_datasets) * n_dims_per_task : len(self.exam_datasets) * n_dims_per_task + common_space_index_s, :] = common_space_v
                
                ### Orthogonalize combined_space_u and combined_space_v ###
                # u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
                # u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
                combined_space_u = self.elem_whitening(combined_space_u)
                combined_space_v = self.elem_whitening(combined_space_v)
                
                combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                        
                new_vectors[each_key] = torch.linalg.multi_dot(
                    (
                        combined_space_u,
                        torch.diag(combined_space_s),
                        combined_space_v,
                    )
                )

                state_dict[each_key] = state_dict[each_key].to(self.device) + self.config.prior * new_vectors[each_key].to(self.device)
            
            self._merged_state_dict = state_dict 
        


        elif merge_method == "CART_TSV":
            
            state_dict = deepcopy(self.pretrained_model.state_dict())
            
            print("Compose average task vector")
            avg_coeff = 1.0 / len(self.config.tasks)
            _theta_avg = self._get_origin(avg_coeff)
            
            print(f"Compose whitened task vector with {self.config.initial_rank_ratio} rank ratio, prior: {self.config.prior}")
            _decomposed_task_vectors = self._get_truncated_task_vectors(
                rank_ratio=self.config.initial_rank_ratio,
                prev_origin=state_dict,
                new_origin=_theta_avg,
                get_decomposed_tv=True,
            )
            _whitened_task_vectors = self.whitening_task_vector(_decomposed_task_vectors)
            processed_tvec = sum(_whitened_task_vectors)
            
            merge_coeff = self.config.prior
            for key in _theta_avg.keys():
                _theta_avg[key] = (
                    _theta_avg[key] + merge_coeff * processed_tvec.vector[key]
                )
            
            self._merged_state_dict = _theta_avg
        
        elif merge_method == "pre_CART":
            print(f"{merge_method} merge method with prior: {self.config.prior} with {len(self.config.tasks)} tasks, rank ratio: {self.config.initial_rank_ratio}")
            
            state_dict = deepcopy(self.pretrained_model.state_dict())
            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=None,
                    new_origin=None,
            )
                )
            
            merge_coeff = self.config.prior
            for key in state_dict.keys():
                state_dict[key] = state_dict[key] + merge_coeff * lowrank_processed_tvec.vector[key]
            
            self._merged_state_dict = state_dict
            
            
        elif merge_method == "TA_CART":
            _intial_merge_coeff = self.config.get("initial_merge_coeff", None)
            assert _intial_merge_coeff is not None, "initial merge coefficient is required for TA_CART merge method"

            print(f"{merge_method} merge method with intial merge coefficient: {_intial_merge_coeff} ")
            
            _theta_ta = self._get_origin(_intial_merge_coeff)
            state_dict = deepcopy(self.pretrained_model.state_dict())
            
            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_ta,
                )
            )
            
            print(f"{merge_method} merge method with prior: {self.config.prior} with {len(self.config.tasks)} tasks, rank ratio: {self.config.initial_rank_ratio}")
            merge_coeff = self.config.prior
            for key in _theta_ta.keys():
                _theta_ta[key] = (
                    _theta_ta[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )
            
            self._merged_state_dict = _theta_ta
        
        elif merge_method == "CART_CART":
            _intial_merge_coeff = self.config.get("initial_merge_coeff", None)
            assert _intial_merge_coeff is not None, "initial merge coefficient is required for TA_CART merge method"
            _intial_merge_rank_ratio = self.config.get("intial_merge_rank_ratio", None)
            assert _intial_merge_rank_ratio is not None, "initial merge rank ratio is required for TA_CART merge method"
            
            avg_coeff = 1.0 / len(self.config.tasks)
            _theta_avg = self._get_origin(avg_coeff)
            state_dict = deepcopy(self.pretrained_model.state_dict())
            
            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=_intial_merge_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_avg,
                )
            )
            
            merge_coeff = _intial_merge_coeff
            for key in _theta_avg.keys():
                _theta_avg[key] = (
                    _theta_avg[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )
            
            print(f"{merge_method} merge method with prior: {self.config.prior} with {len(self.config.tasks)} tasks, rank ratio: {self.config.initial_rank_ratio}")
            
            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_avg,
                )
            )
            
            merge_coeff = self.config.prior
            for key in _theta_avg.keys():
                _theta_avg[key] = (
                    _theta_avg[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )
            
            self._merged_state_dict = _theta_avg
            

        elif merge_method == "TSV_CART":
            print(f"{merge_method} merge method with prior: {self.config.prior} with {len(self.config.tasks)} tasks, rank ratio: {self.config.initial_rank_ratio}")
            
            _rank_ratio = 1.0 / len(self.config.tasks)
            _theta_tsv = self._get_tsv_origin(_rank_ratio, merge_coeff=1.0)
            state_dict = deepcopy(self.pretrained_model.state_dict())
            
            lowrank_processed_tvec = sum(
                self._get_truncated_task_vectors(
                    rank_ratio=self.config.initial_rank_ratio,
                    prev_origin=state_dict,
                    new_origin=_theta_tsv,
                )
            )
            
            merge_coeff = self.config.prior
            for key in _theta_tsv.keys():
                _theta_tsv[key] = (
                    _theta_tsv[key] + merge_coeff * lowrank_processed_tvec.vector[key]
                )
                
            self._merged_state_dict = _theta_tsv

        else:
            raise ValueError(f"Invalid merge type: {merge_method}")

    def get_image_encoder(self, merge_method: str):
        self._merge_weights(merge_method)
        clone_model = deepcopy(self.pretrained_model)
        clone_model.load_state_dict(self._merged_state_dict)
        return clone_model

    def forward(self, x):
        raise NotImplementedError("StaticMergeModule does not support forward method.")